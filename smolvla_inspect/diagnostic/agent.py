from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import numpy as np

from .models import (
    DiagnosticReport, DiagnosticMatrix, Hypothesis, CounterfactualResult,
    EvidenceEntry, Finding, SceneSegmentation, DatasetDiversityReport, Anomaly,
)
from .registry import REGISTRY
from .scene import parse_task_objects, detect_objects, segment_scene, analyze_dataset_diversity
from .regions import attribute_to_regions
from .matrix import build_diagnostic_matrix, detect_anomalies
from .spatial_object import (
    build_spatial_object_diagnosis,
    choose_target_object,
    summarize_spatial_object_diagnosis,
)
from .semantic_probe import (
    build_qk_probe_report,
    build_semantic_probe_report,
    probe_semantic_frame,
    semantic_candidate_labels,
    summarize_qk_probe,
    summarize_semantic_probe,
)
from .prompts import (
    build_hypothesis_prompt, build_synthesis_prompt, build_triage_selection_prompt,
    format_diversity_summary, format_anomalies_json, format_hypotheses_with_results,
)
from ..data import _resolve_task_string, get_episode_frames, build_policy_batch_from_sample


# ---------------------------------------------------------------------------
# Robust JSON extraction — handles common LLM output quirks
# ---------------------------------------------------------------------------

def _find_matching_bracket(text: str, start: int) -> int:
    """Find the matching ] for a [ at position start, respecting string literals."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return i
    return -1


def _extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM output, repairing common formatting issues.

    Handles: markdown code fences, trailing commas, unescaped newlines inside
    string values, single quotes, // comments, and control characters.
    """
    # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*\n?", "", text)

    # 2. Locate the outermost [ ... ] using bracket matching
    start = text.find("[")
    if start < 0:
        # No array found — check if it's a single JSON object (e.g. from response_format: json_object)
        obj_start = text.find("{")
        if obj_start >= 0:
            # Try to parse as a single object or as a JSON object containing an array
            try:
                obj = json.loads(text[obj_start:])
                if isinstance(obj, dict):
                    # Check if any value is a list of dicts (e.g. {"hypotheses": [...]})
                    for v in obj.values():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            return v
                    # Otherwise wrap single object as a list
                    return [obj]
            except json.JSONDecodeError:
                pass
            # Try individual object extraction
            return _parse_objects_individually(text[obj_start:])
        return []

    # Try bracket-matched end first (handles narrative text after JSON)
    end = _find_matching_bracket(text, start)
    if end < 0:
        # Fallback to rfind
        end = text.rfind("]")
    if end <= start:
        return []
    blob = text[start:end + 1]

    # 3. Try parsing as-is first (fast path)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass

    # 3b. If bracket matching failed, try rfind as fallback
    end_rfind = text.rfind("]")
    if end_rfind != end and end_rfind > start:
        blob_rfind = text[start:end_rfind + 1]
        try:
            return json.loads(blob_rfind)
        except json.JSONDecodeError:
            pass
        blob = blob_rfind  # Use the longer blob for repair

    # 4. Repair pass
    # Remove single-line // comments
    blob = re.sub(r"//[^\n]*", "", blob)
    # Remove C-style /* ... */ comments
    blob = re.sub(r"/\*.*?\*/", "", blob, flags=re.DOTALL)
    # Replace single quotes used as string delimiters with double quotes
    # (only outside of already-double-quoted strings)
    blob = _single_to_double_quotes(blob)
    # Remove trailing commas before } or ]
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    # Escape literal newlines inside string values
    blob = _escape_newlines_in_strings(blob)
    # Strip control characters (except \n \r \t)
    blob = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", blob)

    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass

    # 5. Last resort: try to parse individual objects with per-object repair
    return _parse_objects_individually(blob)


def _single_to_double_quotes(s: str) -> str:
    """Replace single-quoted JSON strings with double-quoted ones."""
    result = []
    i = 0
    in_double = False
    while i < len(s):
        ch = s[i]
        if ch == '"' and (i == 0 or s[i - 1] != '\\'):
            in_double = not in_double
            result.append(ch)
        elif ch == "'" and not in_double:
            result.append('"')
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _escape_newlines_in_strings(s: str) -> str:
    """Escape literal newlines that appear inside JSON string values."""
    result = []
    in_string = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '"' and (i == 0 or s[i - 1] != '\\'):
            in_string = not in_string
            result.append(ch)
        elif ch == '\n' and in_string:
            result.append('\\n')
        elif ch == '\r' and in_string:
            result.append('\\r')
        elif ch == '\t' and in_string:
            result.append('\\t')
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _parse_objects_individually(blob: str) -> list[dict]:
    """Try to extract individual JSON objects from a malformed array."""
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(blob):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                fragment = blob[start:i + 1]
                # Apply the same repairs to each fragment
                fragment = re.sub(r",\s*}", "}", fragment)
                fragment = _escape_newlines_in_strings(fragment)
                try:
                    objects.append(json.loads(fragment))
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


class DiagnosticAgent:
    """Orchestrates the full diagnostic pipeline."""

    def __init__(self, policy, dataset, episode_idx: int, image_key: str,
                 device: str, llm_settings: dict, config: dict,
                 existing_run_dir: str | None = None,
                 image_map=None):
        self.policy = policy
        self.dataset = dataset
        self.episode_idx = episode_idx
        self.image_key = image_key
        self.device = device
        self.llm_settings = llm_settings  # {"provider": "anthropic", "model": "...", "api_key": "...", "base_url": ""}
        self.config = config
        self.existing_run_dir = existing_run_dir
        self.image_map = image_map
        self.evidence_log: list[EvidenceEntry] = []
        self.post_hoc = existing_run_dir is not None

    async def run(self, progress_callback=None) -> DiagnosticReport:
        """Run the full diagnostic pipeline."""

        _PHASE_LABELS = {
            "scene_understanding": ("1/9", "Scene Understanding"),
            "dataset_diversity":   ("2/9", "Dataset Diversity"),
            "triage":              ("3/9", "Signal Triage"),
            "matrix":              ("4/9", "Diagnostic Matrix"),
            "representation":      ("5/9", "Representation Probes"),
            "hypothesize":         ("6/9", "Hypothesis Formation"),
            "counterfactuals":     ("7/9", "Counterfactual Tests"),
            "iteration":           ("7/9", "Follow-up Iteration"),
            "disambiguation":      ("8/9", "Spatial vs Object Diagnosis"),
            "synthesis":           ("9/9", "Report Synthesis"),
            "complete":            ("OK", "Complete"),
        }
        _run_start = time.time()
        _phase_start = [_run_start]  # mutable so inner fn can update
        _current_phase = [""]

        def _progress(phase: str, detail: str = ""):
            if progress_callback:
                progress_callback(phase, detail)

            now = time.time()
            elapsed_total = now - _run_start

            # Print phase transition header when phase changes
            if phase != _current_phase[0]:
                # Print elapsed time for previous phase
                if _current_phase[0]:
                    phase_elapsed = now - _phase_start[0]
                    prev_label = _PHASE_LABELS.get(_current_phase[0], ("", _current_phase[0]))[1]
                    print(f"  {'':>5}  done ({phase_elapsed:.1f}s)")

                _current_phase[0] = phase
                _phase_start[0] = now
                step, label = _PHASE_LABELS.get(phase, ("?", phase))
                print(f"\n  [{step}] {label}  ({elapsed_total:.0f}s elapsed)")
                print(f"  {'':>5}  {'-' * 40}")

            # Print detail line
            print(f"  {'':>5}  {detail}", flush=True)

        # Get a sample frame for scene understanding
        sample = self.dataset[self._get_first_frame_idx()]
        task_string = _resolve_task_string(sample, self.dataset)

        # ── Phase 1: Scene Understanding ────────────────────────
        _progress("scene_understanding", "Parsing task objects...")
        object_queries = parse_task_objects(task_string)
        _progress("scene_understanding", f"Objects to find: {object_queries}")

        first_frame = self._get_first_frame_image(sample)

        _progress("scene_understanding", "Running object detection...")
        detections = detect_objects(first_frame, object_queries, device=self.device,
                                    confidence_threshold=self.config.get("detection_confidence", 0.1))

        _progress("scene_understanding", "Running segmentation...")
        scene = segment_scene(first_frame, detections, device=self.device)
        _progress("scene_understanding", f"Found {len(scene.objects)} objects, {len(scene.region_names())} regions")
        target_object = choose_target_object(scene, object_queries)
        if target_object:
            _progress("scene_understanding", f"Primary manipulation target: {target_object}")
        else:
            _progress("scene_understanding", "Primary manipulation target: unavailable")

        self.evidence_log.append(EvidenceEntry(
            phase="scene_understanding",
            primitive_name="scene.segment_scene",
            data={
                "regions": scene.region_names(),
                "num_objects": len(scene.objects),
                "target_object": target_object,
            },
        ))

        # ── Phase 2: Dataset Diversity Analysis ─────────────────
        _progress("dataset_diversity", "Analyzing dataset diversity...")
        diversity = None
        try:
            diversity = analyze_dataset_diversity(
                self.dataset, self.image_key, task_string,
                num_episodes=self.config.get("diversity_episodes", 20),
                frames_per_episode=self.config.get("diversity_frames_per_episode", 3),
                device=self.device,
            )
            _progress("dataset_diversity", f"Sampled {diversity.num_episodes_sampled} episodes")
        except Exception as e:
            _progress("dataset_diversity", f"Skipped: {e}")

        # ── Phase 3a: Cheap Triage — Always-run Signals ─────────
        _progress("triage", "Collecting cheap model signals...")
        if self.post_hoc:
            signals = self._load_existing_signals()
        else:
            signals = self._run_cheap_triage(sample)

        _progress("triage", f"Cheap signals: {list(signals.keys())}")

        # ── Phase 3b: Mandatory expensive signals (always run) ──
        _mandatory_expensive = ["vision_vs_state"]
        if not self.post_hoc and self.policy is not None:
            for sig_name in _mandatory_expensive:
                _progress("triage", f"  Running {sig_name} (mandatory)...")
                try:
                    self._run_expensive_signal(sig_name, sample, signals, scene=scene)
                except Exception as e:
                    _progress("triage", f"  {sig_name} failed: {e}")

        # ── Phase 3c: Adaptive Triage — LLM-selected Expensive Signals ──
        if not self.post_hoc and self.policy is not None:
            selected = await self._select_expensive_signals(
                task_string, scene, signals)
            # Remove any that were already run as mandatory
            selected = [s for s in selected if s not in _mandatory_expensive]
            _progress("triage", f"LLM selected expensive signals: {selected}")

            for sig_name in selected:
                _progress("triage", f"  Running {sig_name}...")
                try:
                    self._run_expensive_signal(sig_name, sample, signals, scene=scene)
                except Exception as e:
                    _progress("triage", f"  {sig_name} failed: {e}")

        _progress("triage", f"All signals: {list(signals.keys())}")

        # ── Phase 4: Build Diagnostic Matrix ────────────────────
        _progress("matrix", "Building diagnostic matrix...")
        # Use all frames that were collected during triage (not just the first)
        attn_heatmaps = signals.get("attention")
        n_signal_frames = len(attn_heatmaps) if attn_heatmaps else 1
        frames_for_matrix = list(range(n_signal_frames))

        matrix = build_diagnostic_matrix(
            frames=frames_for_matrix,
            segmentation=scene,
            attention_heatmaps=signals.get("attention"),
            gradcam_siglip_heatmaps=signals.get("gradcam_siglip"),
            cross_attention_heatmaps=signals.get("cross_attention"),
            saliency_heatmaps=signals.get("saliency"),
            gradcam_connector_heatmaps=signals.get("gradcam_connector"),
            per_action_dim_maps=signals.get("per_action_dim"),
            vision_vs_state=signals.get("vision_vs_state"),
            positional_baseline=signals.get("positional_baseline"),
            language_diff=signals.get("language_diff"),
            temporal_trajectories=signals.get("temporal_trajectories"),
            occlusion_map=signals.get("occlusion_map"),
            connector_analysis=signals.get("connector_analysis"),
            dataset_diversity=diversity,
        )

        internals = signals.get("model_internals")
        anomalies = detect_anomalies(matrix, internals, dataset_diversity=diversity)
        n_crit = sum(1 for a in anomalies if a.severity == "critical")
        n_warn = sum(1 for a in anomalies if a.severity == "warning")
        _progress("matrix", f"Detected {len(anomalies)} anomalies ({n_crit} critical, {n_warn} warnings)")
        for a in anomalies:
            sev_icon = {"critical": "!!", "warning": "! ", "info": "  "}.get(a.severity, "  ")
            _progress("matrix", f"  {sev_icon} {a.type}: {a.description[:80]}")

        self.evidence_log.append(EvidenceEntry(
            phase="matrix",
            primitive_name="matrix.detect_anomalies",
            data={"anomaly_types": [a.type for a in anomalies]},
        ))

        primary_semantic_frame = None
        semantic_internal = None
        semantic_probe = None
        qk_probe = None
        if self.policy is not None and target_object is not None:
            _progress("representation", "Running semantic patch-to-text probe on the primary frame...")
            try:
                primary_semantic_frame, semantic_internal = probe_semantic_frame(
                    self.policy,
                    sample,
                    self.dataset,
                    self.image_key,
                    self.device,
                    target_object=target_object,
                    candidate_labels=semantic_candidate_labels(scene, target_object),
                    attention_heatmap=(signals.get("attention") or [None])[0],
                    gradcam_map=(signals.get("gradcam_siglip") or [None])[0],
                    image_map=self.image_map,
                    frame_id="primary",
                )
                semantic_probe = build_semantic_probe_report(target_object, primary_semantic_frame, [])
                if semantic_probe is not None:
                    _progress("representation", summarize_semantic_probe(semantic_probe))
            except Exception as e:
                _progress("representation", f"Semantic probe skipped: {e}")
        else:
            _progress("representation", "Skipping semantic probe (no model or target object)")

        preliminary_spatial_object = build_spatial_object_diagnosis(
            matrix,
            target_object=target_object,
            cf_results=[],
            dataset_diversity=diversity,
            semantic_probe=semantic_probe,
        )

        # ── Phase 5: LLM Hypothesis Formation ──────────────────
        llm_label = self.llm_settings.get("provider", "rule-based")
        if not self.llm_settings.get("api_key"):
            llm_label = "rule-based (no API key)"
        _progress("hypothesize", f"Forming hypotheses via {llm_label}...")
        hypotheses = await self._form_hypotheses(
            task_string, scene, diversity, matrix, anomalies,
            preliminary_spatial_object, semantic_probe, None)
        _progress("hypothesize", f"Formed {len(hypotheses)} hypotheses:")
        for h in hypotheses:
            test_label = f" -> test: {h.test_type}" if h.test_type != "none" else " (no test needed)"
            _progress("hypothesize", f"  {h.id} [{h.confidence:.0%}] {h.description[:70]}{test_label}")

        # ── Phase 6: Run Agent-Chosen Counterfactuals ───────────
        cf_results: list[CounterfactualResult] = []
        max_cf = self.config.get("max_counterfactuals", 3)
        cf_skip_reason = ""  # tracks why counterfactuals weren't run

        if max_cf <= 0:
            _progress("counterfactuals", "Skipping (--skip-counterfactuals or max_counterfactuals=0)")
            cf_skip_reason = "skipped"
            for h in hypotheses:
                if h.test_type != "none":
                    h.confidence *= 0.6
        elif self.policy is not None:
            for hypothesis in hypotheses:
                if hypothesis.test_type == "none":
                    continue
                if len(cf_results) >= max_cf:
                    cf_skip_reason = "budget_exhausted"
                    break

                _progress("counterfactuals",
                          f"[{len(cf_results)+1}/{max_cf}] Running {hypothesis.test_type} for {hypothesis.id}...")
                try:
                    result = self._run_counterfactual(
                        hypothesis, sample, scene)
                    cf_results.append(result)
                    verdict = "CONFIRMED" if result.confirmed else "not confirmed"
                    _progress("counterfactuals",
                              f"  -> {verdict} (delta L2={result.action_delta_l2:.4f})")
                    self.evidence_log.append(EvidenceEntry(
                        phase="counterfactual",
                        primitive_name=f"counterfactual.{hypothesis.test_type}",
                        data={"hypothesis_id": hypothesis.id,
                              "confirmed": result.confirmed,
                              "action_delta_l2": result.action_delta_l2},
                    ))
                except Exception as e:
                    _progress("counterfactuals", f"  -> FAILED: {e}")
        else:
            _progress("counterfactuals", "Skipping (no model loaded)")
            cf_skip_reason = "no_model"
            for h in hypotheses:
                if h.test_type != "none":
                    h.confidence *= 0.6

        _progress("counterfactuals", f"Completed {len(cf_results)} counterfactual tests")

        # ── Phase 6b: Iterative Follow-up ───────────────────────
        max_iterations = self.config.get("max_hypothesis_iterations", 1)
        if (max_iterations > 0 and self.policy is not None
                and cf_results and len(cf_results) < max_cf):
            # Check for surprising results: any hypothesis that was NOT confirmed
            # but had high confidence, or confirmed with unexpected delta
            surprising = [
                (h, r) for h, r in zip(hypotheses, cf_results)
                if h.id == r.hypothesis_id and (
                    (h.confidence > 0.7 and not r.confirmed)
                    or (r.confirmed and r.action_delta_l2 > 0.1)
                )
            ]
            if surprising:
                _progress("iteration", f"Found {len(surprising)} surprising results, forming follow-up hypotheses...")
                followup_hypotheses = await self._form_hypotheses(
                    task_string, scene, diversity, matrix, anomalies,
                    pre_qk_spatial_object if 'pre_qk_spatial_object' in locals() else preliminary_spatial_object,
                    semantic_probe,
                    qk_probe,
                )
                # Filter to only genuinely new hypotheses
                existing_ids = {h.id for h in hypotheses}
                existing_tests = {(h.test_type, str(h.test_params)) for h in hypotheses}
                for fh in followup_hypotheses:
                    if fh.id in existing_ids:
                        fh.id = f"f{fh.id}"
                    if (fh.test_type, str(fh.test_params)) in existing_tests:
                        continue
                    if fh.test_type == "none":
                        continue
                    if len(cf_results) >= max_cf:
                        break
                    _progress("iteration", f"Follow-up: {fh.test_type} for {fh.id}...")
                    try:
                        result = self._run_counterfactual(fh, sample, scene)
                        cf_results.append(result)
                        hypotheses.append(fh)
                        self.evidence_log.append(EvidenceEntry(
                            phase="iteration",
                            primitive_name=f"counterfactual.{fh.test_type}",
                            data={"hypothesis_id": fh.id,
                                  "confirmed": result.confirmed,
                                  "action_delta_l2": result.action_delta_l2},
                        ))
                    except Exception as e:
                        _progress("iteration", f"  Failed: {e}")

        # ── Phase 6c: Mandatory counterfactuals (always run for comparability) ──
        _mandatory_cf_tests = [
            "background_substitution",
            "object_relocation",
            "task_string_swap",
            "occlusion_targeted",
        ]
        if self.policy is not None and max_cf > 0:
            already_run = {r.test_type for r in cf_results}
            missing = [t for t in _mandatory_cf_tests if t not in already_run]
            if missing:
                _progress("counterfactuals",
                          f"Running {len(missing)} mandatory tests for comparability: {missing}")
                for test_type in missing:
                    # Build test_params with target_object when the test requires it
                    synth_params: dict = {}
                    if test_type in ("object_relocation", "occlusion_targeted"):
                        if target_object is None:
                            _progress("counterfactuals",
                                      f"  Skipping {test_type}: no detected objects for target_object")
                            continue
                        synth_params["target_object"] = target_object
                    # Create a synthetic hypothesis for the mandatory test
                    synth_h = Hypothesis(
                        id=f"mandatory_{test_type}",
                        description=f"Mandatory baseline test: {test_type}",
                        confidence=0.5,
                        supporting_anomalies=[],
                        test_type=test_type,
                        test_params=synth_params,
                        expected_if_true="Model output changes significantly",
                        expected_if_false="Model output remains stable",
                    )
                    _progress("counterfactuals",
                              f"  Running {test_type} (mandatory)...")
                    try:
                        result = self._run_counterfactual(synth_h, sample, scene)
                        cf_results.append(result)
                        hypotheses.append(synth_h)
                        verdict = "CONFIRMED" if result.confirmed else "not confirmed"
                        _progress("counterfactuals",
                              f"  -> {verdict} (delta L2={result.action_delta_l2:.4f})")
                        self.evidence_log.append(EvidenceEntry(
                            phase="counterfactual",
                            primitive_name=f"counterfactual.{test_type}",
                            data={"hypothesis_id": synth_h.id,
                                  "confirmed": result.confirmed,
                                  "action_delta_l2": result.action_delta_l2},
                        ))
                    except Exception as e:
                        _progress("counterfactuals", f"  -> FAILED: {e}")

        semantic_probe = build_semantic_probe_report(target_object, primary_semantic_frame, cf_results)

        # ── Phase 8: Spatial vs Object Diagnosis ─────────────────
        _progress("disambiguation", "Summarizing spatial priors vs object grounding...")
        pre_qk_spatial_object = build_spatial_object_diagnosis(
            matrix,
            target_object=target_object,
            cf_results=cf_results,
            dataset_diversity=diversity,
            semantic_probe=semantic_probe,
        )
        semantic_ambiguous = (
            primary_semantic_frame is not None
            and (
                primary_semantic_frame.target_margin_over_best_non_target is None
                or abs(primary_semantic_frame.target_margin_over_best_non_target) < 0.08
            )
        )
        if (
            self.policy is not None
            and target_object is not None
            and semantic_internal is not None
            and (pre_qk_spatial_object.verdict in {"mixed", "inconclusive"} or semantic_ambiguous)
        ):
            _progress("disambiguation", "Running last-layer SigLIP QK decomposition...")
            try:
                relocation_result = next(
                    (
                        result for result in cf_results
                        if result.test_type == "object_relocation"
                        and (result.metrics or {}).get("target_object") == target_object
                    ),
                    None,
                )
                qk_probe = build_qk_probe_report(
                    self.policy,
                    sample,
                    self.dataset,
                    self.image_key,
                    self.device,
                    target_object=target_object,
                    scene=scene,
                    target_semantic_map=semantic_internal["target_map"],
                    positional_baseline=signals.get("positional_baseline"),
                    relocation_result=relocation_result,
                    image_map=self.image_map,
                )
                if qk_probe is not None:
                    _progress("disambiguation", summarize_qk_probe(qk_probe))
            except Exception as e:
                _progress("disambiguation", f"QK probe skipped: {e}")

        spatial_object_diagnosis = build_spatial_object_diagnosis(
            matrix,
            target_object=target_object,
            cf_results=cf_results,
            dataset_diversity=diversity,
            semantic_probe=semantic_probe,
            qk_probe=qk_probe,
        )
        _progress(
            "disambiguation",
            summarize_spatial_object_diagnosis(spatial_object_diagnosis),
        )

        # ── Phase 9: LLM Synthesis ─────────────────────────────
        _progress("synthesis", f"Synthesizing report via {llm_label}...")
        findings, narrative = await self._synthesize_report(
            task_string, scene, diversity, matrix, anomalies,
            hypotheses, cf_results, spatial_object_diagnosis,
            semantic_probe,
            qk_probe,
            cf_skip_reason=cf_skip_reason)
        _progress("synthesis", f"Generated {len(findings)} findings:")
        for f in findings:
            sev_icon = {"critical": "!!", "warning": "! ", "info": "  "}.get(f.severity, "  ")
            _progress("synthesis", f"  {sev_icon} [{f.severity.upper()}] {f.title}")

        # ── Build Final Report ──────────────────────────────────
        metadata = {
            "task_string": task_string,
            "episode_idx": self.episode_idx,
            "image_key": self.image_key,
            "device": self.device,
            "post_hoc": self.post_hoc,
            "target_object": target_object or "unknown",
        }
        try:
            if self.policy is not None:
                cfg = self.policy.config
                metadata["model_id"] = (
                    getattr(cfg, "repo_id", None)
                    or getattr(cfg, "pretrained_model_name_or_path", None)
                    or getattr(cfg, "_name_or_path", None)
                    or "unknown"
                )
            else:
                metadata["model_id"] = "none"
        except Exception:
            metadata["model_id"] = "unknown"
        try:
            metadata["dataset_id"] = getattr(self.dataset, "repo_id", "unknown")
        except Exception:
            metadata["dataset_id"] = "unknown"

        report = DiagnosticReport(
            metadata=metadata,
            scene=scene,
            dataset_diversity=diversity,
            matrix=matrix,
            semantic_probe=semantic_probe,
            qk_probe=qk_probe,
            spatial_object_diagnosis=spatial_object_diagnosis,
            anomalies=anomalies,
            hypotheses=hypotheses,
            counterfactual_results=cf_results,
            findings=findings,
            llm_synthesis=narrative,
            cf_skip_reason=cf_skip_reason,
        )

        total_time = time.time() - _run_start
        _progress("complete", f"Diagnostic report ready  (total: {total_time:.1f}s)")
        n_crit = sum(1 for f in findings if f.severity == "critical")
        n_warn = sum(1 for f in findings if f.severity == "warning")
        n_info = sum(1 for f in findings if f.severity == "info")
        print(f"  {'':>5}  {n_crit} critical, {n_warn} warnings, {n_info} info findings")
        print(f"  {'':>5}  {len(hypotheses)} hypotheses, {len(cf_results)} counterfactuals")
        return report

    # ─── Internal helpers ───────────────────────────────────────

    def _get_first_frame_idx(self) -> int:
        """Get the dataset index of the first frame in the episode."""
        try:
            return int(self.dataset.meta.episodes["dataset_from_index"][self.episode_idx])
        except (AttributeError, KeyError):
            try:
                return int(self.dataset.episode_data_index["from"][self.episode_idx].item())
            except (AttributeError, KeyError):
                return self.episode_idx * 200

    def _get_first_frame_image(self, sample) -> np.ndarray:
        """Extract the first frame as (H, W, 3) uint8 numpy array."""
        img = sample[self.image_key]
        if hasattr(img, "numpy"):
            arr = img.numpy()
        else:
            arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return arr

    def _load_existing_signals(self) -> dict:
        """Load heatmaps from a completed run's NPZ files (post-hoc mode)."""
        signals = {}
        run_dir = self.existing_run_dir

        def _load_heatmaps(npz_path, prefix):
            if not os.path.exists(npz_path):
                return None
            data = np.load(npz_path)
            keys = sorted([k for k in data.files if k.startswith(prefix)])
            if not keys:
                # Try loading all keys as heatmaps
                keys = sorted(data.files)
            return [data[k] for k in keys] if keys else None

        # Self-attention
        sa_path = os.path.join(run_dir, "data", "self_attention", "heatmaps.npz")
        signals["attention"] = _load_heatmaps(sa_path, "heatmap_")

        # Positional baseline
        bl_path = os.path.join(run_dir, "data", "self_attention", "positional_baseline.npz")
        if os.path.exists(bl_path):
            data = np.load(bl_path)
            signals["positional_baseline"] = data.get("baseline", None)

        # Cross-attention
        ca_path = os.path.join(run_dir, "data", "cross_attention", "heatmaps.npz")
        signals["cross_attention"] = _load_heatmaps(ca_path, "heatmap_")

        # Gradient data
        grad_dir = os.path.join(run_dir, "data", "gradient")

        sal_path = os.path.join(grad_dir, "saliency.npz")
        signals["saliency"] = _load_heatmaps(sal_path, "saliency_")

        gc_path = os.path.join(grad_dir, "gradcam_siglip.npz")
        signals["gradcam_siglip"] = _load_heatmaps(gc_path, "gradcam_")

        conn_path = os.path.join(grad_dir, "gradcam_connector.npz")
        signals["gradcam_connector"] = _load_heatmaps(conn_path, "connector_")

        # Vision vs state
        vs_path = os.path.join(grad_dir, "vision_vs_state.json")
        if os.path.exists(vs_path):
            with open(vs_path) as f:
                signals["vision_vs_state"] = json.load(f)

        # Language diff
        ld_path = os.path.join(grad_dir, "language_diff.npz")
        if os.path.exists(ld_path):
            data = np.load(ld_path)
            # Build language_diff structure
            diff_keys = [k for k in data.files if "_diff" in k]
            if diff_keys:
                signals["language_diff"] = {"diff_cam": [data[k] for k in sorted(diff_keys)]}

        # Model internals
        internals_path = os.path.join(run_dir, "data", "model_internals")
        if os.path.isdir(internals_path):
            internals = {}
            for fname in os.listdir(internals_path):
                if fname.endswith(".json"):
                    with open(os.path.join(internals_path, fname)) as f:
                        internals[fname.replace(".json", "")] = json.load(f)
            if internals:
                signals["model_internals"] = internals

        # Filter out None values
        return {k: v for k, v in signals.items() if v is not None}

    def _run_cheap_triage(self, sample) -> dict:
        """Run cheap, always-on signals: self-attention + positional baseline."""
        import torch
        signals = {}

        try:
            from ..data import find_vision_encoder
            from ..capture import SigLIPAttentionCapture
            from ..heatmap import compute_patch_attention_scores, attention_to_heatmap
            from ..gradient import _patch_eager_attention_bool_mask
        except ImportError as e:
            print(f"  Warning: Could not import triage modules: {e}")
            return signals

        try:
            vision_encoder = find_vision_encoder(self.policy)
            if vision_encoder is None:
                return signals

            # Force eager attention so hooks receive actual weights
            # (SDPA returns None for weights by default)
            for mod in vision_encoder.modules():
                cfg = getattr(mod, "config", None)
                if cfg is not None and hasattr(cfg, "_attn_implementation"):
                    cfg._attn_implementation = "eager"

            attn_capture = SigLIPAttentionCapture()
            attn_capture.register_hooks(vision_encoder)

            frames_data = get_episode_frames(
                self.dataset, self.episode_idx,
                self.config.get("num_frames", 4), self.image_key)

            heatmaps = []
            for frame_idx, img_tensor in frames_data:
                frame_sample = self.dataset[frame_idx]
                batch, _ = build_policy_batch_from_sample(
                    frame_sample, self.policy, self.device,
                    dataset=self.dataset, image_map=self.image_map)
                self.policy.reset()
                with _patch_eager_attention_bool_mask(self.policy), torch.no_grad():
                    self.policy.select_action(batch)
                last_attn = attn_capture.get_last_layer_attention()
                if last_attn is not None:
                    # Squeeze batch dimension: (1, heads, seq, seq) → (heads, seq, seq)
                    if last_attn.dim() == 4:
                        last_attn = last_attn.squeeze(0)
                    scores = compute_patch_attention_scores(last_attn)
                    try:
                        vc = vision_encoder.config
                        gs = vc.image_size // vc.patch_size
                        target_hw = (gs * vc.patch_size, gs * vc.patch_size)
                    except Exception:
                        gs = 32
                        target_hw = (512, 512)
                    hm = attention_to_heatmap(scores, (gs, gs), target_hw)
                    heatmaps.append(hm)
                attn_capture.reset_maps()

            if heatmaps:
                signals["attention"] = heatmaps

            # Positional baseline (cheap)
            try:
                from ..heatmap import compute_positional_baseline
                baseline_scores, _ = compute_positional_baseline(
                    vision_encoder, attn_capture, self.device, "last-layer",
                    input_hw=(512, 512))
                if baseline_scores is not None:
                    try:
                        vc = vision_encoder.config
                        gs = vc.image_size // vc.patch_size
                        target_hw = (gs * vc.patch_size, gs * vc.patch_size)
                    except Exception:
                        gs = 32
                        target_hw = (512, 512)
                    signals["positional_baseline"] = attention_to_heatmap(
                        baseline_scores, (gs, gs), target_hw)
            except Exception:
                pass

            attn_capture.clear()
        except Exception as e:
            print(f"  Warning: Attention extraction failed: {e}")

        return signals

    async def _select_expensive_signals(self, task_string: str,
                                         scene, signals: dict) -> list[str]:
        """Use LLM to select which expensive signals to collect."""
        # Summarise cheap signals for the LLM
        summary_lines = []
        if "attention" in signals:
            n = len(signals["attention"])
            summary_lines.append(f"- Self-attention heatmaps: {n} frames collected")
            # Quick foreground ratio estimate
            if signals["attention"] and hasattr(scene, 'background_mask'):
                from .regions import foreground_ratio
                fg = foreground_ratio(signals["attention"][0], scene)
                summary_lines.append(f"  - Foreground ratio (first frame): {fg:.2%}")
        if "positional_baseline" in signals:
            from .regions import spatial_prior_ratio
            if "attention" in signals and signals["attention"]:
                pr = spatial_prior_ratio(signals["attention"][0], signals["positional_baseline"])
                summary_lines.append(f"- Positional baseline similarity: {pr:.2f}")
        if not summary_lines:
            summary_lines.append("- No cheap signals available (attention extraction failed)")

        max_signals = self.config.get("max_expensive_signals", 5)

        prompt = build_triage_selection_prompt(
            task_string=task_string,
            detected_objects=scene.region_names(),
            cheap_signals_summary="\n".join(summary_lines),
            max_signals=max_signals,
        )

        response = await self._call_llm(prompt)

        # Parse JSON array of signal names
        try:
            selected = _extract_json_array(response)
            if selected and isinstance(selected[0], str):
                return selected[:max_signals]
            # If it parsed as list of dicts, try extracting strings
            return [str(s) for s in json.loads(response.strip()) if isinstance(s, str)][:max_signals]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback: run a sensible default set
        return ["gradcam_siglip", "saliency", "vision_vs_state"]

    def _run_expensive_signal(self, signal_name: str, sample, signals: dict,
                              scene=None):
        """Run a single expensive signal and store in signals dict."""
        import torch
        frame_sample = self.dataset[self._get_first_frame_idx()]

        if signal_name == "gradcam_siglip":
            from ..gradient import compute_gradcam_map
            cam = compute_gradcam_map(
                self.policy, frame_sample, self.dataset, self.image_key,
                self.device, image_map=self.image_map)
            if cam is not None:
                signals["gradcam_siglip"] = [cam]

        elif signal_name == "saliency":
            from ..gradient import compute_saliency_map
            sal = compute_saliency_map(
                self.policy, frame_sample, self.dataset, self.image_key,
                self.device, image_map=self.image_map)
            if sal is not None:
                signals["saliency"] = [sal]

        elif signal_name == "vision_vs_state":
            from ..gradient import compute_vision_vs_state_ratio
            vs = compute_vision_vs_state_ratio(
                self.policy, frame_sample, self.dataset, self.image_key,
                self.device, image_map=self.image_map)
            if vs is not None:
                signals["vision_vs_state"] = [vs]

        elif signal_name == "per_action_dim_gradcam":
            from ..gradient import compute_per_action_dim_gradcam
            result = compute_per_action_dim_gradcam(
                self.policy, frame_sample, self.dataset, self.image_key,
                self.device, image_map=self.image_map)
            if result is not None:
                # Convert to the dict format expected by build_diagnostic_matrix
                dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
                per_dim = {}
                for i, cam in enumerate(result["maps"]):
                    name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
                    per_dim[name] = [cam]
                signals["per_action_dim"] = per_dim

        elif signal_name == "connector_analysis":
            self._run_connector_analysis(frame_sample, signals, scene=scene)

        elif signal_name == "occlusion_sensitivity":
            try:
                from .occlusion import compute_occlusion_sensitivity
                occ = compute_occlusion_sensitivity(
                    self.policy, frame_sample, self.dataset,
                    self.image_key, self.device,
                    patch_size=self.config.get("occlusion_patch_size", 64),
                    stride=self.config.get("occlusion_stride", 32),
                    image_map=self.image_map)
                if occ is not None:
                    signals["occlusion_map"] = occ
            except Exception as e:
                print(f"  Warning: Occlusion sensitivity failed: {e}")

        elif signal_name == "temporal_trajectory":
            try:
                from .temporal import compute_temporal_trajectory
                traj = compute_temporal_trajectory(
                    self.policy, self.dataset, self.episode_idx,
                    self.image_key, self.device,
                    num_frames=self.config.get("temporal_frames", 20),
                    image_map=self.image_map)
                if traj is not None:
                    signals["temporal_trajectories"] = [traj]
            except Exception as e:
                print(f"  Warning: Temporal trajectory failed: {e}")

        elif signal_name == "language_diff":
            from ..gradient import compute_language_conditional_diff
            diff = compute_language_conditional_diff(
                self.policy, frame_sample, self.dataset, self.image_key,
                self.device, image_map=self.image_map)
            if diff is not None:
                signals["language_diff"] = diff

    def _run_connector_analysis(self, sample, signals: dict, scene=None):
        """Compare SigLIP (pre-connector) and connector (post-connector) attribution."""
        from ..gradient import compute_gradcam_map, compute_gradcam_connector
        from .models import ConnectorAnalysis

        pre_cam = compute_gradcam_map(
            self.policy, sample, self.dataset, self.image_key,
            self.device, image_map=self.image_map)
        post_cam = compute_gradcam_connector(
            self.policy, sample, self.dataset, self.image_key,
            self.device, image_map=self.image_map)

        if pre_cam is None or post_cam is None:
            return

        # Store raw heatmaps for the matrix builder
        signals["gradcam_siglip"] = signals.get("gradcam_siglip", []) or []
        if not signals["gradcam_siglip"]:
            signals["gradcam_siglip"] = [pre_cam]
        signals["gradcam_connector"] = [post_cam]

        # Build ConnectorAnalysis with per-region information loss
        if scene is not None:
            pre_shares = attribute_to_regions(pre_cam, scene, signal_name="gradcam_siglip")
            post_shares = attribute_to_regions(post_cam, scene, signal_name="gradcam_connector")
            info_loss = {}
            for region in pre_shares:
                pre_val = pre_shares.get(region, 0.0)
                post_val = post_shares.get(region, 0.0)
                info_loss[region] = max(0.0, pre_val - post_val)
            total_loss = sum(info_loss.values())
            signals["connector_analysis"] = ConnectorAnalysis(
                pre_connector_shares=pre_shares,
                post_connector_shares=post_shares,
                information_loss_per_region=info_loss,
                total_information_loss=total_loss,
            )

    def _run_counterfactual(self, hypothesis: Hypothesis, sample,
                            scene: SceneSegmentation) -> CounterfactualResult:
        """Run a single counterfactual test."""
        primitive_name = f"counterfactual.{hypothesis.test_type}"
        if primitive_name not in REGISTRY:
            raise ValueError(f"Unknown counterfactual: {primitive_name}")

        primitive = REGISTRY[primitive_name]
        params = dict(hypothesis.test_params)
        params.update({
            "policy": self.policy,
            "sample": sample,
            "dataset": self.dataset,
            "image_key": self.image_key,
            "device": self.device,
            "image_map": self.image_map,
        })

        # Add segmentation if needed
        if "segmentation" in primitive.fn.__code__.co_varnames:
            params["segmentation"] = scene

        # Add episode_idx for temporal tests
        if "episode_idx" in primitive.fn.__code__.co_varnames:
            params["episode_idx"] = self.episode_idx

        result = primitive.fn(**params)
        result.hypothesis_id = hypothesis.id
        result.confirmed = self._evaluate_result(hypothesis, result)
        return result

    def _evaluate_result(self, hypothesis: Hypothesis, result: CounterfactualResult) -> bool:
        """Evaluate whether a counterfactual result confirms the hypothesis.

        Uses ``hypothesis.confirms_on_change``:
        - True (default): confirmed when action_delta_l2 > threshold (model
          was sensitive to the perturbation).
        - False: confirmed when action_delta_l2 <= threshold (model was
          *insensitive*, e.g. weak grounding or language blindness).
        """
        significant_change = result.action_delta_l2 > 0.02
        if hypothesis.confirms_on_change:
            return significant_change
        return not significant_change

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM via the configured provider."""
        provider = self.llm_settings.get("provider", "anthropic")
        model = self.llm_settings.get("model", "claude-sonnet-4-20250514")
        api_key = self.llm_settings.get("api_key", "")
        base_url = self.llm_settings.get("base_url", "")

        if not api_key:
            # Try env vars
            if provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            else:
                api_key = os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            print("  Warning: No LLM API key configured. Using fallback reasoning.")
            return self._fallback_llm_response(prompt)

        if provider == "anthropic":
            return await self._call_anthropic(prompt, model, api_key)
        else:
            return await self._call_openai(prompt, model, api_key, base_url)

    async def _call_anthropic(self, prompt: str, model: str, api_key: str) -> str:
        """Call Anthropic API."""
        try:
            import anthropic
        except ImportError:
            return self._fallback_llm_response(prompt)

        client = anthropic.AsyncAnthropic(api_key=api_key)
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            if not content or content.strip() in ("", "[]"):
                print(f"  Warning: Anthropic LLM returned empty response (model={model})")
                return self._fallback_llm_response(prompt)
            return content
        except Exception as e:
            print(f"  Warning: Anthropic API call failed: {e}")
            return self._fallback_llm_response(prompt)

    async def _call_openai(self, prompt: str, model: str, api_key: str, base_url: str) -> str:
        """Call OpenAI or compatible API."""
        try:
            import openai
        except ImportError:
            return self._fallback_llm_response(prompt)

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.AsyncOpenAI(**kwargs)

        is_json_prompt = "Output ONLY a JSON" in prompt or "Output valid JSON" in prompt

        create_kwargs: dict = {
            "model": model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Add system prompt for JSON output requests
        if is_json_prompt:
            create_kwargs["messages"].insert(0, {
                "role": "system",
                "content": (
                    "You are an expert robotics ML researcher. "
                    "Output valid JSON arrays when asked. "
                    "Do not wrap JSON in markdown code fences. "
                    "Start your response immediately with the [ character."
                ),
            })
            # Note: we intentionally do NOT use response_format: json_object
            # because it forces a single JSON object, but our prompts request
            # JSON arrays. The system message is sufficient.

        try:
            response = await client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content
            if not content or content.strip() in ("", "null"):
                print(f"  Warning: LLM returned empty/null response (model={model})")
                return self._fallback_llm_response(prompt)
            return content
        except Exception as e:
            print(f"  Warning: LLM API call failed: {e}")
            return self._fallback_llm_response(prompt)

    def _fallback_llm_response(self, prompt: str) -> str:
        """Generate a rule-based response when no LLM is available."""
        if "hypothesis" in prompt.lower() or "hypothesize" in prompt.lower():
            return '[]'
        return '[]---NARRATIVE---\nDiagnostic analysis completed. LLM synthesis unavailable — review the diagnostic matrix and anomalies above for details.'

    async def _form_hypotheses(self, task_string: str, scene: SceneSegmentation,
                                diversity, matrix: DiagnosticMatrix,
                                anomalies: list[Anomaly],
                                spatial_object_diagnosis=None,
                                semantic_probe=None,
                                qk_probe=None) -> list[Hypothesis]:
        """Use LLM to form hypotheses from the diagnostic data."""
        prompt = build_hypothesis_prompt(
            task_string=task_string,
            detected_objects=scene.region_names(),
            diversity_summary=format_diversity_summary(diversity),
            matrix_markdown=matrix.to_markdown(),
            anomalies_json=format_anomalies_json(anomalies),
            spatial_object_summary=summarize_spatial_object_diagnosis(spatial_object_diagnosis),
            semantic_probe_summary=summarize_semantic_probe(semantic_probe),
            qk_probe_summary=summarize_qk_probe(qk_probe),
        )

        response = await self._call_llm(prompt)

        # Parse JSON response (with repair for common LLM quirks)
        hypotheses = []
        try:
            items = _extract_json_array(response)
            # Filter to only dict items (LLM may return string arrays)
            dict_items = [item for item in items if isinstance(item, dict)]
            for item in dict_items:
                hypotheses.append(Hypothesis(
                    id=item.get("id", f"h{len(hypotheses)+1}"),
                    description=item.get("description", ""),
                    confidence=float(item.get("confidence", 0.5)),
                    supporting_anomalies=item.get("supporting_anomalies", []),
                    test_type=item.get("test_type", "none"),
                    test_params=item.get("test_params", {}),
                    expected_if_true=item.get("expected_if_true", ""),
                    expected_if_false=item.get("expected_if_false", ""),
                    confirms_on_change=item.get("confirms_on_change", True),
                ))
            if not dict_items:
                preview = response[:500] if len(response) > 500 else response
                print(f"  Warning: LLM returned no parseable hypotheses.")
                print(f"  Response preview: {preview!r}")
        except (KeyError, TypeError, ValueError) as e:
            print(f"  Warning: Failed to parse LLM hypotheses: {e}")

        # Merge rule-based hypotheses for any anomaly-test mappings the LLM missed
        rule_based = self._rule_based_hypotheses(
            anomalies,
            scene,
            target_object=getattr(spatial_object_diagnosis, "target_object", None),
        )
        llm_test_types = {h.test_type for h in hypotheses}
        for rh in rule_based:
            if rh.test_type not in llm_test_types:
                hypotheses.append(rh)
                llm_test_types.add(rh.test_type)

        return hypotheses[:self.config.get("max_hypotheses", 7)]

    def _rule_based_hypotheses(self, anomalies: list[Anomaly],
                                scene: SceneSegmentation,
                                target_object: str | None = None) -> list[Hypothesis]:
        """Generate hypotheses from anomalies without LLM."""
        hypotheses = []
        target_objects = [r for r in scene.region_names() if r != "background" and r != "robot gripper"]
        target = target_object or (target_objects[0] if target_objects else "object")

        for anomaly in anomalies:
            if anomaly.type == "high_background_attribution":
                hypotheses.append(Hypothesis(
                    id=f"h{len(hypotheses)+1}",
                    description="Model relies on background features rather than task-relevant objects",
                    confidence=0.7,
                    supporting_anomalies=[anomaly.type],
                    test_type="background_substitution",
                    test_params={"replacement": "gray"},
                    expected_if_true="Action prediction changes significantly when background is replaced",
                    expected_if_false="Action prediction remains stable",
                ))
            elif anomaly.type == "spatial_shortcut":
                hypotheses.append(Hypothesis(
                    id=f"h{len(hypotheses)+1}",
                    description=f"Model uses spatial shortcuts — memorized {target} position rather than recognizing it",
                    confidence=0.8,
                    supporting_anomalies=[anomaly.type],
                    test_type="object_relocation",
                    test_params={"target_object": target, "shift_pixels": [100, -80]},
                    expected_if_true="Action prediction MSE increases >2x when object is relocated",
                    expected_if_false="Model correctly adjusts actions to new object position",
                ))
            elif anomaly.type == "low_object_attribution":
                hypotheses.append(Hypothesis(
                    id=f"h{len(hypotheses)+1}",
                    description=f"Model has weak visual grounding for {target}",
                    confidence=0.6,
                    supporting_anomalies=[anomaly.type],
                    test_type="object_recolor",
                    test_params={"target_object": target, "hue_shift": 0.5},
                    expected_if_true="Model is insensitive to object appearance changes",
                    expected_if_false="Action changes, showing some object-appearance sensitivity",
                    confirms_on_change=False,
                ))
            elif anomaly.type == "dead_state_pathway":
                hypotheses.append(Hypothesis(
                    id=f"h{len(hypotheses)+1}",
                    description="Proprioceptive state input is not contributing to action decisions",
                    confidence=0.6,
                    supporting_anomalies=[anomaly.type],
                    test_type="none",
                    test_params={},
                    expected_if_true="N/A — confirmed by vision vs state analysis",
                    expected_if_false="N/A",
                ))
            elif anomaly.type == "low_dataset_diversity":
                hypotheses.append(Hypothesis(
                    id=f"h{len(hypotheses)+1}",
                    description=f"Model memorized fixed {target} positions due to low dataset diversity",
                    confidence=0.7,
                    supporting_anomalies=[anomaly.type],
                    test_type="object_relocation",
                    test_params={"target_object": target, "shift_pixels": [80, -60]},
                    expected_if_true="Action breaks when object is moved from memorized position",
                    expected_if_false="Model generalizes to new positions despite low training diversity",
                ))
            elif anomaly.type == "gripper_fixation":
                hypotheses.append(Hypothesis(
                    id=f"h{len(hypotheses)+1}",
                    description="Model fixates on robot gripper instead of manipulation target",
                    confidence=0.7,
                    supporting_anomalies=[anomaly.type],
                    test_type="occlusion_targeted",
                    test_params={"target_object": target, "fill": "gray"},
                    expected_if_true="Occluding target object has minimal effect (model relies on gripper)",
                    expected_if_false="Occluding target changes actions significantly",
                    confirms_on_change=False,
                ))
            elif anomaly.type == "cross_attention_diffuse":
                hypotheses.append(Hypothesis(
                    id=f"h{len(hypotheses)+1}",
                    description="Action expert cross-attention is near-uniform — not selectively querying",
                    confidence=0.5,
                    supporting_anomalies=[anomaly.type],
                    test_type="distractor_insertion",
                    test_params={"position": [100, 100], "distractor_size": 80},
                    expected_if_true="Model is equally distracted by novel objects",
                    expected_if_false="Model ignores distractors despite diffuse attention",
                ))
            elif anomaly.type == "language_insensitivity":
                hypotheses.append(Hypothesis(
                    id=f"h{len(hypotheses)+1}",
                    description="Model ignores language conditioning — acts the same regardless of instruction",
                    confidence=0.7,
                    supporting_anomalies=[anomaly.type],
                    test_type="task_string_swap",
                    test_params={"replacement_task": "do nothing"},
                    expected_if_true="Changing instruction to 'do nothing' has no effect on actions",
                    expected_if_false="Actions change, suggesting some language sensitivity",
                    confirms_on_change=False,
                ))
            elif anomaly.type == "temporal_attention_instability":
                hypotheses.append(Hypothesis(
                    id=f"h{len(hypotheses)+1}",
                    description="Attention is temporally unstable — jumps between frames",
                    confidence=0.6,
                    supporting_anomalies=[anomaly.type],
                    test_type="temporal_consistency",
                    test_params={"perturbation_type": "background_substitution", "num_frames": 5},
                    expected_if_true="Model responds inconsistently to same perturbation across frames",
                    expected_if_false="Model responds consistently despite attention instability",
                ))

        return hypotheses

    async def _synthesize_report(self, task_string: str, scene: SceneSegmentation,
                                  diversity, matrix: DiagnosticMatrix,
                                  anomalies: list[Anomaly],
                                  hypotheses: list[Hypothesis],
                                  cf_results: list[CounterfactualResult],
                                  spatial_object_diagnosis=None,
                                  semantic_probe=None,
                                  qk_probe=None,
                                  cf_skip_reason: str = "",
                                  ) -> tuple[list[Finding], str]:
        """Use LLM to synthesize findings and narrative from all evidence."""
        prompt = build_synthesis_prompt(
            task_string=task_string,
            detected_objects=scene.region_names(),
            matrix_markdown=matrix.to_markdown(),
            spatial_object_summary=summarize_spatial_object_diagnosis(spatial_object_diagnosis),
            semantic_probe_summary=summarize_semantic_probe(semantic_probe),
            qk_probe_summary=summarize_qk_probe(qk_probe),
            anomalies_summary=format_anomalies_json(anomalies),
            diversity_summary=format_diversity_summary(diversity),
            hypotheses_with_results=format_hypotheses_with_results(
                hypotheses, cf_results, skip_reason=cf_skip_reason),
        )

        response = await self._call_llm(prompt)

        # Save raw response for debugging
        try:
            with open("/tmp/smolvla_llm_synthesis_raw.txt", "w") as f:
                f.write(response)
        except Exception:
            pass

        findings = []
        narrative = ""

        try:
            # Split on separator — try multiple variations LLMs might produce
            json_part = response
            for sep in ("---NARRATIVE---", "--- NARRATIVE ---", "---narrative---", "---Narrative---"):
                if sep in response:
                    json_part, narrative = response.split(sep, 1)
                    narrative = narrative.strip()
                    break

            # Parse findings JSON (with repair for common LLM quirks)
            items = _extract_json_array(json_part)
            dict_items = [item for item in items if isinstance(item, dict)]
            for item in dict_items:
                findings.append(Finding(
                    id=item.get("id", f"f{len(findings)+1}"),
                    severity=item.get("severity", "info"),
                    title=item.get("title", ""),
                    observation=item.get("observation", ""),
                    test_description=item.get("test_description", ""),
                    test_result=item.get("test_result", ""),
                    interpretation=item.get("interpretation", ""),
                    fix=item.get("fix", ""),
                    expected_impact=item.get("expected_impact", ""),
                    evidence_refs=item.get("evidence_refs", []),
                ))

            if not dict_items:
                # Log what the LLM actually returned so we can debug format issues
                preview = response[:500] if len(response) > 500 else response
                print(f"  Warning: LLM synthesis returned no parseable JSON findings.")
                print(f"  Response preview: {preview!r}")
        except (KeyError, TypeError, ValueError) as e:
            print(f"  Warning: Failed to parse LLM synthesis: {e}")
            preview = response[:300] if len(response) > 300 else response
            print(f"  Response preview: {preview!r}")

        # If LLM failed, generate rule-based findings
        if not findings:
            print("  Falling back to rule-based findings.")
            findings = self._rule_based_findings(
                anomalies,
                hypotheses,
                cf_results,
                spatial_object_diagnosis=spatial_object_diagnosis,
            )

        if not narrative:
            narrative = self._rule_based_narrative(
                anomalies,
                findings,
                spatial_object_diagnosis=spatial_object_diagnosis,
            )

        return findings, narrative

    def _rule_based_findings(self, anomalies: list[Anomaly],
                              hypotheses: list[Hypothesis],
                              cf_results: list[CounterfactualResult],
                              spatial_object_diagnosis=None) -> list[Finding]:
        """Generate expert-quality findings without LLM, using anomaly-specific knowledge."""
        findings = []
        result_map = {r.hypothesis_id: r for r in cf_results}

        if spatial_object_diagnosis is not None:
            diag = spatial_object_diagnosis
            verdict_titles = {
                "spatial_prior": "Policy appears anchored to memorized spatial positions",
                "object_grounded": "Policy appears grounded on object features",
                "mixed": "Policy uses a mixed object-and-location strategy",
                "inconclusive": "Spatial-vs-object verdict is inconclusive",
            }
            verdict_interpretations = {
                "spatial_prior": (
                    "The model is using location as a shortcut. It may succeed when objects remain in "
                    "their usual positions, but it is likely to fail when object layout changes."
                ),
                "object_grounded": (
                    "The model is using visual evidence from the target object itself. This is the "
                    "desired behavior for robust generalization across positions."
                ),
                "mixed": (
                    "The model has learned some object grounding, but it still retains a meaningful "
                    "location prior. This will produce partial generalization with brittle failures at "
                    "larger spatial shifts."
                ),
                "inconclusive": (
                    "The current signals are not strong enough to cleanly determine whether the model "
                    "is reading the object, the location, or both."
                ),
            }
            verdict_fix = {
                "spatial_prior": (
                    "Prioritize breaking the spatial shortcut: randomize object positions, widen camera "
                    "pose diversity, and use relocation-focused evaluations during training."
                ),
                "object_grounded": (
                    "Preserve this behavior while stress-testing for edge cases: evaluate on larger "
                    "position shifts, new object instances, and new backgrounds."
                ),
                "mixed": (
                    "Reduce the remaining spatial prior without losing object grounding: train with "
                    "larger object position variance and relocation-heavy augmentation, then re-run the "
                    "relocation follow probe."
                ),
                "inconclusive": (
                    "Collect stronger disambiguation evidence: ensure the target object is segmented "
                    "cleanly and inspect relocation/occlusion probes on a representative episode."
                ),
            }
            diag_severity = {
                "spatial_prior": "critical",
                "mixed": "warning",
                "object_grounded": "info",
                "inconclusive": "info",
            }.get(diag.verdict, "info")
            findings.append(Finding(
                id=f"f{len(findings)+1}",
                severity=diag_severity,
                title=verdict_titles.get(diag.verdict, "Spatial-vs-object diagnosis"),
                observation=diag.summary,
                test_description="Combined positional-baseline, attribution, dataset-diversity, and counterfactual evidence.",
                test_result=(
                    f"Spatial score={diag.spatial_score:.4f}, object score={diag.object_score:.4f}, "
                    f"confidence={diag.confidence:.0%}."
                ),
                interpretation=verdict_interpretations.get(diag.verdict, diag.summary),
                fix=verdict_fix.get(diag.verdict, "Review the structured evidence and rerun the spatial-vs-object probes."),
                expected_impact=(
                    "Clarifies whether failures come from memorized coordinates, object recognition, or a mixed strategy."
                ),
                evidence_refs=[f"spatial_object:{diag.verdict}"],
            ))

        # Anomaly-type → expert knowledge database
        _EXPERT_DB = {
            "high_background_attribution": {
                "title": "Background texture dependence",
                "interpretation": (
                    "The model uses background texture/color as a spatial reference frame rather than "
                    "attending to task-relevant objects. This means the policy will fail when deployed "
                    "in environments with different backgrounds, tables, or lighting conditions."
                ),
                "fix": (
                    "Add aggressive background augmentation during training:\n"
                    "1. Random background substitution: swap backgrounds from a diverse image bank (20+ environments)\n"
                    "2. Color jitter: `brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1`\n"
                    "3. Random crops with resize to break spatial priors\n"
                    "4. Gaussian blur augmentation (kernel 5-15, sigma 0.5-2.0)\n"
                    "5. Consider training with green-screen backgrounds and compositing varied scenes\n\n"
                    "In the training config, add:\n"
                    "```yaml\n"
                    "augmentation:\n"
                    "  background_randomization: true\n"
                    "  color_jitter: {brightness: 0.4, contrast: 0.4, saturation: 0.3, hue: 0.1}\n"
                    "  random_crop_scale: [0.8, 1.0]\n"
                    "```"
                ),
                "expected_impact": (
                    "Should reduce background attribution from >70% to <30% and improve "
                    "cross-environment transfer. Expect 2-3x improvement in novel environment success rate."
                ),
            },
            "low_object_attribution": {
                "title": "Weak visual grounding on manipulation targets",
                "interpretation": (
                    "The model's vision encoder assigns negligible attention to the objects it needs to "
                    "manipulate. It is likely relying on spatial priors (memorized positions) or background "
                    "cues rather than actually recognizing and tracking the objects. This will cause "
                    "failures when objects appear in novel positions, orientations, or appearances."
                ),
                "fix": (
                    "Improve object-level grounding through:\n"
                    "1. **Object-centric augmentation**: Randomly vary object color, size, and texture during training\n"
                    "2. **Cutout/CutMix augmentation**: Force the model to use multiple visual cues\n"
                    "3. **Auxiliary object detection loss**: Add a lightweight detection head that predicts object bounding boxes\n"
                    "4. **Increase object diversity in training data**: Collect episodes with 3-5 different instances of each object class\n"
                    "5. **Attention supervision**: If available, add soft attention guidance toward task-relevant regions\n\n"
                    "Quick win: add `RandomErasing(p=0.3, scale=(0.02, 0.15))` to the training augmentation pipeline."
                ),
                "expected_impact": (
                    "Should increase object attribution from <5% to >20% per task-relevant object. "
                    "Expect improved generalization to novel object instances and positions."
                ),
            },
            "spatial_shortcut": {
                "title": "Spatial shortcut learning from fixed object positions",
                "interpretation": (
                    "The attention pattern closely matches a positional baseline (what the model would "
                    "attend to with a blank image), indicating the model has memorized fixed spatial "
                    "positions rather than learning true visual recognition. This is a memorization failure "
                    "that will catastrophically break when objects are in different positions."
                ),
                "fix": (
                    "Break spatial memorization through data diversity:\n"
                    "1. **Randomize object positions**: Collect new episodes with objects placed at 10+ different positions\n"
                    "2. **Spatial jitter augmentation**: Apply random affine transforms (translate ±20%, rotate ±15°)\n"
                    "3. **Multi-camera training**: If possible, add side/wrist camera views to triangulate\n"
                    "4. **Randomize camera position**: Even small ±5cm shifts in camera placement help\n"
                    "5. **Curriculum**: Start with fixed positions, gradually increase randomization\n\n"
                    "Critical: ensure the training set has at least 5 distinct object positions per object class."
                ),
                "expected_impact": (
                    "Should eliminate positional memorization. Model should generalize to novel object "
                    "placements within the workspace. Expect >3x improvement in position-varied evaluations."
                ),
            },
            "low_dataset_diversity": {
                "title": "Insufficient dataset diversity risks memorization",
                "interpretation": (
                    "The training dataset has extremely low diversity in one or more dimensions: "
                    "object positions (std < 15px), backgrounds (near-identical), lighting, or task instructions. "
                    "The model is likely memorizing a single scenario rather than learning a generalizable policy."
                ),
                "fix": (
                    "Expand dataset diversity systematically:\n"
                    "1. **Object positions**: Collect 50+ episodes with objects in random positions across the workspace\n"
                    "2. **Backgrounds**: Collect episodes in 5+ different environments or with varied tablecloths/surfaces\n"
                    "3. **Lighting**: Vary lighting conditions (bright, dim, directional, diffuse)\n"
                    "4. **Task instructions**: Use 5+ paraphrases of each task (e.g., 'grab the block', 'pick up the lego', 'take the brick')\n"
                    "5. **Object instances**: Use 3+ instances of each object class (different colors, sizes)\n\n"
                    "Minimum viable dataset: 200 episodes with varied conditions. Current dataset likely needs 5-10x more diversity."
                ),
                "expected_impact": (
                    "Should eliminate memorization artifacts and produce a policy that generalizes "
                    "across environments. This is the highest-impact fix — without data diversity, "
                    "no amount of model tuning will help."
                ),
            },
            "action_attention_misalignment": {
                "title": "Action dimensions attend to background instead of objects",
                "interpretation": (
                    "Spatial action dimensions (x, y, yaw) should attend primarily to the manipulation "
                    "target to compute correct motion vectors. Instead, they attend to background regions, "
                    "suggesting the model computes actions from background spatial cues rather than "
                    "object-relative positioning. This will fail in novel environments."
                ),
                "fix": (
                    "Address the root cause (likely background dependence + spatial shortcuts):\n"
                    "1. Fix background attribution first (see above) — this is usually the upstream cause\n"
                    "2. **Object-relative action representation**: Transform actions to be relative to detected object positions\n"
                    "3. **Attention regularization**: Add a loss term penalizing background attention for spatial action dims\n"
                    "4. Consider **object-centric architectures** that explicitly route object features to action heads\n\n"
                    "Quick diagnostic: this finding usually resolves automatically when background dependence is fixed."
                ),
                "expected_impact": (
                    "Once background dependence is addressed, action dimensions should shift attention "
                    "to objects. Expect improved precision in reaching and grasping."
                ),
            },
            "dead_state_pathway": {
                "title": "Proprioceptive state input is unused",
                "interpretation": (
                    "The gradient ratio between vision and proprioceptive state shows the model "
                    "effectively ignores the state input. For manipulation tasks, proprioceptive state "
                    "(joint angles, gripper width) provides critical feedback for closed-loop control."
                ),
                "fix": (
                    "Investigate and fix the state pathway:\n"
                    "1. **Verify state normalization**: Ensure proprioceptive inputs are properly normalized (zero-mean, unit-variance)\n"
                    "2. **Check state embedding**: Verify the state projection layer has non-zero gradients during training\n"
                    "3. **State dropout**: Add `dropout=0.1` on the vision pathway during training to force state usage\n"
                    "4. **State prediction auxiliary loss**: Add a loss that predicts next state from current state + action\n"
                    "5. **Verify data pipeline**: Ensure state values are correctly loaded and not all zeros"
                ),
                "expected_impact": (
                    "Should enable closed-loop control. Vision provides 'what to do' while state provides "
                    "'where I am' — both are needed for precise manipulation."
                ),
            },
            "gripper_fixation": {
                "title": "Model fixates on robot gripper instead of manipulation target",
                "interpretation": (
                    "The model attends primarily to the robot gripper, which is always present and "
                    "highly salient, rather than the objects being manipulated. This suggests the model "
                    "is tracking its own end-effector position from vision rather than planning relative "
                    "to the target object."
                ),
                "fix": (
                    "Reduce gripper dominance:\n"
                    "1. **Gripper masking augmentation**: Randomly mask/occlude the gripper region during 30% of training frames\n"
                    "2. **Use wrist camera**: Add a wrist-mounted camera that doesn't see the gripper\n"
                    "3. **Proprioceptive bypass**: If gripper position comes from state, the model shouldn't need to extract it from vision\n"
                    "4. **Increase object saliency**: Add object-level augmentation to make targets more visually distinct"
                ),
                "expected_impact": (
                    "Should shift attention from gripper to manipulation targets. Model should use "
                    "state input for self-localization and vision for target localization."
                ),
            },
            "cross_attention_diffuse": {
                "title": "Action expert cross-attention is near-uniform",
                "interpretation": (
                    "The action expert's cross-attention over the VLM's KV cache is nearly uniform, "
                    "meaning it queries all tokens equally rather than selectively attending to "
                    "task-relevant information. This suggests the expert hasn't learned to extract "
                    "specific features for action generation."
                ),
                "fix": (
                    "Improve cross-attention specificity:\n"
                    "1. **Temperature scaling**: Add learnable temperature to cross-attention softmax\n"
                    "2. **Longer training**: Cross-attention patterns often sharpen with more training\n"
                    "3. **Attention dropout**: Apply dropout to cross-attention to encourage sparse patterns\n"
                    "4. **Verify KV cache content**: Ensure the VLM prefix contains meaningful, differentiated tokens"
                ),
                "expected_impact": (
                    "Sharper cross-attention should improve action quality by focusing the expert "
                    "on the most task-relevant VLM representations."
                ),
            },
            "language_insensitivity": {
                "title": "Model ignores language conditioning",
                "interpretation": (
                    "Changing the task instruction produces negligible change in the model's attention "
                    "or actions, suggesting the model does not use language input. With only 1 unique "
                    "task string in training, the model has learned to ignore language entirely."
                ),
                "fix": (
                    "Enable language grounding:\n"
                    "1. **Multi-task training**: Train on 5+ distinct tasks with different instructions\n"
                    "2. **Paraphrase augmentation**: Use 5+ paraphrases per task during training\n"
                    "3. **Contrastive language loss**: Add a loss that ensures different instructions produce different actions\n"
                    "4. **Language dropout**: Randomly blank the instruction during 10% of training to create gradient signal for language use\n"
                    "5. **Verify tokenization**: Ensure the instruction is correctly tokenized and within the model's vocabulary"
                ),
                "expected_impact": (
                    "Should enable multi-task capability and instruction following. "
                    "Critical for deployment where tasks are specified via language."
                ),
            },
            "temporal_attention_instability": {
                "title": "Temporally unstable attention patterns",
                "interpretation": (
                    "Attention patterns jump erratically between consecutive frames, rather than "
                    "smoothly tracking objects over time. This may cause jerky or inconsistent actions "
                    "during execution."
                ),
                "fix": (
                    "Improve temporal consistency:\n"
                    "1. **Temporal augmentation**: Apply consistent augmentations across consecutive frames\n"
                    "2. **Frame stacking**: Use multi-frame input to provide temporal context\n"
                    "3. **Temporal smoothing loss**: Penalize large changes in attention between consecutive frames\n"
                    "4. **Video pretraining**: SmolVLM2 supports video — ensure video pretraining features are leveraged"
                ),
                "expected_impact": (
                    "Should produce smoother action sequences and more stable manipulation behavior."
                ),
            },
            "connector_bottleneck": {
                "title": "Connector bottleneck drops task-critical information",
                "interpretation": (
                    "The pixel-shuffle connector compresses 1024 SigLIP patches into 64 VLM tokens "
                    "(93.75% compression). Some task-relevant information is being lost in this process. "
                    "If the lost information includes target object features, the VLM and action expert "
                    "cannot recover it."
                ),
                "fix": (
                    "Mitigate connector information loss:\n"
                    "1. **Increase connector output tokens**: If feasible, use 128 or 256 output tokens\n"
                    "2. **Fine-tune the connector**: Train the connector projection with a larger learning rate\n"
                    "3. **Skip connections**: Add a direct path from SigLIP patches to the action expert\n"
                    "4. **Learnable pooling**: Replace pixel-shuffle with attention-based pooling that can prioritize object regions"
                ),
                "expected_impact": (
                    "Should preserve more spatial detail through the connector, improving "
                    "fine-grained manipulation accuracy."
                ),
            },
        }

        for anomaly in anomalies:
            expert = _EXPERT_DB.get(anomaly.type, {})
            evidence = anomaly.evidence or {}

            # Build observation from actual evidence data
            obs_parts = [f"Anomaly detected: {anomaly.type}."]
            if anomaly.type == "high_background_attribution":
                bg_share = evidence.get("background_share", 0)
                signal = evidence.get("signal", "unknown")
                obs_parts.append(f"{signal} shows {bg_share:.1%} background attribution.")
            elif anomaly.type == "low_object_attribution":
                low_objs = evidence.get("low_objects", {})
                for obj, val in low_objs.items():
                    obs_parts.append(f"{obj}: {val:.1%} attribution.")
            elif anomaly.type == "spatial_shortcut":
                ratio = evidence.get("positional_baseline_ratio", evidence.get("correlation", 0))
                obs_parts.append(f"Positional baseline similarity: {ratio:.2f}.")
            elif anomaly.type == "low_dataset_diversity":
                lp = evidence.get("low_position_objects", [])
                ut = evidence.get("unique_task_strings", "?")
                if lp:
                    obs_parts.append(f"Low position variance objects: {', '.join(lp)}.")
                obs_parts.append(f"Unique task strings: {ut}.")
            elif anomaly.type == "action_attention_misalignment":
                dims = evidence.get("background_attributed_dims", [])
                obs_parts.append(f"Background-attributed action dims: {', '.join(dims)}.")
            else:
                obs_parts.append(json.dumps(evidence))

            finding = Finding(
                id=f"f{len(findings)+1}",
                severity=anomaly.severity,
                title=expert.get("title", anomaly.description),
                observation=" ".join(obs_parts),
                test_description="See counterfactual results below.",
                test_result="",
                interpretation=expert.get("interpretation", anomaly.description),
                fix=expert.get("fix", f"Address the {anomaly.type} anomaly based on the diagnostic data."),
                expected_impact=expert.get("expected_impact", "Improvement expected upon resolution."),
                evidence_refs=[anomaly.type],
            )

            # Link counterfactual results
            for h in hypotheses:
                if anomaly.type in h.supporting_anomalies:
                    r = result_map.get(h.id)
                    if r:
                        finding.test_description = f"Ran {r.test_type} counterfactual"
                        finding.test_result = (
                            f"Action delta L2: {r.action_delta_l2:.4f} "
                            f"({'Confirmed' if r.confirmed else 'Not confirmed'}: "
                            f"{'significant' if r.action_delta_l2 > 0.02 else 'minimal'} sensitivity to perturbation)"
                        )
                        finding.evidence_refs.append(h.id)
                    break

            findings.append(finding)

        return findings

    def _rule_based_narrative(self, anomalies: list[Anomaly],
                               findings: list[Finding],
                               spatial_object_diagnosis=None) -> str:
        """Generate expert-quality narrative without LLM."""
        critical = [f for f in findings if f.severity == "critical"]
        warnings = [f for f in findings if f.severity == "warning"]
        info_items = [f for f in findings if f.severity == "info"]

        lines = ["## Diagnostic Summary\n"]

        if spatial_object_diagnosis is not None:
            lines.append(f"**Spatial vs object verdict**: {spatial_object_diagnosis.summary}\n")

        if critical:
            lines.append(f"**{len(critical)} critical issue(s) detected.**\n")
            for f in critical:
                lines.append(f"- **{f.title}**: {f.observation.split('. ', 1)[-1] if '. ' in f.observation else f.observation}")
        if warnings:
            lines.append(f"\n**{len(warnings)} warning(s) detected.**\n")
            for f in warnings:
                lines.append(f"- **{f.title}**: {f.observation.split('. ', 1)[-1] if '. ' in f.observation else f.observation}")
        if info_items:
            lines.append(f"\n**{len(info_items)} informational finding(s).**\n")
            for f in info_items:
                lines.append(f"- {f.title}")
        if not findings:
            lines.append("No critical issues detected. The model appears reasonably healthy based on the available signals.")
            return "\n".join(lines)

        # Prioritized action plan
        lines.append("\n## Recommended Action Plan\n")
        lines.append("Prioritized by expected impact:\n")

        priority_order = [
            "low_dataset_diversity",      # Data is always #1
            "high_background_attribution", # Environment generalization
            "spatial_shortcut",            # Memorization
            "low_object_attribution",      # Object grounding
            "language_insensitivity",      # Multi-task capability
            "dead_state_pathway",          # Closed-loop control
            "gripper_fixation",            # Attention allocation
            "action_attention_misalignment", # Usually follows from above
            "connector_bottleneck",        # Architecture
            "cross_attention_diffuse",     # Architecture
            "temporal_attention_instability", # Temporal
        ]
        anomaly_types = [a.type for a in anomalies]
        seen = set()
        step = 1
        for atype in priority_order:
            if atype in anomaly_types and atype not in seen:
                seen.add(atype)
                matching = [f for f in findings if atype in f.evidence_refs]
                if matching:
                    f = matching[0]
                    # Extract just the first sentence of the fix
                    fix_first = f.fix.split("\n")[0] if "\n" in f.fix else f.fix
                    lines.append(f"{step}. **{f.title}** — {fix_first}")
                    step += 1

        # Overall health assessment
        lines.append("\n## Overall Assessment\n")
        if len(critical) >= 3:
            lines.append(
                "The model has **multiple critical issues** that will prevent reliable deployment. "
                "The most impactful fix is expanding dataset diversity — without it, the model will "
                "continue to memorize specific scenarios rather than learning generalizable manipulation skills. "
                "Address data diversity first, then retrain with augmentation to fix background dependence."
            )
        elif len(critical) >= 1:
            lines.append(
                "The model has critical issues that should be addressed before deployment. "
                "Focus on the highest-priority fixes above and re-run diagnostics after retraining."
            )
        elif warnings:
            lines.append(
                "The model shows some concerning patterns but may be functional in controlled settings. "
                "Address the warnings above to improve robustness before expanding to new environments."
            )
        else:
            lines.append("The model appears healthy. Consider edge-case testing before deployment.")

        return "\n".join(lines)
