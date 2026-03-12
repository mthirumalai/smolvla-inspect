from __future__ import annotations

import asyncio
import json
import os
import numpy as np

from .models import (
    DiagnosticReport, DiagnosticMatrix, Hypothesis, CounterfactualResult,
    EvidenceEntry, Finding, SceneSegmentation, DatasetDiversityReport, Anomaly,
)
from .registry import REGISTRY
from .scene import parse_task_objects, detect_objects, segment_scene, analyze_dataset_diversity
from .regions import attribute_to_regions
from .matrix import build_diagnostic_matrix, detect_anomalies
from .prompts import (
    build_hypothesis_prompt, build_synthesis_prompt,
    format_diversity_summary, format_anomalies_json, format_hypotheses_with_results,
)
from ..data import _resolve_task_string, get_episode_frames, build_policy_batch_from_sample


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

        def _progress(phase: str, detail: str = ""):
            if progress_callback:
                progress_callback(phase, detail)
            print(f"  [{phase}] {detail}")

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

        self.evidence_log.append(EvidenceEntry(
            phase="scene_understanding",
            primitive_name="scene.segment_scene",
            data={"regions": scene.region_names(), "num_objects": len(scene.objects)},
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

        # ── Phase 3: Triage — Collect Model Primitive Outputs ──
        _progress("triage", "Collecting model signals...")
        if self.post_hoc:
            signals = self._load_existing_signals()
        else:
            signals = self._run_triage(sample)

        _progress("triage", f"Collected signals: {list(signals.keys())}")

        # ── Phase 4: Build Diagnostic Matrix ────────────────────
        _progress("matrix", "Building diagnostic matrix...")
        frames_for_matrix = [first_frame]  # Use first frame for now

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
        )

        internals = signals.get("model_internals")
        anomalies = detect_anomalies(matrix, internals)
        _progress("matrix", f"Detected {len(anomalies)} anomalies")

        self.evidence_log.append(EvidenceEntry(
            phase="matrix",
            primitive_name="matrix.detect_anomalies",
            data={"anomaly_types": [a.type for a in anomalies]},
        ))

        # ── Phase 5: LLM Hypothesis Formation ──────────────────
        _progress("hypothesize", "Forming hypotheses via LLM...")
        hypotheses = await self._form_hypotheses(
            task_string, scene, diversity, matrix, anomalies)
        _progress("hypothesize", f"Formed {len(hypotheses)} hypotheses")

        # ── Phase 6: Run Agent-Chosen Counterfactuals ───────────
        cf_results: list[CounterfactualResult] = []
        max_cf = self.config.get("max_counterfactuals", 3)

        if self.policy is not None:
            for hypothesis in hypotheses:
                if hypothesis.test_type == "none":
                    continue
                if len(cf_results) >= max_cf:
                    break

                _progress("counterfactuals", f"Running {hypothesis.test_type} for {hypothesis.id}...")
                try:
                    result = self._run_counterfactual(
                        hypothesis, sample, scene)
                    cf_results.append(result)
                    self.evidence_log.append(EvidenceEntry(
                        phase="counterfactual",
                        primitive_name=f"counterfactual.{hypothesis.test_type}",
                        data={"hypothesis_id": hypothesis.id,
                              "confirmed": result.confirmed,
                              "action_delta_l2": result.action_delta_l2},
                    ))
                except Exception as e:
                    _progress("counterfactuals", f"  Failed: {e}")
        else:
            _progress("counterfactuals", "Skipping (no model loaded)")
            for h in hypotheses:
                if h.test_type != "none":
                    h.confidence *= 0.6

        _progress("counterfactuals", f"Completed {len(cf_results)} counterfactual tests")

        # ── Phase 7: LLM Synthesis ─────────────────────────────
        _progress("synthesis", "Synthesizing report via LLM...")
        findings, narrative = await self._synthesize_report(
            task_string, scene, diversity, matrix, anomalies,
            hypotheses, cf_results)
        _progress("synthesis", f"Generated {len(findings)} findings")

        # ── Build Final Report ──────────────────────────────────
        metadata = {
            "task_string": task_string,
            "episode_idx": self.episode_idx,
            "image_key": self.image_key,
            "device": self.device,
            "post_hoc": self.post_hoc,
        }
        try:
            metadata["model_id"] = getattr(self.policy.config, "pretrained_model_name_or_path", "unknown") if self.policy else "none"
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
            anomalies=anomalies,
            hypotheses=hypotheses,
            counterfactual_results=cf_results,
            findings=findings,
            llm_synthesis=narrative,
        )

        _progress("complete", "Diagnostic report ready")
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

    def _run_triage(self, sample) -> dict:
        """Run model primitives to collect signals (integrated mode)."""
        signals = {}

        # Import existing pipeline functions
        try:
            from ..cli import extract_attention_maps
            from ..gradient import (
                compute_saliency_map,
                compute_gradcam_connector_maps,
                compute_vision_vs_state_maps,
                compute_language_conditional_maps,
            )
            from ..heatmap import compute_positional_baseline
            from ..data import find_vision_encoder
            from ..capture import SigLIPAttentionCapture
        except ImportError as e:
            print(f"  Warning: Could not import triage modules: {e}")
            return signals

        # Run attention extraction
        try:
            from ..capture import SigLIPAttentionCapture, ActionVisionAttentionCapture
            vision_encoder = find_vision_encoder(self.policy)
            if vision_encoder is not None:
                attn_capture = SigLIPAttentionCapture(vision_encoder)
                attn_capture.register_hooks()

                # Get frames
                frames_data = get_episode_frames(
                    self.dataset, self.episode_idx,
                    self.config.get("num_frames", 4), self.image_key)

                heatmaps = []
                for frame_idx, img_tensor in frames_data:
                    frame_sample = self.dataset[frame_idx]
                    batch, _ = build_policy_batch_from_sample(
                        frame_sample, self.policy, self.device,
                        dataset=self.dataset, image_map=self.image_map)
                    with __import__('torch').no_grad():
                        self.policy.select_action(batch)
                    last_attn = attn_capture.get_last_layer_attention()
                    if last_attn is not None:
                        from ..heatmap import compute_patch_attention_scores, attention_to_heatmap
                        scores = compute_patch_attention_scores(last_attn)
                        # Determine grid size from vision encoder config
                        try:
                            vc = vision_encoder.config
                            gs = vc.image_size // vc.patch_size
                        except Exception:
                            gs = 32
                        try:
                            vc = vision_encoder.config
                            target_hw = (gs * vc.patch_size, gs * vc.patch_size)
                        except Exception:
                            target_hw = (512, 512)
                        hm = attention_to_heatmap(scores, (gs, gs), target_hw)
                        heatmaps.append(hm)
                    attn_capture.reset_maps()

                if heatmaps:
                    signals["attention"] = heatmaps

                # Positional baseline
                try:
                    baseline = compute_positional_baseline(
                        vision_encoder, attn_capture, self.device, "last-layer",
                        input_hw=(512, 512))
                    if baseline is not None:
                        signals["positional_baseline"] = baseline
                except Exception:
                    pass

                attn_capture.remove_hooks()
        except Exception as e:
            print(f"  Warning: Attention extraction failed: {e}")

        # Run gradient-based attribution on first frame
        try:
            from ..gradient import compute_saliency_map, _run_forward_with_grad
            frame_sample = self.dataset[self._get_first_frame_idx()]

            # GradCAM SigLIP
            try:
                from ..gradient import compute_gradcam_siglip_maps
            except ImportError:
                pass

            # Saliency
            try:
                saliency = compute_saliency_map(
                    self.policy, frame_sample, self.dataset, self.image_key,
                    self.device, image_map=self.image_map)
                if saliency is not None:
                    signals["saliency"] = [saliency]
            except Exception as e:
                print(f"  Warning: Saliency failed: {e}")

            # Vision vs State
            try:
                vs_result = compute_vision_vs_state_maps(
                    self.policy, frame_sample, self.dataset, self.device,
                    image_map=self.image_map)
                if vs_result is not None:
                    signals["vision_vs_state"] = [vs_result]
            except Exception as e:
                print(f"  Warning: Vision vs state failed: {e}")

        except Exception as e:
            print(f"  Warning: Gradient attribution failed: {e}")

        return signals

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

        result = primitive.fn(**params)
        result.hypothesis_id = hypothesis.id
        result.confirmed = self._evaluate_result(hypothesis, result)
        return result

    def _evaluate_result(self, hypothesis: Hypothesis, result: CounterfactualResult) -> bool:
        """Evaluate whether a counterfactual result confirms the hypothesis."""
        # Simple heuristic: if action prediction changed significantly, the
        # model was relying on whatever we perturbed
        if result.action_delta_l2 > 0.02:
            # Significant change — model was sensitive to the perturbation
            # For most hypotheses about reliance on X, this confirms
            return True
        return False

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
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

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
        response = await client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def _fallback_llm_response(self, prompt: str) -> str:
        """Generate a rule-based response when no LLM is available."""
        if "hypothesis" in prompt.lower() or "hypothesize" in prompt.lower():
            return '[]'
        return '[]---NARRATIVE---\nDiagnostic analysis completed. LLM synthesis unavailable — review the diagnostic matrix and anomalies above for details.'

    async def _form_hypotheses(self, task_string: str, scene: SceneSegmentation,
                                diversity, matrix: DiagnosticMatrix,
                                anomalies: list[Anomaly]) -> list[Hypothesis]:
        """Use LLM to form hypotheses from the diagnostic data."""
        prompt = build_hypothesis_prompt(
            task_string=task_string,
            detected_objects=scene.region_names(),
            diversity_summary=format_diversity_summary(diversity),
            matrix_markdown=matrix.to_markdown(),
            anomalies_json=format_anomalies_json(anomalies),
        )

        response = await self._call_llm(prompt)

        # Parse JSON response
        hypotheses = []
        try:
            # Try to find JSON array in response
            text = response.strip()
            # Find first [ and last ]
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                items = json.loads(text[start:end + 1])
                for item in items:
                    hypotheses.append(Hypothesis(
                        id=item.get("id", f"h{len(hypotheses)+1}"),
                        description=item.get("description", ""),
                        confidence=float(item.get("confidence", 0.5)),
                        supporting_anomalies=item.get("supporting_anomalies", []),
                        test_type=item.get("test_type", "none"),
                        test_params=item.get("test_params", {}),
                        expected_if_true=item.get("expected_if_true", ""),
                        expected_if_false=item.get("expected_if_false", ""),
                    ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  Warning: Failed to parse LLM hypotheses: {e}")

        # If LLM failed, generate rule-based hypotheses from anomalies
        if not hypotheses:
            hypotheses = self._rule_based_hypotheses(anomalies, scene)

        return hypotheses[:self.config.get("max_hypotheses", 5)]

    def _rule_based_hypotheses(self, anomalies: list[Anomaly],
                                scene: SceneSegmentation) -> list[Hypothesis]:
        """Generate hypotheses from anomalies without LLM."""
        hypotheses = []
        target_objects = [r for r in scene.region_names() if r != "background" and r != "robot gripper"]
        target = target_objects[0] if target_objects else "object"

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

        return hypotheses

    async def _synthesize_report(self, task_string: str, scene: SceneSegmentation,
                                  diversity, matrix: DiagnosticMatrix,
                                  anomalies: list[Anomaly],
                                  hypotheses: list[Hypothesis],
                                  cf_results: list[CounterfactualResult],
                                  ) -> tuple[list[Finding], str]:
        """Use LLM to synthesize findings and narrative from all evidence."""
        prompt = build_synthesis_prompt(
            task_string=task_string,
            detected_objects=scene.region_names(),
            matrix_markdown=matrix.to_markdown(),
            anomalies_summary=format_anomalies_json(anomalies),
            diversity_summary=format_diversity_summary(diversity),
            hypotheses_with_results=format_hypotheses_with_results(hypotheses, cf_results),
        )

        response = await self._call_llm(prompt)

        findings = []
        narrative = ""

        try:
            # Split on separator
            if "---NARRATIVE---" in response:
                json_part, narrative = response.split("---NARRATIVE---", 1)
                narrative = narrative.strip()
            else:
                json_part = response

            # Parse findings JSON
            text = json_part.strip()
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                items = json.loads(text[start:end + 1])
                for item in items:
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
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  Warning: Failed to parse LLM synthesis: {e}")

        # If LLM failed, generate rule-based findings
        if not findings:
            findings = self._rule_based_findings(anomalies, hypotheses, cf_results)

        if not narrative:
            narrative = self._rule_based_narrative(anomalies, findings)

        return findings, narrative

    def _rule_based_findings(self, anomalies: list[Anomaly],
                              hypotheses: list[Hypothesis],
                              cf_results: list[CounterfactualResult]) -> list[Finding]:
        """Generate findings without LLM."""
        findings = []
        result_map = {r.hypothesis_id: r for r in cf_results}

        for anomaly in anomalies:
            finding = Finding(
                id=f"f{len(findings)+1}",
                severity=anomaly.severity,
                title=anomaly.description,
                observation=f"Anomaly detected: {anomaly.type}. {json.dumps(anomaly.evidence)}",
                test_description="See counterfactual results below.",
                test_result="",
                interpretation=anomaly.description,
                fix="Review the diagnostic matrix and anomaly details for specific recommendations.",
                expected_impact="Varies based on the fix applied.",
                evidence_refs=[anomaly.type],
            )

            # Link counterfactual results
            for h in hypotheses:
                if anomaly.type in h.supporting_anomalies:
                    r = result_map.get(h.id)
                    if r:
                        finding.test_description = f"Ran {r.test_type} counterfactual"
                        finding.test_result = f"Action delta L2: {r.action_delta_l2:.4f}, Confirmed: {r.confirmed}"
                    break

            findings.append(finding)

        return findings

    def _rule_based_narrative(self, anomalies: list[Anomaly],
                               findings: list[Finding]) -> str:
        """Generate narrative without LLM."""
        critical = [a for a in anomalies if a.severity == "critical"]
        warnings = [a for a in anomalies if a.severity == "warning"]

        lines = ["## Diagnostic Summary\n"]
        if critical:
            lines.append(f"**{len(critical)} critical issue(s) detected.**\n")
            for a in critical:
                lines.append(f"- {a.description}")
        if warnings:
            lines.append(f"\n**{len(warnings)} warning(s) detected.**\n")
            for a in warnings:
                lines.append(f"- {a.description}")
        if not critical and not warnings:
            lines.append("No critical issues detected. The model appears reasonably healthy based on the available signals.")

        return "\n".join(lines)
