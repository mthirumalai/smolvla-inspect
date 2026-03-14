"""LLM prompt templates for diagnostic hypothesis formation and report synthesis."""

from __future__ import annotations

# Reuse architecture context from existing prompt templates
_ARCH_CONTEXT = """SmolVLA is a vision-language-action policy built on SmolVLM2-500M-Video-Instruct. Its pipeline:
1. **SigLIP vision encoder** (12 layers, 12 heads): processes 512x512 images into 32x32 = 1024 patch embeddings.
2. **Pixel-shuffle connector**: compresses 1024 SigLIP patches into 64 VLM tokens (16x spatial compression). This is the information bottleneck.
3. **VLM text model** (16 layers, 15 heads): builds a prefix KV cache from [vision tokens | language tokens | proprioceptive state tokens].
4. **Action expert** (16 layers, 8 heads): generates actions via flow matching (iterative denoising). Cross-attends to the VLM's KV cache.

Key facts:
- Self-attention shows what the vision encoder "looks at" — a representation signal, not causal.
- Cross-attention shows what the action expert queries — closer to what drives actions.
- GradCAM and saliency are causal attribution — they show what actually influences output.
- The connector is a critical bottleneck: 1024 patches → 64 tokens = 93.75% spatial compression."""


HYPOTHESIS_PROMPT = """You are an expert robotics ML researcher diagnosing a vision-language-action model.

**Architecture:**
{arch_context}

**Task instruction:** "{task_string}"
**Detected objects in scene:** {detected_objects}
**Dataset diversity summary:** {diversity_summary}

**Diagnostic Matrix (attribution mass: signal type × region):**
{matrix_markdown}

**Spatial vs Object Diagnosis:**
{spatial_object_summary}

**Semantic Probe:**
{semantic_probe_summary}

**QK Decomposition:**
{qk_probe_summary}

**Detected Anomalies:**
{anomalies_json}

**Available counterfactual tests:**
__COUNTERFACTUAL_TESTS__

For each hypothesis, provide:
1. A clear description of what you hypothesize and why (cite specific matrix values and anomalies)
2. Which counterfactual test would best verify or reject it (or "none" if matrix evidence is sufficient)
3. The test parameters as a JSON object
4. What you expect to see if the hypothesis is true
5. What you expect to see if the hypothesis is false

Output ONLY a JSON array. Each element must have these exact keys:
- "id": string (e.g. "h1", "h2", ...)
- "description": string
- "confidence": float 0-1
- "supporting_anomalies": list of anomaly type strings
- "test_type": string (one of the counterfactual names above, or "none")
- "test_params": object (parameters for the counterfactual, or empty object for "none")
- "expected_if_true": string
- "expected_if_false": string
- "confirms_on_change": boolean (true if the hypothesis is confirmed when the model's action CHANGES after the perturbation, false if confirmed when the action stays the SAME — e.g. false for weak-grounding or language-blindness hypotheses where insensitivity is the signal)

Maximum 7 hypotheses, ranked by severity/confidence. Output valid JSON only, no markdown fences.

Example output format (abbreviated):
[{{"id": "h1", "description": "Model relies on background texture...", "confidence": 0.8, "supporting_anomalies": ["high_background_attribution"], "test_type": "background_substitution", "test_params": {{"replacement": "gray"}}, "expected_if_true": "Action delta > 0.05", "expected_if_false": "Action delta < 0.01", "confirms_on_change": true}}, {{"id": "h2", "description": "Model ignores language instruction...", "confidence": 0.7, "supporting_anomalies": ["language_blindness"], "test_type": "task_string_swap", "test_params": {{"replacement_task": "do nothing"}}, "expected_if_true": "Action unchanged despite new instruction", "expected_if_false": "Action changes, showing language sensitivity", "confirms_on_change": false}}]"""


SYNTHESIS_PROMPT = """You are an expert robotics ML researcher writing a diagnostic report for a vision-language-action model.

**Architecture:**
{arch_context}

**Task instruction:** "{task_string}"
**Detected objects in scene:** {detected_objects}

**Diagnostic Matrix:**
{matrix_markdown}

**Spatial vs Object Diagnosis:**
{spatial_object_summary}

**Semantic Probe:**
{semantic_probe_summary}

**QK Decomposition:**
{qk_probe_summary}

**Anomalies detected:** {anomalies_summary}

**Dataset diversity:** {diversity_summary}

**Hypotheses tested:**
{hypotheses_with_results}

Based on ALL evidence above, write a structured diagnostic report. For each confirmed finding:

1. **Title** — short, specific (e.g., "Spatial shortcut learning from fixed object positions")
2. **Severity** — "critical", "warning", or "info"
3. **Observation** — What was observed in the diagnostic data (cite specific numbers)
4. **Test** — What counterfactual test was run (if any) and what happened
5. **Interpretation** — What this means for the model's behavior and generalization
6. **Fix** — Specific, actionable recommendation with concrete details (dataset changes, training config, augmentation strategies). Include example configurations or code snippets where helpful.
7. **Expected impact** — What improvement to expect from the fix

Also include unconfirmed hypotheses as lower-severity findings with the caveat that they need verification.

Output ONLY a JSON array of findings. Each element must have these exact keys:
- "id": string (e.g. "f1", "f2", ...)
- "severity": string ("critical"|"warning"|"info")
- "title": string
- "observation": string
- "test_description": string (or "No counterfactual test run" if none)
- "test_result": string (or "N/A" if no test)
- "interpretation": string
- "fix": string
- "expected_impact": string
- "evidence_refs": list of strings (anomaly IDs, hypothesis IDs)

After the JSON array, on a new line write "---NARRATIVE---" followed by a full narrative synthesis (2-4 paragraphs) summarizing the overall model health, key issues, and prioritized action plan. This narrative should be written for a robotics engineer who wants to understand what's wrong and what to do about it.

IMPORTANT: Start your response IMMEDIATELY with the JSON array `[` character. No preamble text.

Example output format (abbreviated):
[{{"id": "f1", "severity": "critical", "title": "Background texture dependence", "observation": "GradCAM shows 78% background attribution", "test_description": "Ran background_substitution with gray fill", "test_result": "Action delta L2 = 0.17, confirming background reliance", "interpretation": "Model uses background texture as a spatial reference frame", "fix": "Add background augmentation: random crops, color jitter (brightness=0.3, contrast=0.3), and synthetic background swaps during training", "expected_impact": "Should reduce background attribution from 78% to <30%", "evidence_refs": ["high_background_attribution", "h1"]}}]
---NARRATIVE---
The model shows critical dependence on background features..."""


def build_hypothesis_prompt(task_string: str, detected_objects: list[str],
                            diversity_summary: str, matrix_markdown: str,
                            anomalies_json: str,
                            spatial_object_summary: str = "Spatial-vs-object diagnosis unavailable.",
                            semantic_probe_summary: str = "Semantic patch-to-text probe unavailable.",
                            qk_probe_summary: str = "QK decomposition unavailable.",
                            ) -> str:
    """Build the hypothesis formation prompt with all context filled in."""
    from .registry import format_counterfactual_prompt_section
    prompt = HYPOTHESIS_PROMPT.format(
        arch_context=_ARCH_CONTEXT,
        task_string=task_string,
        detected_objects=", ".join(detected_objects),
        diversity_summary=diversity_summary,
        matrix_markdown=matrix_markdown,
        spatial_object_summary=spatial_object_summary,
        semantic_probe_summary=semantic_probe_summary,
        qk_probe_summary=qk_probe_summary,
        anomalies_json=anomalies_json,
    )
    prompt = prompt.replace("__COUNTERFACTUAL_TESTS__", format_counterfactual_prompt_section())
    return prompt


def build_synthesis_prompt(task_string: str, detected_objects: list[str],
                           matrix_markdown: str, anomalies_summary: str,
                           diversity_summary: str,
                           hypotheses_with_results: str,
                           spatial_object_summary: str = "Spatial-vs-object diagnosis unavailable.",
                           semantic_probe_summary: str = "Semantic patch-to-text probe unavailable.",
                           qk_probe_summary: str = "QK decomposition unavailable.",
                           ) -> str:
    """Build the synthesis/report prompt with all context filled in."""
    return SYNTHESIS_PROMPT.format(
        arch_context=_ARCH_CONTEXT,
        task_string=task_string,
        detected_objects=", ".join(detected_objects),
        matrix_markdown=matrix_markdown,
        spatial_object_summary=spatial_object_summary,
        semantic_probe_summary=semantic_probe_summary,
        qk_probe_summary=qk_probe_summary,
        anomalies_summary=anomalies_summary,
        diversity_summary=diversity_summary,
        hypotheses_with_results=hypotheses_with_results,
    )


def format_diversity_summary(diversity) -> str:
    """Format a DatasetDiversityReport as a concise text summary for LLM prompts."""
    if diversity is None:
        return "Dataset diversity analysis not available."

    lines = []
    lines.append(f"Sampled {diversity.num_episodes_sampled} episodes.")

    if diversity.object_position_stats:
        for obj, stats in diversity.object_position_stats.items():
            std = stats.get("centroid_std", "N/A")
            lines.append(f"  {obj}: centroid std = {std}px")

    lines.append(f"Background diversity score: {diversity.background_diversity_score:.2f}")
    lines.append(f"Lighting: mean brightness = {diversity.lighting_stats.get('mean_brightness', 'N/A')}, "
                 f"contrast variance = {diversity.lighting_stats.get('contrast_variance', 'N/A')}")

    if diversity.task_string_diversity:
        lines.append(f"Task strings: {diversity.task_string_diversity.get('unique_count', 'N/A')} unique")

    return "\n".join(lines)


def format_anomalies_json(anomalies) -> str:
    """Format anomalies list as JSON string for the prompt."""
    import json
    items = []
    for a in anomalies:
        items.append({
            "type": a.type,
            "severity": a.severity,
            "description": a.description,
            "evidence": a.evidence,
        })
    return json.dumps(items, indent=2)


def format_hypotheses_with_results(hypotheses, cf_results, *, skip_reason: str = "") -> str:
    """Format hypotheses with their counterfactual results for the synthesis prompt.

    Args:
        hypotheses: List of Hypothesis objects.
        cf_results: List of CounterfactualResult objects.
        skip_reason: Why counterfactuals weren't run. Common values:
            "skipped" — user passed --skip-counterfactuals
            "no_model" — policy not loaded (post-hoc mode)
            "budget_exhausted" — max_counterfactuals reached
            "" — unknown / default
    """
    lines = []
    result_map = {r.hypothesis_id: r for r in cf_results}

    for h in hypotheses:
        lines.append(f"### Hypothesis {h.id}: {h.description}")
        lines.append(f"Confidence: {h.confidence:.2f}")
        lines.append(f"Supporting anomalies: {', '.join(h.supporting_anomalies)}")

        result = result_map.get(h.id)
        if result:
            lines.append(f"Test: {result.test_type}")
            lines.append(f"Action delta L2: {result.action_delta_l2:.4f}")
            lines.append(f"GradCAM shift: {result.gradcam_shift:.4f}")
            lines.append(f"Confirmed: {result.confirmed}")
            metrics = result.metrics or {}
            if metrics:
                metric_parts = []
                for key, value in metrics.items():
                    if isinstance(value, float):
                        metric_parts.append(f"{key}={value:.4f}")
                    else:
                        metric_parts.append(f"{key}={value}")
                lines.append(f"Probe metrics: {', '.join(metric_parts)}")
            if result.attribution_shift_per_region:
                shifts = ", ".join(f"{k}: {v:+.3f}" for k, v in result.attribution_shift_per_region.items())
                lines.append(f"Attribution shifts: {shifts}")
        elif h.test_type == "none":
            lines.append("Test: none (matrix evidence sufficient)")
        else:
            # Context-aware reason for why the test wasn't run
            if skip_reason == "skipped":
                reason = "not run — counterfactuals skipped by user"
            elif skip_reason == "no_model":
                reason = "not run — no model loaded (post-hoc mode)"
            elif skip_reason == "budget_exhausted":
                reason = "not run — counterfactual budget exhausted"
            else:
                reason = "not run"
            lines.append(f"Test: {h.test_type} ({reason})")
            lines.append(f"Confidence reduced to: {h.confidence:.2f}")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Triage selection prompt (adaptive triage)
# ---------------------------------------------------------------------------

TRIAGE_SELECTION_PROMPT = """You are an expert robotics ML researcher. Based on cheap diagnostic signals, decide which expensive signals to collect next.

**Architecture:**
{arch_context}

**Task instruction:** "{task_string}"
**Detected objects:** {detected_objects}

**Cheap signals already collected:**
{cheap_signals_summary}

**Available expensive signals (each has a compute cost):**
__EXPENSIVE_SIGNALS__

**Budget:** Select at most {max_signals} expensive signals. Prioritise signals that would best diagnose potential issues visible in the cheap signals.

Output ONLY a JSON array of signal names to collect. Example: ["gradcam_siglip", "vision_vs_state", "temporal_trajectory"]
No explanation needed, just the JSON array."""


def build_triage_selection_prompt(
    task_string: str,
    detected_objects: list[str],
    cheap_signals_summary: str,
    max_signals: int = 4,
) -> str:
    """Build the adaptive triage selection prompt."""
    from .registry import format_signals_prompt_section
    prompt = TRIAGE_SELECTION_PROMPT.format(
        arch_context=_ARCH_CONTEXT,
        task_string=task_string,
        detected_objects=", ".join(detected_objects),
        cheap_signals_summary=cheap_signals_summary,
        max_signals=max_signals,
    )
    prompt = prompt.replace("__EXPENSIVE_SIGNALS__", format_signals_prompt_section())
    return prompt
