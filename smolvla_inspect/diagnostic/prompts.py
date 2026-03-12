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

**Detected Anomalies:**
{anomalies_json}

**Available counterfactual tests:**
- background_substitution: Replace background with gray/noise/blur. Tests if model relies on background features.
  Params: {{replacement: "gray"|"noise"|"blur"}}
- object_relocation: Digitally move an object to a different position. Tests if model tracks objects or uses spatial shortcuts.
  Params: {{target_object: str, shift_pixels: [dx, dy]}}
- lighting_perturbation: Shift brightness/contrast. Tests lighting robustness.
  Params: {{brightness_delta: float, contrast_delta: float}}
- object_recolor: Change object color via HSV shift. Tests if model uses color cues.
  Params: {{target_object: str, hue_shift: float}}
- none: No test needed — hypothesis is already well-supported by the matrix data alone.

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

Maximum 5 hypotheses, ranked by severity/confidence. Output valid JSON only, no markdown fences."""


SYNTHESIS_PROMPT = """You are an expert robotics ML researcher writing a diagnostic report for a vision-language-action model.

**Architecture:**
{arch_context}

**Task instruction:** "{task_string}"
**Detected objects in scene:** {detected_objects}

**Diagnostic Matrix:**
{matrix_markdown}

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

Output valid JSON for the findings array, then the separator, then the narrative text."""


def build_hypothesis_prompt(task_string: str, detected_objects: list[str],
                            diversity_summary: str, matrix_markdown: str,
                            anomalies_json: str) -> str:
    """Build the hypothesis formation prompt with all context filled in."""
    return HYPOTHESIS_PROMPT.format(
        arch_context=_ARCH_CONTEXT,
        task_string=task_string,
        detected_objects=", ".join(detected_objects),
        diversity_summary=diversity_summary,
        matrix_markdown=matrix_markdown,
        anomalies_json=anomalies_json,
    )


def build_synthesis_prompt(task_string: str, detected_objects: list[str],
                           matrix_markdown: str, anomalies_summary: str,
                           diversity_summary: str,
                           hypotheses_with_results: str) -> str:
    """Build the synthesis/report prompt with all context filled in."""
    return SYNTHESIS_PROMPT.format(
        arch_context=_ARCH_CONTEXT,
        task_string=task_string,
        detected_objects=", ".join(detected_objects),
        matrix_markdown=matrix_markdown,
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


def format_hypotheses_with_results(hypotheses, cf_results) -> str:
    """Format hypotheses with their counterfactual results for the synthesis prompt."""
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
            if result.attribution_shift_per_region:
                shifts = ", ".join(f"{k}: {v:+.3f}" for k, v in result.attribution_shift_per_region.items())
                lines.append(f"Attribution shifts: {shifts}")
        elif h.test_type == "none":
            lines.append("Test: none (matrix evidence sufficient)")
        else:
            lines.append(f"Test: {h.test_type} (not run — model not available)")
            lines.append(f"Confidence reduced to: {h.confidence:.2f}")

        lines.append("")

    return "\n".join(lines)
