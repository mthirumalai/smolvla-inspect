"""Default LLM prompt templates per analysis type."""

TEMPLATES: dict[str, str] = {
    # -- Per-visualization prompts --
    "single_viz_self_attention": """You are analyzing self-attention heatmaps from a SmolVLA vision-language-action model.

**Model**: {model_id}
**Dataset**: {dataset_id}
**Task**: "{task_string}"
**Episode**: {episode_idx}, Frames: {num_frames}
**Method**: {method}

The attached images show attention heatmaps overlaid on camera frames. Warmer colors (red/yellow) indicate higher attention.

Please analyze:
1. What regions does the model attend to? Are they task-relevant (e.g., the object to manipulate, the target location)?
2. How does the attention focus shift across frames during the episode?
3. Are there any unexpected patterns (attending to background, ignoring the object)?
4. What does this suggest about the model's understanding of the task?

Be specific about spatial locations and frame-to-frame changes.""",

    "single_viz_per_head": """You are analyzing per-head attention patterns from the SigLIP vision encoder (12 heads).

**Model**: {model_id}
**Task**: "{task_string}"

Each cell shows one attention head's heatmap for the first frame. Entropy values indicate how focused (low) or diffuse (high) each head is.

Please analyze:
1. Which heads appear specialized (focused on specific regions)?
2. Which heads are diffuse or potentially redundant?
3. What might each specialized head be tracking (object, gripper, background structure)?
4. Are there any dead heads (near-uniform attention)?
5. Recommendations for pruning or fine-tuning.""",

    "single_viz_cross_attention": """You are analyzing cross-attention heatmaps showing where the action expert attends to in the visual input.

**Model**: {model_id}
**Task**: "{task_string}"

Cross-attention shows how the action generation module queries the vision representation. Green heatmaps show cross-attention focus.

Please analyze:
1. Does the action expert focus on task-relevant regions?
2. How does cross-attention differ from self-attention?
3. Does focus shift appropriately during the manipulation sequence?
4. Any signs of the action expert ignoring important visual information?""",

    "single_viz_gradcam_siglip": """You are analyzing GradCAM attribution maps from the SigLIP vision encoder.

**Model**: {model_id}
**Task**: "{task_string}"

GradCAM shows which visual regions causally drive the action predictions (gradient-weighted activation maps). Unlike attention, this reflects actual causal influence on outputs.

Please analyze:
1. Which regions causally drive action predictions?
2. How does GradCAM differ from attention heatmaps? What does this tell us?
3. Is there significant attribution to background regions (potential spurious correlations)?
4. Does the causal attribution align with task-relevant objects?""",

    "single_viz_vision_vs_state": """You are analyzing the balance between vision and proprioceptive state inputs.

**Model**: {model_id}
**Task**: "{task_string}"

The chart shows gradient norms for vision vs. state (proprioception) inputs across frames, indicating which modality more strongly influences action predictions.

Data:
{chart_data}

Please analyze:
1. Is the vision/state balance appropriate for this manipulation task?
2. Does the balance shift during the episode? When and why might this happen?
3. Signs of over-reliance on one modality?
4. Recommendations for improving the balance if needed.""",

    "single_viz_language_diff": """You are analyzing language-conditional GradCAM differences.

**Model**: {model_id}
**Original task**: "{task_string}"
**Alternative task**: (auto-selected contrasting instruction)

The visualization shows GradCAM with the original instruction, the alternative instruction, and their difference. Red regions attend more with original, blue with alternative.

Please analyze:
1. Does the model meaningfully differentiate between instructions?
2. Are the difference regions semantically meaningful (e.g., different objects)?
3. Does this suggest the model truly understands language, or uses it superficially?
4. Implications for multi-task generalization.""",

    "single_viz_per_action_dim": """You are analyzing per-action-dimension GradCAM maps.

**Model**: {model_id}
**Task**: "{task_string}"
**Action dimensions**: {action_dim_names}

Each sub-image shows which visual regions drive a specific action dimension (e.g., x-translation, rotation, gripper).

Please analyze:
1. Do different action dimensions attend to different spatial regions?
2. Are the attributions physically sensible (e.g., gripper dim attends to gripper area)?
3. Which dimensions have the strongest/weakest visual dependence?
4. What does this reveal about the model's action generation strategy?""",

    # -- Health report prompt --
    "health": """You are interpreting a model health diagnostics report for a SmolVLA vision-language-action model.

**Model**: {model_id}

Health metrics:
- **WeightWatcher spectral alpha**: Layer-wise spectral analysis (alpha near 2-4 is healthy)
- **Attention entropy**: How focused vs. diffuse attention is (ratio to uniform maximum)
- **Head redundancy**: Cosine similarity between heads (high = redundant)

Data:
{health_data}

Please provide:
1. Overall health assessment (healthy / warning / critical)
2. Specific issues identified (collapsed heads, high redundancy, spectral anomalies)
3. Which components (vision encoder, VLM, expert) show the most concern
4. Actionable fine-tuning recommendations to address any issues""",

    # -- Comparison prompt --
    "comparison": """You are comparing results across multiple runs of the SmolVLA attention analyzer.

Runs being compared:
{run_descriptions}

Please analyze:
1. Key differences between runs (attention patterns, GradCAM, health metrics)
2. Which run shows better task understanding and why
3. What might explain the differences (training data, fine-tuning, model architecture)
4. Specific recommendations for improvement based on the comparison""",
}


def get_template(analysis_type: str) -> str:
    return TEMPLATES.get(analysis_type, TEMPLATES.get("single_viz_self_attention", ""))


def format_template(template: str, **kwargs) -> str:
    """Safe string formatting that ignores missing keys."""
    for key, val in kwargs.items():
        template = template.replace(f"{{{key}}}", str(val))
    return template
