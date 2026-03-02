"""Default LLM prompt templates per analysis type."""

_ARCH_CONTEXT = """SmolVLA is a vision-language-action policy built on SmolVLM2-500M-Video-Instruct. Its pipeline:
1. **SigLIP vision encoder** (12 layers, 12 heads): processes 512x512 images into 32x32 = 1024 patch embeddings. Pure patch-based (no CLS token).
2. **Pixel-shuffle connector**: compresses 1024 SigLIP patches into 64 VLM tokens (16x spatial compression). This is the information bottleneck.
3. **VLM text model** (16 layers, 15 heads): builds a prefix KV cache from [vision tokens | language tokens | proprioceptive state tokens].
4. **Action expert** (16 layers, 8 heads): generates actions via flow matching (iterative denoising). Cross-attends to the VLM's KV cache — it queries the vision+language+state representation to decide what action to take.

Key architectural facts to keep in mind:
- Self-attention shows what the vision encoder "looks at" — this is a representation signal, not a causal one.
- Cross-attention shows what the action expert queries from the VLM context — this is closer to what directly drives actions.
- GradCAM and saliency are causal attribution methods — they show what actually influences the output, not just what the model represents.
- The connector is a critical bottleneck: 1024 patches compressed to 64 tokens means 93.75% of spatial positions are merged."""

TEMPLATES: dict[str, str] = {
    # -- Per-visualization prompts --
    "single_viz_self_attention": """You are an expert robotics ML researcher analyzing self-attention heatmaps from a SmolVLA vision-language-action model.

**Run context:**
- Model: {model_id}
- Dataset: {dataset_id}
- Task instruction: "{task_string}"
- Episode: {episode_idx}, Frames: {num_frames}
- Aggregation method: {method}

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are looking at:**
The attached images show self-attention heatmaps from the SigLIP vision encoder overlaid on camera frames. Warmer colors (red/yellow) = higher attention weight. These maps show which spatial regions the vision encoder represents most strongly — think of this as "what the model is looking at" before any action decision is made. The aggregation method "{method}" determines how multi-layer attention is combined (rollout = multiplicative propagation through layers; last-layer = final layer only; all-layers = mean across all layers).

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Spatial focus
- What specific regions does the model attend to in each frame? Name them concretely (e.g., "left edge of the blue cube", "gripper fingers", "table surface near the target").
- Are these regions task-relevant given the instruction "{task_string}"? What objects, landmarks, or spatial relationships should the model attend to for this task?
- Is there attention "leaking" to irrelevant background regions (walls, table edges, distant objects)? Quantify roughly how much of the attention mass falls on task-irrelevant areas.

### Temporal dynamics
- How does the attention focus shift frame-to-frame? Map the attention centroid trajectory to the likely manipulation phases (approach, grasp, transport, place).
- Reference the centroid drift values — are shifts smooth and continuous (good) or erratic/jumpy (concerning)?
- Reference temporal stability — high stability suggests consistent focus, but could also mean the model isn't adapting to changing scene dynamics.

### Statistical interpretation
- What does the entropy tell you? Low entropy = peaked/focused (good if on the right region, bad if fixated on one spot). High entropy = diffuse (could mean the model is "confused" or genuinely needs broad scene awareness).
- What does the coverage metric indicate? Low coverage + low entropy = laser-focused. Low coverage + high entropy = noisy. High coverage + moderate entropy = broad but structured.
- Are there frames where statistics change dramatically? These transitions often align with task phase changes.

### Red flags
- Any frames where the model appears to ignore the manipulated object entirely?
- Signs of "texture bias" — attending to high-frequency patterns rather than semantically meaningful regions?
- Attention collapse — all mass on a single patch or uniform distribution across all patches?

## Recommendations

### Confidence assessment
- Rate your confidence in this model's visual grounding: strong / moderate / weak. Justify with specific evidence from the heatmaps and stats.

### If attention is well-focused and task-relevant:
- What aspects of the attention pattern suggest robust generalization? What might break under distribution shift (new objects, lighting changes, camera angles)?

### If attention shows problems:
- **Spatial misalignment**: Recommend attention supervision losses, guided attention during fine-tuning, or data augmentation strategies that could help.
- **Temporal instability**: Suggest frame-to-frame consistency regularization, temporal smoothing, or whether this indicates a data quality issue (jerky demonstrations).
- **Background distraction**: Recommend background randomization during training, cropping strategies, or domain randomization.
- **Entropy anomalies**: If too peaked, suggest temperature scaling or attention dropout. If too diffuse, suggest training with auxiliary spatial losses.

### Broader implications
- Given what self-attention reveals about the model's representation, what should the researcher examine next? (e.g., "Cross-attention maps to see if the action expert uses these representations", "GradCAM to check causal influence vs. mere attention", "Per-head analysis to find dead or redundant heads")
- Does the attention pattern suggest the model has learned a useful visual representation for this task, or is it relying on shortcuts?""",

    "single_viz_per_head": """You are an expert robotics ML researcher analyzing per-head attention patterns from a SigLIP vision encoder.

**Run context:**
- Model: {model_id}
- Task instruction: "{task_string}"

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are looking at:**
Each cell in the attached grid shows one of the 12 SigLIP attention heads' heatmaps for the first frame of the episode. In multi-head attention, different heads can specialize to detect different visual features — some may track objects, others spatial structure, others may be effectively dead (uniform attention). Entropy values appear per head: low entropy = focused/specialized, high entropy = diffuse/general.

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Head specialization inventory
For each of the 12 heads, provide a brief characterization:
- What spatial region or feature does this head focus on? (e.g., "gripper area", "object of interest", "background edges", "top-center of frame", "uniform/dead")
- Classify each head as: **task-specialized** (focused on task-relevant region), **structure-specialized** (focused on scene structure like edges or surfaces), **positional** (focused on a fixed spatial location regardless of content), **diffuse** (spread broadly), or **dead** (near-uniform, entropy close to maximum).

### Head diversity and redundancy
- How many functionally distinct groups of heads are there? (e.g., "3 heads track the object, 2 track the gripper, 4 are diffuse, 3 are dead")
- Reference inter-head similarity scores — are any pairs of heads near-duplicates (cosine similarity > 0.9)? These represent wasted capacity.
- What is the effective number of unique attention patterns? (12 heads with 6 redundant = effectively 6 heads)

### Task relevance assessment
- What fraction of heads are directly task-relevant (attending to the object, gripper, or goal location)?
- Are there heads that attend to potentially useful auxiliary information (table edges for spatial reference, background for scene context)?
- Are there heads that attend to nothing useful (true dead weight)?

### Comparison to known patterns
- In well-trained vision transformers, we expect: some heads for local features, some for global context, some for position encoding. Does this model follow that pattern?
- Are there any unusual or surprising specializations?

## Recommendations

### Head health assessment
- Rate the overall head utilization: excellent (>10 useful heads), good (7-9), concerning (4-6), poor (<4).
- Identify the most and least valuable heads by number.

### Dead head remediation
- If dead heads are found: recommend strategies (re-initialization + continued training, attention dropout during fine-tuning, pruning + knowledge distillation).
- Estimate how much capacity is wasted and whether reducing head count could improve efficiency without hurting performance.

### Redundancy reduction
- If redundant head pairs exist: suggest head pruning, diversity-encouraging regularization (orthogonality loss between heads), or architectural changes.
- Would reducing from 12 to N heads likely maintain quality? What's your recommended N?

### Missing specializations
- Given the task "{task_string}", are there visual features you'd expect a head to specialize on but none do? (e.g., if no head tracks the gripper, that's concerning for a manipulation task)
- Recommend training interventions to encourage the missing specializations.

### Architecture recommendations
- Is 12 heads appropriate for this model size and task complexity, or does the utilization pattern suggest it's over/under-provisioned?
- Would the model benefit from multi-scale attention (different heads operating at different spatial resolutions)?""",

    "single_viz_cross_attention": """You are an expert robotics ML researcher analyzing cross-attention heatmaps from the action expert in a SmolVLA model.

**Run context:**
- Model: {model_id}
- Task instruction: "{task_string}"

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are looking at:**
These heatmaps (green colormap) show cross-attention weights from the action expert as it queries the VLM's KV cache. This is the most direct window into what visual information the action generation module actually uses to produce motor commands. Unlike self-attention (which shows what the vision encoder represents), cross-attention shows what the action expert *pulls from* the combined vision+language+state context. High cross-attention to a region means that region's representation strongly influences the next action prediction.

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Action-vision grounding
- Where does the action expert look when generating actions? Map the focus regions to physical scene elements (gripper, object, target, obstacles).
- Is the expert attending to the right things for "{task_string}"? For a pick-and-place task, we'd expect: object during approach, gripper+object during grasp, target during transport.
- How concentrated vs. distributed is the cross-attention? A very peaked pattern suggests the expert is "decisive" about what matters; a diffuse pattern could mean uncertainty or that the expert genuinely needs broad context.

### Temporal dynamics through task phases
- How does cross-attention shift through the episode? Identify which frames correspond to which manipulation phases and whether the attention follows the expected progression.
- Reference centroid drift — is the expert's focus smoothly tracking the task progression or jumping erratically?
- Are there frames where the expert's attention "collapses" (all on one point) or "explodes" (completely diffuse)? These often correspond to moments of policy uncertainty.

### Cross-attention vs. self-attention comparison
- The self-attention (SigLIP) shows what the vision encoder represents. Cross-attention shows what the action expert consumes. If these differ substantially, it means the expert is selectively reading from the visual representation — which could be good (selective reading) or bad (ignoring useful information).
- Are there task-relevant regions that self-attention highlights but cross-attention ignores? This would suggest an information utilization gap.

### Statistical interpretation
- Entropy: low values mean the expert is confident about which visual region matters; very low might mean over-reliance on a single region.
- Coverage: what fraction of the visual field is the expert using? For manipulation, moderate coverage (20-40%) is typical; very low suggests tunnel vision, very high suggests the expert hasn't learned to be selective.
- Frame-to-frame stability vs. adaptiveness: the expert should be both stable within task phases and adaptive at phase transitions.

## Recommendations

### Policy quality assessment
- Based on the cross-attention patterns, rate the action expert's visual grounding: strong / moderate / weak.
- Is the expert using the visual context efficiently, or is there evidence of "lazy" attention (always looking at the same spot regardless of task phase)?

### If cross-attention is misaligned:
- Recommend action-conditioned attention supervision during training.
- Suggest curriculum strategies: start with simple tasks where the correct attention target is obvious, then progress to complex ones.
- Consider whether the VLM's representation is the bottleneck (the information the expert needs might not be well-represented in the KV cache).

### If cross-attention is well-aligned:
- What failure modes should be tested? (e.g., "Does the expert still attend correctly if the object is in an unusual position?", "What happens with distractors?")
- Robustness recommendations: evaluate with visual perturbations, object substitutions, or adversarial placement.

### Connector bottleneck analysis
- The connector compresses 1024 patches → 64 tokens. Is there evidence that the expert's cross-attention is limited by this compression? (e.g., attending to a broad region when it should be precise, because fine spatial detail was lost)
- Recommend examining GradCAM connector maps to check if spatial information survives the bottleneck.

### Next diagnostic steps
- What should the researcher look at next to deepen understanding of the action expert's behavior?""",

    "single_viz_gradcam_siglip": """You are an expert robotics ML researcher analyzing GradCAM attribution maps from the SigLIP vision encoder in a SmolVLA model.

**Run context:**
- Model: {model_id}
- Task instruction: "{task_string}"

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are looking at:**
GradCAM (Gradient-weighted Class Activation Mapping) shows which spatial regions in the SigLIP encoder causally drive the model's action predictions. This is computed by backpropagating from the action output through the entire network to the vision encoder activations, then weighting activations by their gradients. Unlike attention maps (which show what the model "looks at"), GradCAM reveals what actually *causes* the output — a region with low attention but high GradCAM means it's influencing actions through a pathway attention maps don't capture.

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Causal attribution mapping
- Which regions have the strongest causal influence on actions? Name them concretely relative to the scene (object, gripper, surface, landmarks).
- Are the causally important regions aligned with what's task-relevant for "{task_string}"? A well-grounded model should attribute to the object being manipulated, the gripper, and the goal location.
- How localized vs. diffuse is the attribution? Sharp peaks suggest the model has learned precise spatial features; broad attribution suggests it relies on holistic scene context.

### GradCAM vs. attention comparison
- Where do GradCAM and attention maps agree? These are regions the model both represents strongly AND uses causally — the strongest evidence of grounded understanding.
- Where do they disagree?
  - High attention, low GradCAM = the model "looks at" this region but it doesn't influence actions. Could be a feature extraction artifact or an unused representation.
  - Low attention, high GradCAM = the model's action is influenced by this region through indirect pathways. This is often a sign of shortcut learning or unexpected causal paths.
- Quantify the overlap if the stats support it.

### Temporal causal dynamics
- How does causal attribution shift across frames? Does it track the task progression (approach → grasp → transport → place)?
- Are there frames with dramatically different attribution patterns? These "causal transitions" indicate moments where the model's decision basis changes.
- Reference centroid drift and stability metrics — is causal attribution smooth or erratic?

### Spurious correlation detection
- Is there significant GradCAM attribution to background regions, table textures, lighting patterns, or other task-irrelevant features? This is the primary red flag for spurious correlations.
- Use the entropy and Gini coefficient: high Gini (concentrated) on task-relevant regions is ideal. Low Gini (spread out) with background attribution is concerning.
- Could the model be relying on visual shortcuts (e.g., "the object is always in the same position" rather than understanding the task)?

## Recommendations

### Causal grounding assessment
- Rate the model's causal visual grounding: strong / moderate / weak. This is the most informative assessment of whether the model has learned the right visual features.

### Spurious correlation remediation
- If background attribution is found: recommend specific data augmentation (background randomization, texture randomization, domain randomization). Quantify how much background attribution exists.
- Suggest training with causal intervention: swapping backgrounds while keeping task-relevant objects to force the model to ignore irrelevant features.
- Consider whether the training data itself has correlations (e.g., object always on left side, specific table always used).

### Attribution quality improvements
- If attribution is too diffuse: suggest attention supervision, auxiliary spatial losses, or higher-resolution training.
- If attribution is too concentrated on a single point: suggest spatial regularization or data augmentation to encourage the model to use multiple cues.
- If GradCAM and attention are misaligned: investigate intermediate layers to find where the causal path diverges from the attention path.

### Generalization predictions
- Based on the GradCAM patterns, predict which aspects of the task the model will generalize well on and which will fail.
- What specific test-time variations would you recommend to validate these predictions? (new object colors, positions, backgrounds, lighting)

### Research directions
- What follow-up analyses would deepen understanding? (connector GradCAM for bottleneck analysis, per-action-dim attribution for motor decomposition, language-diff for instruction grounding)""",

    "single_viz_vision_vs_state": """You are an expert robotics ML researcher analyzing the balance between vision and proprioceptive state inputs in a SmolVLA model.

**Run context:**
- Model: {model_id}
- Task instruction: "{task_string}"

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are looking at:**
This chart shows gradient norm magnitudes for vision inputs vs. proprioceptive state inputs across frames of the episode. Gradient norms indicate how strongly each modality influences the action prediction at each timestep. A higher gradient norm means that modality is more "listened to" by the model for that frame. This is computed by backpropagating from the action output and measuring the gradient magnitude flowing into each input modality.

Vision inputs = image features from SigLIP → connector → VLM.
State inputs = robot proprioception (joint positions, velocities, gripper state) → state encoder → VLM.

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Modality balance overview
- What is the mean vision share across the episode? A ratio near 50:50 means balanced; skewed ratios indicate modality dominance.
- For the task "{task_string}", what balance would you expect? (e.g., precise grasping needs high vision share; following a known trajectory might lean on proprioception; approach phase might need vision, execution might need proprioception)
- Reference the max imbalance ratio — how extreme does the skew get at any individual frame?

### Temporal modality dynamics
- How does the vision/state balance change across the episode? Map changes to task phases:
  - **Approach phase**: vision should typically dominate (finding the object, planning the path)
  - **Contact/grasp phase**: proprioception often increases (detecting contact forces, confirming grasp)
  - **Transport phase**: can be mixed (vision for goal, proprioception for arm configuration)
  - **Release/place phase**: vision for placement accuracy, proprioception for gripper control
- Does the observed pattern match these expectations?
- Reference the trend direction and slope — is the balance shifting systematically across the episode?

### Anomaly detection
- Are there abrupt shifts in modality balance? These might correspond to contact events, task transitions, or policy uncertainty.
- Is one modality dominant throughout with near-zero contribution from the other? This would suggest the model hasn't learned to use both modalities.
- Are there frames where vision drops to near-zero? The model might be "closing its eyes" and relying purely on proprioception — concerning for visual manipulation tasks.

### Absolute vs. relative magnitudes
- Beyond the ratio, what are the absolute gradient norms? Very small absolute norms for both modalities might indicate the model's actions are insensitive to its inputs (potential training issue).
- Are there frames where both modalities have high gradients (model is integrating both) vs. frames where one is high and the other near-zero (switching between modalities)?

## Recommendations

### Balance assessment
- Rate the modality balance as: appropriate / vision-heavy / state-heavy / disconnected. Justify.

### If one modality dominates:
- **Vision-dominant**: The model may not be using proprioception effectively. Recommend:
  - Check if the state encoder is well-trained or has been frozen.
  - Introduce proprioceptive perturbation during training to force the model to attend to state.
  - Consider auxiliary losses that encourage state utilization.
  - Evaluate whether the task genuinely doesn't need proprioception (unlikely for manipulation).
- **State-dominant**: The model may be ignoring visual input. Recommend:
  - Check if the visual pipeline (SigLIP → connector → VLM) is passing useful gradients during training.
  - The connector bottleneck (1024 → 64 tokens) might be too aggressive, losing visual detail.
  - Introduce visual perturbation during evaluation to test robustness.
  - Consider whether training data had low visual diversity (model learned it can ignore vision).

### If balance is appropriate:
- What would break this balance? Predict failure modes under distribution shift.
- Recommend evaluation scenarios that stress-test each modality independently (e.g., occluding the camera, adding joint noise).

### Task-phase-specific recommendations
- For each phase where the balance seems wrong, suggest targeted interventions:
  - Phase-conditioned training with modality dropout.
  - Curriculum learning that introduces modality challenges progressively.
  - Attention supervision that encourages vision use during approach and proprioception use during contact.

### Broader architecture implications
- Does the modality balance suggest the model architecture is appropriately designed? Should the state encoder be larger/smaller? Should vision and state be fused differently?""",

    "single_viz_language_diff": """You are an expert robotics ML researcher analyzing language-conditional GradCAM differences in a SmolVLA model.

**Run context:**
- Model: {model_id}
- Original task instruction: "{task_string}"
- Alternative task: (auto-selected contrasting instruction)

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are looking at:**
This visualization compares GradCAM attribution maps under two different language instructions applied to the same visual scene. It shows:
1. GradCAM with the **original instruction** — where the model looks given the actual task.
2. GradCAM with an **alternative instruction** — where the model looks given a different/contrasting task.
3. The **difference map** — red regions attend more with the original instruction, blue regions attend more with the alternative.

This is a critical test of language grounding: a model that truly understands language instructions should shift its visual attention to different task-relevant regions depending on what it's asked to do. If the difference map is near-zero, the model is ignoring the language instruction and using a fixed visual strategy regardless of the task.

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Language grounding assessment
- How large are the differences between the two instruction conditions? Quantify: are we seeing large spatial shifts, or just subtle intensity changes in the same region?
- Are the difference regions semantically meaningful? For example, if one instruction says "pick up the red cube" and the other says "push the blue cylinder", the red regions should be near the red cube and the blue regions near the blue cylinder.
- Map the difference regions to physical objects/locations in the scene. Name them.

### Spatial analysis of each condition
- Under the original instruction "{task_string}", what does the model attend to? Is it the right target object?
- Under the alternative instruction, does the model shift to the correct alternative target?
- Are there regions that have high attribution under BOTH instructions (instruction-invariant features)? These might be general scene structure the model always relies on (table, workspace boundaries).

### Grounding quality classification
Classify the model into one of these categories:
- **Strong grounding**: Large, semantically correct spatial shifts between instructions. The model clearly attends to different objects/regions depending on the instruction.
- **Weak grounding**: Small differences that are in the right direction but not decisive. The model partially responds to language but still relies heavily on visual priors.
- **No grounding**: Near-zero differences. The model ignores the language instruction and uses a fixed visual strategy.
- **Confused grounding**: Differences exist but are semantically wrong (e.g., switching to background when told to pick up a different object).

### Frame-by-frame consistency
- Is the language grounding consistent across all frames, or does it degrade in certain phases?
- Are there frames where the model appears to "lose" the language instruction and revert to a default strategy?

## Recommendations

### If strong grounding:
- What are the limits of this grounding? Recommend testing with:
  - More similar instructions (e.g., "pick up the red cube" vs. "pick up the blue cube") to test fine-grained grounding.
  - Adversarial instructions ("pick up the cube" in a scene with no cube) to test robustness.
  - Paraphrased instructions to test language robustness beyond specific phrasings.
- The model likely generalizes well to new instructions — but verify with held-out task descriptions.

### If weak grounding:
- Recommend language-conditioned fine-tuning strategies:
  - Multi-task training with diverse instructions to force language dependence.
  - Instruction dropout during training (randomly replacing instructions with null/blank to explicitly teach the model to use them when present).
  - Auxiliary losses that penalize instruction-invariant behavior.
- Investigate whether the VLM text model is the bottleneck or if language features are lost at the connector.

### If no grounding:
- This is a critical failure for a language-conditioned policy. The model has learned a task-agnostic policy that ignores instructions.
- Recommend:
  - Verifying the language instruction reaches the action expert (check if the VLM encodes instructions differently — might be a connector bottleneck issue).
  - Training with forced language dependence (e.g., the correct action is ambiguous without the instruction).
  - Consider architectural changes to give language a more direct path to the action expert.

### Multi-task generalization prediction
- Based on the language grounding level, predict the model's ability to generalize to new instructions. Will it follow novel instructions or default to its most-trained behavior?
- What training data changes would most improve language grounding?""",

    "single_viz_per_action_dim": """You are an expert robotics ML researcher analyzing per-action-dimension GradCAM maps from a SmolVLA model.

**Run context:**
- Model: {model_id}
- Task instruction: "{task_string}"
- Action dimensions: {action_dim_names}

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are looking at:**
Each sub-image shows GradCAM attribution for a single action dimension. In robot manipulation, the action vector typically contains [x, y, z, roll, pitch, yaw, gripper] (or a subset). Each dimension may depend on different visual features — for example, the x-translation might depend on where the object is horizontally, while the gripper dimension depends on object size and grasp readiness. This visualization reveals whether the model has learned a decomposed visual strategy (different dims use different spatial features) or a monolithic one (all dims use the same region).

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Per-dimension attribution inventory
For each action dimension, describe:
- What spatial region does this dimension's attribution concentrate on?
- Is this physically sensible? (e.g., x-translation should depend on horizontal object position; gripper should depend on object shape/size and proximity)
- How strong is the attribution compared to other dimensions? Weak attribution might mean that dimension relies more on proprioception than vision.

### Spatial decomposition quality
- Do different action dimensions attend to meaningfully different regions? This is the key quality metric — reference the IoU overlap values between dimensions.
  - Low overlap (IoU < 0.3): good spatial decomposition, each dim extracts different visual information.
  - High overlap (IoU > 0.6): monolithic strategy, all dims use the same visual features.
- Group the dimensions by their spatial focus patterns. Which dimensions share visual features and which are independent?

### Physical plausibility assessment
- Map each dimension to expected visual dependencies:
  - Translation dims (x, y, z): should attend to object position relative to gripper.
  - Rotation dims (roll, pitch, yaw): should attend to object orientation and approach angle.
  - Gripper dim: should attend to object shape, size, and gripper-object proximity.
- Which dimensions match expectations and which don't?
- Are there dimensions with almost no visual attribution? These might be computed purely from proprioception (which could be correct for some dims).

### Strength ranking analysis
- Which dimensions have the strongest visual dependence? These are the dims the model relies on vision for most heavily.
- Which have the weakest? Are these dimensions that could reasonably be computed from proprioception alone, or is the model failing to use useful visual information?

## Recommendations

### Motor decomposition assessment
- Rate the model's visual-motor decomposition: well-decomposed / partially-decomposed / monolithic.

### If monolithic (high overlap):
- The model hasn't learned to extract dimension-specific visual features. Recommend:
  - Per-dimension auxiliary losses during training to encourage spatial specialization.
  - Disentangled action representations or factored action heads.
  - Data augmentation that varies spatial features independently per dimension.
- Consider whether the model architecture supports decomposition (does the action expert have enough capacity to learn specialized attention per dim?).

### If well-decomposed:
- Which dimensions have the best visual grounding? These are likely the most robust to visual perturbations.
- Which dimensions have unexpected patterns? Flag these for targeted evaluation.
- Recommend stress-testing each dimension independently (e.g., moving the object to test position-sensitive dims, rotating it to test orientation-sensitive dims).

### Dimension-specific recommendations
- For each dimension with problematic attribution: suggest specific interventions (targeted data augmentation, auxiliary spatial losses, architecture changes).
- For visually-weak dimensions: determine if they should depend more on vision or if proprioceptive dependence is appropriate for that dimension.

### Implications for action prediction quality
- Based on the attribution patterns, which action dimensions are likely most/least accurate?
- Which would fail first under visual distribution shift?""",

    "single_viz_gradcam_vlm_layers": """You are an expert robotics ML researcher analyzing layer-wise GradCAM attribution maps from the VLM decoder in a SmolVLA model.

**Run context:**
- Model: {model_id}
- Task instruction: "{task_string}"

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are looking at:**
Each sub-image shows GradCAM at a different layer of the VLM text model (16 layers total). This reveals how the visual representation is transformed as it flows through the decoder layers. Early layers typically capture low-level spatial features (edges, textures, positions), while later layers capture higher-level semantic features (object identity, spatial relationships, task-relevant structure). The progression from early to late layers tells us how the model builds its understanding of the scene for action generation.

The VLM processes a prefix of [vision tokens (64 after connector) | language tokens | state tokens], and the GradCAM shows which spatial positions in the vision tokens are most influential at each layer.

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Layer progression
- Describe how the attribution pattern evolves from layer 0 (earliest) to layer 15 (latest):
  - Early layers (0-4): What spatial features dominate? Are they low-level (edges, corners, textures) or already semantic?
  - Middle layers (5-10): Is there a transition from spatial to semantic? At which specific layer does the pattern shift?
  - Late layers (11-15): Are the attributions focused on task-relevant objects/regions? How abstract is the representation?

### Task-relevant emergence
- At which layer does the model first show clear attribution to the task-relevant object or region? This is the "grounding layer" — where visual meaning emerges.
- Is the transition gradual (attribution slowly sharpens onto the target) or abrupt (sudden shift at a specific layer)?
- After the grounding layer, does attribution remain stable or continue to evolve?

### Anomaly detection across layers
- Are there layers with very diffuse, near-uniform attribution? These might be "bottleneck layers" where spatial information is temporarily lost.
- Are there layers with collapsed attribution (all mass on one patch)? This could indicate a representational bottleneck or skip-connection issue.
- Do any late layers "regress" to early-layer patterns (losing spatial precision)? This would suggest representational degradation.

### Information flow assessment
- Is the spatial information preserved through all layers, or is there evidence of spatial resolution loss?
- Does the connector compression (1024 → 64 tokens) cause visible spatial coarsening in early layers?
- How do the final-layer attributions compare to the SigLIP GradCAM? Differences indicate how the VLM transforms the visual representation.

## Recommendations

### Representation quality assessment
- Rate the VLM's visual representation pipeline: strong / moderate / weak, based on whether it builds a clear, task-relevant representation across layers.

### Layer-specific interventions
- If early layers show poor spatial encoding: the connector may be losing critical spatial information. Recommend connector architecture changes (less aggressive compression, spatial position embeddings).
- If middle layers show a bottleneck: consider adding residual connections or skip connections across the bottleneck region.
- If late layers lose spatial precision: consider auxiliary spatial losses at the output layer to encourage spatial feature preservation.

### Training recommendations
- Based on the layer progression, at which layers would adding auxiliary supervision be most beneficial?
- Should any layers be frozen during fine-tuning? (e.g., if early layers already have good spatial representations, freeze them to prevent catastrophic forgetting)
- Would layer-wise learning rate schedules help? (lower rates for well-formed early layers, higher for undertrained late layers)

### Architecture insights
- Is 16 layers appropriate for this task complexity? Are there layers that appear redundant (near-identical attribution patterns in consecutive layers)?
- Would a shallower model suffice if the task-relevant representation emerges early?
- Recommendations for the VLM architecture based on what the layer analysis reveals.""",

    "single_viz_per_step_cross_attention": """You are an expert robotics ML researcher analyzing per-denoising-step cross-attention from the action expert in a SmolVLA model.

**Run context:**
- Model: {model_id}
- Task instruction: "{task_string}"

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are looking at:**
SmolVLA uses flow matching to generate actions — the action expert iteratively denoises a noisy action vector over multiple steps (typically 10 steps) to produce a clean action prediction. At each denoising step, the expert cross-attends to the VLM's KV cache (which contains vision + language + state context). This visualization shows the cross-attention pattern at each denoising step, revealing how the expert's visual queries evolve as it refines its action prediction from pure noise to a final action.

This is analogous to how diffusion models refine images — early steps establish coarse structure, later steps add fine detail. For actions: early steps establish the general action direction, later steps fine-tune the precise motor command.

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Denoising progression
- How does the cross-attention pattern change from step 0 (noisiest) to the final step (cleanest)?
- At step 0 (pure noise → first refinement): is the attention diffuse (model is "exploring" the visual context broadly) or already somewhat focused?
- At intermediate steps: does the focus progressively sharpen? At which step does the pattern "crystallize" into its final form?
- At the final step: how focused is the attention? Is it concentrated on task-relevant regions?

### Phase transitions
- Are there steps where the attention pattern changes dramatically between consecutive steps? These "phase transitions" indicate critical moments in the denoising process where the model's action hypothesis shifts.
- How many distinct attention phases are there? (e.g., "steps 0-3 are diffuse, steps 4-7 focus on the object, steps 8-9 narrow to the grasp point")
- Is the transition smooth or discontinuous?

### Spatial focus evolution
- What region does the expert attend to at each phase?
  - Early steps: broad scene awareness or already object-focused?
  - Middle steps: object-level or sub-object-level (e.g., specific part of the object)?
  - Final steps: precise grasp point, contact region, or still at object level?
- Does the attention trajectory (from broad to narrow) make physical sense for the task?

### Consistency and stability
- Is the denoising attention progression consistent across frames, or does it vary significantly per frame?
- Are there steps where the model appears to "hesitate" (attention oscillates between two regions)?
- Is the final step's attention pattern always consistent, or is there step-to-step noise even at the end?

## Recommendations

### Denoising quality assessment
- Rate the denoising visual grounding: strong (clear progression from diffuse to focused), moderate (some progression but noisy), weak (no clear progression or attention doesn't converge).

### If progression is well-structured:
- The model has learned a meaningful coarse-to-fine action refinement strategy.
- Recommend testing robustness: what happens with fewer denoising steps? Can the model produce good actions with 5 steps instead of 10?
- Could the number of denoising steps be reduced for inference speed without sacrificing action quality? Reference at which step the attention "converges."

### If progression is weak or absent:
- The model may not be effectively using the iterative refinement. Recommend:
  - Noise schedule tuning — the current schedule might not provide enough signal at intermediate steps.
  - Progressive denoising supervision during training (auxiliary losses at intermediate steps).
  - Check if the flow matching loss is appropriately balanced across denoising steps.

### If attention oscillates or is unstable:
- Suggest adding consistency regularization across denoising steps.
- Consider whether the VLM KV cache representation is ambiguous (the expert is uncertain because the visual features don't provide a clear answer).
- Test with higher-quality demonstrations to rule out data noise as the cause.

### Step count optimization
- Based on the convergence analysis, recommend the minimum number of denoising steps needed for this task.
- For real-time deployment, which steps could be skipped with minimal quality loss?
- Would step distillation (training a student to predict the final action in fewer steps) be feasible given the current attention patterns?""",

    "single_viz_saliency": """You are an expert robotics ML researcher analyzing saliency maps from a SmolVLA vision-language-action model.

**Run context:**
- Model: {model_id}
- Task instruction: "{task_string}"

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are analyzing:**
Saliency maps show the magnitude of input gradients — for each pixel in the input image, how much would a small perturbation to that pixel change the model's action output? This is the most fine-grained attribution method available, operating at pixel resolution rather than patch resolution (like GradCAM). High-saliency pixels are those the model is most sensitive to — changes there would most affect the predicted action.

Unlike GradCAM (which highlights *regions*), saliency can reveal pixel-level sensitivity patterns: edge detection, texture sensitivity, color boundaries, and high-frequency features the model relies on. However, saliency can also be noisy and sometimes highlights irrelevant high-frequency patterns.

**IMPORTANT:** The user is viewing standalone per-frame saliency heatmaps rendered with the inferno (yellow-orange) colormap — one heatmap per frame. They are NOT viewing a multi-row grid. Do NOT reference "Row N", grid rows, or other visualization types visible in any attached reference image. Focus your analysis entirely on the saliency data. If an episode dashboard image is attached for reference, use it only to understand the scene context (original frames), not to describe the grid layout.

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Spatial saliency patterns
- Which regions show the highest saliency? Are they concentrated on task-relevant objects or scattered across the scene?
- At pixel level, what features are highlighted? (object edges, color boundaries, texture patterns, surface markings)
- Is the saliency pattern "clean" (well-defined regions with clear boundaries) or "noisy" (scattered high-saliency pixels without clear structure)?

### Task relevance of salient features
- For the task "{task_string}", are the salient features the right ones? (e.g., for grasping, object edges and gripper proximity are relevant; background texture is not)
- Are there strongly salient regions that are clearly task-irrelevant? These indicate the model's sensitivity to visual features it shouldn't depend on.
- What fraction of total saliency falls on the primary task object vs. background vs. other objects?

### Temporal dynamics
- How does the saliency pattern change across frames? Is the "sensitive region" tracking the task progression?
- Reference centroid drift — are the salient regions moving smoothly as the manipulation progresses?
- Reference temporal stability — are there frames with dramatically different saliency patterns? These might indicate moments where the model's sensitivity shifts unexpectedly.

### Comparison with other attribution methods
- How does saliency compare with GradCAM and attention? Key relationships to consider:
  - Saliency + GradCAM + attention all agree: strongest evidence of genuine visual grounding.
  - High saliency but low GradCAM: the model is sensitive to pixel perturbations but the region's activations aren't strongly used — could indicate noise sensitivity.
  - High GradCAM but low saliency: the region matters at the feature level but not at the pixel level — typical of robust feature detection.
- Is the saliency pattern more or less noisy than GradCAM?

### Noise and artifact detection
- Are there "salt and pepper" saliency patterns (isolated high-saliency pixels with no spatial structure)? This often indicates gradient noise rather than meaningful feature detection.
- Is there high saliency on image compression artifacts, JPEG boundaries, or other non-semantic features?
- Does SmoothGrad (if used) significantly clean up the pattern? If so, the raw saliency has high noise.

## Recommendations

### Saliency quality assessment
- Rate the saliency quality: clean and informative / somewhat noisy but interpretable / too noisy to be useful.

### If saliency is clean and task-aligned:
- The model has learned pixel-level sensitivity to the right visual features.
- Test robustness by evaluating with: image noise, compression artifacts, slight camera shifts, lighting changes. If saliency is concentrated on robust features (edges, shapes), the model should be resilient; if on textures or colors, it may be fragile.

### If saliency is noisy:
- Recommend using SmoothGrad (averaging gradients over noisy inputs) to get a cleaner signal.
- High noise may indicate the model's gradients are unstable — consider gradient regularization during training.
- Evaluate whether the noise is random (harmless) or structured (indicating sensitivity to non-semantic features).

### If saliency highlights irrelevant features:
- The model is sensitive to visual features it shouldn't depend on. This is a robustness vulnerability.
- Recommend: input augmentation during training (noise, blur, color jitter) to reduce sensitivity to irrelevant features.
- Consider adversarial training to specifically reduce saliency on background regions.
- Test with visual perturbations in the high-saliency irrelevant regions to quantify the actual impact on action quality.

### Input preprocessing recommendations
- Based on the saliency pattern, would image preprocessing (cropping, masking, resolution changes) improve or hurt performance?
- Are there spatial regions that could be masked or downsampled without losing task-relevant information?""",

    "single_viz_gradcam_connector": """You are an expert robotics ML researcher analyzing GradCAM attribution maps from the pixel-shuffle connector in a SmolVLA model.

**Run context:**
- Model: {model_id}
- Task instruction: "{task_string}"

**Architecture context:**
""" + _ARCH_CONTEXT + """

**What you are looking at:**
The pixel-shuffle connector is the critical information bottleneck in SmolVLA — it compresses 1024 SigLIP vision patches (32x32 spatial grid) into just 64 VLM tokens (8x8 grid). This is a 16x spatial compression, meaning each output token must represent a 4x4 block of input patches.

GradCAM at the connector shows which spatial regions survive this compression and remain causally influential for action predictions. This is the most direct measure of the bottleneck's impact: if task-relevant regions have low connector GradCAM despite high SigLIP GradCAM, critical spatial information is being lost in the compression.

**Computed statistics for this visualization:**
{computed_stats}

---

**Response constraint: ≤350 words total. Be terse — 1-2 sentences per bullet. Skip sub-questions you lack data for.**

Please provide a structured analysis in two parts:

## Observations

### Spatial survival analysis
- Which regions have strong connector GradCAM? These are the spatial regions whose information successfully passes through the 16x compression bottleneck.
- Are task-relevant regions (object, gripper, goal) among the survivors? Or has the connector compressed away important spatial details?
- How does the spatial resolution of the attribution compare to SigLIP GradCAM? (Connector operates at 8x8 effective resolution vs. SigLIP's 32x32.)

### Bottleneck impact assessment
- Compare connector GradCAM to SigLIP GradCAM:
  - Regions preserved: both high → information passes through the bottleneck successfully.
  - Regions lost: high SigLIP GradCAM but low connector GradCAM → the connector is discarding causally important information.
  - Regions gained: low SigLIP GradCAM but high connector GradCAM → the connector may be amplifying features or the VLM layers are creating new spatial relevance.
- Quantify how much spatial information is preserved vs. lost. What fraction of high-SigLIP-GradCAM regions remain high in connector GradCAM?

### Spatial compression patterns
- Is the connector's compression spatially uniform (all regions equally compressed) or non-uniform (some regions preserved, others aggressively compressed)?
- Does the compression favor central regions over edges? Objects over background? This would indicate the connector has learned a content-aware compression strategy.
- Are there visible 4x4 block artifacts in the attribution (reflecting the pixel shuffle grouping)?

### Temporal dynamics
- Does the connector consistently preserve the same regions across frames, or does what survives the bottleneck change?
- Are there frames where task-critical information appears to be lost? These would be the most dangerous frames for action quality.

## Recommendations

### Bottleneck severity assessment
- Rate the connector bottleneck: minimal impact (task-relevant info preserved) / moderate impact (some loss) / severe impact (critical information lost).

### If information is being lost:
- **Architecture changes**: Consider reducing the compression ratio (e.g., 1024 → 128 instead of 64) or using a learned/adaptive compression that preserves task-relevant regions.
- **Positional encoding**: Add explicit spatial position embeddings to the connector output to preserve spatial information even after compression.
- **Multi-scale connector**: Use different compression ratios for different spatial regions (less compression near task-relevant areas).
- **Skip connections**: Add a direct connection from SigLIP to the action expert that bypasses the connector bottleneck.

### If information is well-preserved:
- The current compression ratio is sufficient for this task.
- Test with more complex scenes (multiple objects, clutter) to see if the bottleneck becomes limiting.
- Consider whether the connector could be more aggressive (fewer tokens) for inference speed without quality loss.

### Training recommendations
- Would training the connector with an auxiliary reconstruction loss help preserve more spatial information?
- Should the connector be fine-tuned with task-specific spatial attention (preserving regions that matter for manipulation)?
- Consider connector warm-up: pre-train the connector to reconstruct SigLIP features before end-to-end training.

### Quantitative validation
- Recommend measuring action prediction quality with different connector compression ratios (ablation study).
- Test whether adding spatial noise after the connector degrades performance differently in preserved vs. lost regions.""",

    # -- Health report prompt --
    "health": """You are an expert ML diagnostics specialist interpreting a model health report for a SmolVLA vision-language-action model.

**Model**: {model_id}

**Architecture context:**
""" + _ARCH_CONTEXT + """

**Health metrics being evaluated:**

1. **WeightWatcher spectral alpha** (per layer): Measures the power-law distribution of singular values in weight matrices.
   - alpha 2-4: healthy, well-trained layer with good generalization capacity.
   - alpha < 2: undertrained or poorly initialized.
   - alpha > 6: overtrained, potentially memorizing.
   - alpha >> 10: severely overtrained or numerically unstable.

2. **Attention entropy** (per component): Ratio of actual attention entropy to maximum possible entropy.
   - 0.0-0.2: highly focused (potentially collapsed — all attention on one token).
   - 0.2-0.5: focused and likely specialized.
   - 0.5-0.8: moderate, balanced attention.
   - 0.8-1.0: near-uniform (possibly dead/untrained attention heads).

3. **Head redundancy** (per component): Pairwise cosine similarity between attention heads.
   - < 0.3: diverse heads, good utilization of capacity.
   - 0.3-0.7: moderate redundancy.
   - > 0.7: high redundancy, heads are near-duplicates.
   - > 0.9: effectively wasted capacity.

**Data:**
{health_data}

---

**Response constraint: ≤400 words total. Be terse — flag issues only where metrics are clearly out of range.**

Please provide a structured health assessment:

## Component-by-Component Diagnosis

### SigLIP Vision Encoder (12 layers, 12 heads)
- Spectral health: Are layer alphas in the healthy 2-4 range? Flag any layers outside this range.
- Attention health: Which heads are well-specialized vs. dead vs. collapsed?
- Redundancy: Are the 12 heads diverse or are many redundant?
- Overall vision encoder grade: healthy / warning / critical.

### VLM Text Model (16 layers, 15 heads)
- Spectral health across layers. Are early/late layers differently affected?
- Attention patterns: Is the VLM building useful contextual representations?
- Head diversity within layers.
- Overall VLM grade: healthy / warning / critical.

### Action Expert (16 layers, 8 heads)
- Spectral health. Since this is the action generation component, overtrained layers are especially concerning.
- Attention health: Are heads focused (good for action generation) or diffuse?
- With only 8 heads, any dead or redundant heads represent significant wasted capacity.
- Overall expert grade: healthy / warning / critical.

## Cross-Component Analysis
- Is there a component that stands out as the weakest link?
- Are there signs of training imbalance (one component overtrained while another is undertrained)?
- Does the spectral profile suggest the components were trained jointly or separately?

## Recommendations

### Immediate actions (if critical issues found)
- Specific layers or heads that need intervention (re-initialization, pruning, focused fine-tuning).
- Training adjustments needed (learning rate, weight decay, warmup schedule).

### Medium-term improvements
- Head pruning recommendations (which redundant heads can be removed without quality loss).
- Architecture right-sizing (does each component have the right number of heads/layers?).
- Training strategy changes (curriculum, loss weighting, component-specific learning rates).

### Monitoring recommendations
- Which metrics should be tracked during future training?
- What thresholds should trigger re-evaluation?

## Overall Health Score
Provide a summary score: HEALTHY / WARNING / CRITICAL, with the primary justification.""",

    # -- Comparison prompt --
    "comparison": """You are comparing results across multiple runs of the SmolVLA attention analyzer.

Runs being compared:
{run_descriptions}

{computed_stats}

**Response constraint: ≤250 words total. Reference specific numbers. Be direct.**

Please analyze:
1. Key differences between runs (attention patterns, GradCAM, health metrics). Reference specific numerical deltas.
2. Which run shows better task understanding and why
3. What might explain the differences (training data, fine-tuning, model architecture)
4. Specific recommendations for improvement based on the comparison""",

    # -- Run insights (holistic per-run analysis) --
    "run_insights": """You are an expert robotics ML researcher providing a comprehensive analysis of a SmolVLA model inspection run.

**Run context:**
- Model: {model_id}
- Dataset: {dataset_id}
- Task instruction: "{task_string}"
- Episode: {episode_idx}, Frames: {num_frames}

**Architecture context:**
""" + _ARCH_CONTEXT + """

**Available statistics across all visualization types:**
{computed_stats}

The attached images show representative visualizations from this run.

---

**Response constraint: ≤500 words total. Synthesize across methods — don't repeat findings per method. Be specific, not exhaustive.**

Please provide a comprehensive three-part analysis:

## Observations

### Visual grounding quality
- How well does the model attend to task-relevant regions? Synthesize evidence from attention heatmaps, GradCAM, and saliency.
- Is there agreement between attention (what the model looks at) and GradCAM (what causally drives actions)? Disagreement indicates the model's attention doesn't reflect its decision-making process.

### Information flow assessment
- SigLIP → Connector → VLM → Expert: Is information preserved at each stage?
- Are there bottleneck layers or components where spatial/semantic information degrades?
- Does the action expert effectively utilize the VLM's representation?

### Temporal coherence
- Does the model's attention/attribution follow a sensible trajectory through the manipulation episode?
- Are transitions between task phases (approach → grasp → transport → place) reflected in the attention patterns?
- Are there unstable or erratic frames that suggest policy uncertainty?

### Modality integration
- How well does the model integrate vision, language, and proprioception?
- Does language grounding appear effective (if language diff data is available)?
- Is the vision/state balance appropriate for this task?

### Head and layer health
- Are there dead or redundant heads in any component?
- Do the per-head patterns suggest good capacity utilization?
- Are there concerning spectral properties in any layers?

## Key Findings

Summarize the 3-5 most important findings from this inspection run. For each:
- What was observed (reference specific metrics and visualizations).
- Why it matters for model quality and deployment readiness.
- Severity: critical / important / informational.

## Recommendations

### Training improvements
- What specific training changes would most improve this model? Rank by expected impact.
- Should attention supervision, data augmentation, or curriculum changes be considered?
- Are there signs of overfitting, underfitting, or training instability?

### Architecture considerations
- Is the current architecture well-suited for this task, or are there structural limitations?
- Specific component-level recommendations (connector capacity, head count, layer count).

### Evaluation recommendations
- What specific robustness tests should be run before deployment?
- What edge cases or failure modes do the visualizations suggest?
- What additional inspection analyses would provide the most insight?

### Priority action plan
Provide a numbered list of the top 5 actions the researcher should take next, ordered by priority.

Be specific and reference numerical statistics throughout the analysis.""",

    # -- Multi-run comparison --
    "multi_run_comparison": """You are comparing multiple SmolVLA inspection runs to track model development over time.

**Runs being compared:**
{run_descriptions}

**Configuration differences:**
{config_diff}

**Cross-run statistics (deltas):**
{computed_stats}

**Researcher notes:**
{run_notes}

**Response constraint: ≤300 words total. Focus on concrete deltas and the one most actionable recommendation.**

Please provide a three-part analysis:

## Comparison
- What changed between runs? Reference specific numerical deltas for each visualization type.
- Which metrics improved and which degraded?

## Trajectory Analysis
- Is the model improving or degrading overall? What's the trajectory?
- Which configuration changes (if any) likely caused which behavioral changes?
- Are there trade-offs (e.g., more focused attention but less temporal stability)?

## Next steps
- Based on the trajectory, what should the researcher try next?
- Which changes had the most positive/negative impact?
- Specific hyperparameter or architecture recommendations.

Be specific and reference the numbers throughout. If researcher notes provide context about what changed, use that to explain the observed differences.""",
}


def get_template(analysis_type: str) -> str:
    return TEMPLATES.get(analysis_type, "")


def format_template(template: str, **kwargs) -> str:
    """Safe string formatting that ignores missing keys."""
    for key, val in kwargs.items():
        template = template.replace(f"{{{key}}}", str(val))
    return template
