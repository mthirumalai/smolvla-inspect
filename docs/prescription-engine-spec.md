# Prescription Engine Spec

**Status:** Draft — freezing for phase 1a
**Repo:** `smolvla-prescribe` (separate from `smolvla-inspect`)
**Goal:** Transform diagnostic findings into precise, empirically-validated data prescriptions that tell the user exactly what data to collect, how much, and with what characteristics — then help them generate it. Over time, learn which prescriptions work across users and models.

**Spec freeze scope:** This spec freeze covers **phase 1a only**: data augmentation, GP-backed adaptive pilots, behavioral gating (with slice-based eval), and physical collection protocols. Phases 1b (training config), 1c (architecture changes), and 2 (retrieval) are documented for context and direction but are not part of the acceptance bar for v1a.

**v1a success contract:** An intervention counts as successful only if (1) the diagnostic target metric improves, (2) no behavioral regression is detected (global or slice-based), and (3) the change survives automated rollback checks.

---

## Repository Structure

The prescription engine lives in its own repository (`smolvla-prescribe`), separate from the diagnostic/visualization tool (`smolvla-inspect`). The dependency is one-directional: prescribe depends on inspect, never the reverse.

```
smolvla-prescribe/                     # THIS SPEC — new repo
├── prescribe/
│   ├── __init__.py
│   ├── sweep.py                       # run_sweeps(), parameterized counterfactual loops
│   ├── pilot.py                       # run_adaptive_pilot(), GP fitting, acquisition function
│   ├── loop.py                        # checkpoint loop orchestrator
│   ├── experiment_agent.py            # LLM research agent, prompt construction, proposal parsing
│   ├── coding_agent.py                # headless coding agent interface (Claude Code, Codex, local)
│   ├── verification.py                # code verification pipeline (compile, instantiate, forward, step)
│   ├── retrieval.py                   # trajectory retrieval index, similarity search, context formatting
│   ├── logging.py                     # IterationLog, TrajectoryLog, serialization
│   ├── collection_protocol.py         # physical collection protocol generator
│   └── models.py                      # SensitivitySweep, LearningCurveModel, DataPrescription,
│                                      #   ExperimentProposal, PrescriptionReport, etc.
├── configs/
│   ├── prescription_targets.yaml      # configurable thresholds and targets
│   └── prescription_program.md        # default research agenda template
├── run_prescribe.py                   # CLI entry point
├── requirements.txt                   # includes smolvla-inspect as dependency
└── README.md

smolvla-inspect/                       # EXISTING repo (unchanged)
├── smolvla_inspect/
│   ├── diagnostic/                    # consumed by smolvla-prescribe as a library
│   │   ├── models.py                  # DiagnosticReport, Symptom, Finding, CounterfactualResult, etc.
│   │   ├── counterfactual.py          # perturbation primitives (reused by sweeps)
│   │   ├── matrix.py                  # DiagnosticMatrix, symptom detectors
│   │   ├── comparison.py             # cross-run comparison (reused by improvement loop)
│   │   ├── scene.py                   # DatasetDiversityReport (consumed by prescriptions)
│   │   └── ...
│   └── ...
└── ...
```

### Why Two Repos

| Concern | Single repo (monorepo) | Two repos |
|---------|----------------------|-----------|
| **Users** | Everyone installs training infra + GP + coding agent SDKs even if they just want a heatmap | Inspect users get a lightweight install; prescribe users opt into heavier dependencies |
| **Dependencies** | smolvla-inspect already has torch, transformers, SAM, OWL-ViT, FastAPI, React. Adding LeRobot trainer, scikit-learn/GPyTorch, coding agent SDKs makes install painful | Each repo has a focused dependency set |
| **Release cycles** | Broken pilot loop blocks someone from running `./run.sh` | Independent releases. Inspect is stable; prescribe iterates fast |
| **Dependency direction** | Bidirectional coupling risk | One-directional: prescribe imports inspect, never the reverse |
| **Contributor scope** | Contributors to visualization don't need to understand training loops | Clear boundaries |

### Dependency Interface

`smolvla-prescribe` imports from `smolvla-inspect` as a library:

```python
# What smolvla-prescribe imports from smolvla-inspect
from smolvla_inspect.diagnostic.models import (
    DiagnosticReport,
    DiagnosticMatrix,
    Symptom,
    Finding,
    CounterfactualResult,
    DatasetDiversityReport,
    SceneSegmentation,
)
from smolvla_inspect.diagnostic.counterfactual import (
    background_substitution,
    object_relocation,
    lighting_perturbation,
    object_recolor,
    distractor_insertion,
    task_string_swap,
    occlusion_targeted,
)
from smolvla_inspect.diagnostic.comparison import compare_diagnostic_runs
from smolvla_inspect.diagnostic import run_diagnostic
```

The interface is the existing public API of `smolvla_inspect.diagnostic`. No changes to smolvla-inspect are required — the prescription engine is a downstream consumer.

For the MCP server: smolvla-inspect's MCP server remains read-only (48 tools for querying diagnostic data). `smolvla-prescribe` can optionally register its own MCP tools (e.g., `run_sweep`, `get_prescription_report`, `list_trajectories`) as a separate MCP server, or as an extension to the existing one via a plugin mechanism.

### Installation

```bash
# Users who just want diagnostics/visualization (existing workflow, unchanged)
pip install smolvla-inspect

# Users who want the prescription engine
pip install smolvla-prescribe  # automatically installs smolvla-inspect as a dependency
```

Or from source:

```bash
# Clone both repos side by side
git clone https://github.com/your-org/smolvla-inspect.git
git clone https://github.com/your-org/smolvla-prescribe.git

# Install inspect first (or use -e for development)
cd smolvla-inspect && pip install -e . && cd ..
cd smolvla-prescribe && pip install -e . && cd ..
```

---

## Problem

The diagnostic agent identifies *what's wrong* (e.g., "high background attribution") and offers generic fix advice (e.g., "add background augmentation"). But users want:

- Specific episode counts: "Add 25 episodes with patterned backgrounds"
- Specific characteristics: "15 episodes with 30-50% dimmer lighting"
- Tools to generate the prescribed data directly
- Confidence that the prescription will actually help before investing in full-scale collection

The current system has all the raw signals to produce this — dataset diversity metrics, counterfactual sensitivity, attribution mass — but no layer that translates signals into quantified, validated prescriptions.

A purely heuristic approach (map metric deficit → episode count via thresholds) is rigid, brittle, and doesn't account for interactions between axes. A purely factorial experimental design explodes combinatorially and can't adapt to surprising interactions. Instead, this system uses **empirical measurement** (sensitivity sweeps + pilot fine-tunes) for each model, and **learns across users** over time via structured outcome logging.

---

## Design Overview

The prescription engine adds three stages after the existing diagnostic pipeline, orchestrated by an **LLM research agent** that proposes experiments, evaluates outcomes, and iterates within a human-defined research agenda:

```
Existing pipeline:
  Symptoms → Hypotheses → Counterfactuals → Findings

New stages:
  1. Sensitivity sweeps    — measure model response curves per axis (factorial, cheap)
  2. Adaptive pilots       — GP-backed sequential experiments to estimate episode counts
  3. Prescription report   — precise, validated recommendations with calibrated uncertainty

Orchestration:
  4. Research agenda        — prescription_program.md written by the human
  5. LLM research agent     — reads agenda + diagnostics, proposes + evaluates experiments
  6. Structured logging     — record (state, action, outcome) for every iteration
  7. Automated guardrails   — regression rollback, budget enforcement, replay buffer
```

The core pattern: **human writes a research agenda in markdown → LLM agent proposes experiments within fixed budgets → data augmentation auto-runs, code changes require approval → keep or discard based on measured metrics → iterate.** The human refines the agenda across sessions; the agent handles execution.

### Three Intervention Types

Data augmentation is one lever. The diagnostic already identifies issues that data alone can't fix. The system supports three types of interventions, each with different risk profiles:

| Intervention | What changes | Risk level | Executed by | Verification |
|-------------|-------------|------------|-------------|-------------|
| **Data augmentation** | Training dataset (new/modified episodes) | Low — model architecture unchanged, action labels preserved for augmentation-safe axes | Counterfactual engine (batch mode) | Re-diagnostic: sweep sensitivity, attribution metrics |
| **Training config** | Hyperparameters: LR, dropout, batch size, optimizer, loss weights, augmentation pipeline | Medium — same architecture, but bad configs cause divergence or forgetting | Headless coding agent modifies training config/script | Pilot fine-tune: compare loss curves, re-diagnostic |
| **Architecture** | Model structure: connector size, attention heads, layer count, auxiliary losses, new modules | High — changes model capacity, can break pretrained weights | Headless coding agent modifies model code | Compilation check → shape check → pilot fine-tune → re-diagnostic |

The LLM research agent decides which intervention type to propose based on the diagnostic findings. Data issues get data fixes. Architectural issues get code fixes. Each iteration targets one intervention type (see "One Intervention Per Iteration" in the Code-Level Interventions section).

### Headless Coding Agent

For training config and architecture interventions, the system spawns a **headless coding agent** (Claude Code in headless mode, Codex, or similar) that:

1. Reads the current model code and training config
2. Receives a precise modification spec from the research agent
3. Makes the code changes
4. Verifies its own work (compilation, shape checks, smoke test)
5. Returns the modified code for the research loop to evaluate

The coding agent operates on a **git worktree** — an isolated copy of the training codebase. Changes are committed to a branch, never to main. If the experiment fails, the branch is discarded. If it succeeds, the branch is available for the user to review and merge.

The LLM modifies training code (or its equivalent), trains for a fixed budget, checks if the result improved, keeps or discards.

### Implementation Phases

Phase 1 is split into three sub-phases so the hardest part (architecture changes) doesn't define the schedule. Each sub-phase is independently shippable and testable.

| Phase | Intervention type | Approval model | Ship when |
|-------|------------------|----------------|-----------|
| **1a** | Data augmentation + physical collection protocols | Auto-run (safe axes) or human-review (task-dependent axes) | First — validates the core sweep/pilot/GP loop |
| **1b** | Training config changes (LR, dropout, loss weights, augmentation pipeline) | Human approval required | After 1a is stable — adds the coding agent for config files |
| **1c** | Architecture changes (connector size, aux heads, attention modifications) | Human approval required | After 1b is stable — adds weight loading, shape verification |
| **2** | Retrieval over logged trajectories | Same as underlying intervention | After 1a-1c are validated and ~50-100 trajectories collected |
| **Fallback** | Data augmentation only (greedy, no LLM) | Auto-run (safe axes only) | Available from day one for no-LLM environments |

**1a is the critical path.** It validates the core loop end-to-end: sweeps → GP-backed adaptive pilots → behavioral eval → prescription report. Everything else builds on top.

**1b adds the coding agent** but only for config files — lower risk than architecture changes because the model structure is unchanged. This validates the worktree isolation, verification pipeline, and human approval flow.

**1c is the riskiest** — architecture changes can break pretrained weights and require the full weight loading strategy. It ships last.

**Phase 2 (retrieval)** is enabled explicitly, not automatically. It adds value only if the underlying loop already works:

```bash
# Phase 1a (default): data augmentation loop
smolvla-prescribe run --model X --dataset Y

# Phase 1b: also allow training config changes
smolvla-prescribe run --model X --dataset Y \
    --training-codebase /path/to/training --allow-config-changes

# Phase 1c: also allow architecture changes
smolvla-prescribe run --model X --dataset Y \
    --training-codebase /path/to/training --allow-config-changes --allow-architecture-changes

# Phase 2 (opt-in): add retrieval over past trajectories
smolvla-prescribe run --model X --dataset Y --enable-retrieval
```

The fallback exists for air-gapped environments or CI pipelines where LLM API access is impractical. It is analogous to the diagnostic agent's rule-based hypothesis mode when no API key is set.

### The `prescription_program.md` Pattern

Instead of encoding the research strategy in code or CLI flags, the human writes a markdown document that the LLM agent interprets. This is more expressive than parameters — the user can encode domain knowledge, priorities, trade-offs, and constraints in natural language:

```markdown
# prescription_program.md

## Objective
Reduce background dependence to < 30% attribution.
Secondary: improve lighting robustness to ±40% brightness.
Do not attempt to fix object positioning — we are collecting physical data next week.

## Constraints
- Budget: 2 hours GPU time, max 5 iterations
- Max 15 augmented episodes per experiment
- Replay ratio: ≥ 0.7 (non-negotiable)

## Strategy preferences
- Try the cheapest experiment first — if 5 episodes of targeted textures work,
  don't generate 30
- I care more about avoiding regressions than maximizing improvement

## Domain knowledge
- This robot operates in a kitchen environment — wood countertops and tile floors
  are the deployment backgrounds. Prioritize those over abstract textures.
- The stainless steel cup is highly reflective — lighting augmentation should
  include specular highlight variation, not just brightness shifts.

## Augmentation safety overrides
- Color: SAFE — task is "pick up the block" with no color qualifier
- Clutter: SAFE — workspace is clear, distractors won't obstruct
- Language: UNSAFE — single-task model, do not swap instructions

## Stop conditions
- Stop if background_attribution < 30% (primary goal met)
- Stop if improvement plateaus for 2 consecutive iterations
- Stop if any previously-healthy metric crosses its warning threshold

## Future phases (not active in v1a)
# Uncomment when running with --allow-config-changes (phase 1b):
# - For the dead state pathway, propose vision_dropout=0.1 or state prediction aux loss
# - Training codebase: /path/to/lerobot-training/
# - Training entry point: train.py
# - Model config: configs/smolvla_base.yaml
```

The user refines this document across sessions. The LLM agent reads it alongside the diagnostic data to design experiments that are both data-driven (informed by sweeps) and context-aware (informed by human domain knowledge).

When no `prescription_program.md` exists, the system generates a default one from the diagnostic report and asks the user to review it before starting the loop. When no LLM is available, the system falls back to V1 (greedy rule-based) and ignores the program.md.

---

## Stage 1: Sensitivity Sweeps

### Purpose

A single counterfactual test (e.g., one background swap) gives a binary signal: "background matters" or "it doesn't." A sweep gives a **response curve**: how the model degrades as conditions vary along an axis.

### Sweep Axes

Each axis corresponds to a counterfactual primitive, extended to run at multiple parameter values:

| Axis | Primitive | Sweep parameter | Variation points |
|------|-----------|----------------|-----------------|
| Background | `background_substitution` | `replacement` type | gray, noise, blur, 4-5 texture fills (wood, tile, fabric, concrete, outdoor) |
| Lighting | `lighting_perturbation` | `brightness_delta` | -0.5, -0.4, -0.3, -0.2, -0.1, +0.1, +0.2, +0.3, +0.4, +0.5 |
| Position | `object_relocation` | `shift_pixels` | 8-10 positions on a grid spanning the workspace |
| Color | `object_recolor` | `hue_shift` | 0.0, 0.1, 0.2, ..., 0.9 (10 steps around the hue wheel) |
| Clutter | `distractor_insertion` | `position` + `distractor_size` | 6-8 positions × 2 sizes |
| Language | `task_string_swap` | `replacement_task` | 5-8 paraphrases + 2-3 unrelated tasks |

### Sweep Output

```python
@dataclass
class SweepPoint:
    """A single measurement along a sweep axis."""
    params: dict                    # e.g., {"brightness_delta": -0.3}
    action_delta_l2: float          # L2 action change from baseline
    action_delta_per_dim: list[float]
    gradcam_shift: float
    attribution_shift_per_region: dict[str, float]

@dataclass
class SensitivitySweep:
    """Full response curve for one axis."""
    axis: str                       # "background" | "lighting" | "position" | ...
    points: list[SweepPoint]        # ordered measurements
    baseline_action: list[float]    # original model output for reference

    # Derived metrics (computed after sweep)
    max_sensitivity: float          # largest action_delta_l2 across points
    mean_sensitivity: float         # average action_delta_l2
    breakpoint: str | None          # human-readable description of where degradation starts
    robust_range: str | None        # human-readable description of safe operating range
    is_sensitive: bool              # True if max_sensitivity > axis-specific threshold
```

### Sensitivity Thresholds

An axis is flagged as sensitive if the sweep reveals meaningful action changes. Thresholds are calibrated relative to the baseline action magnitude:

| Axis | Sensitive if max delta > | Rationale |
|------|-------------------------|-----------|
| Background | 0.15 × baseline_action_norm | 15% action change from background alone |
| Lighting | 0.10 × baseline_action_norm | 10% from lighting (expected to be more robust) |
| Position | 0.20 × baseline_action_norm | 20% expected since actions legitimately change |
| Color | 0.15 × baseline_action_norm | 15% from recoloring |
| Clutter | 0.10 × baseline_action_norm | 10% from distractors |
| Language | 0.05 × baseline_action_norm | 5% — language should have minimal effect for same-task paraphrases |

These thresholds live in `configs/prescription_targets.yaml` and are user-configurable.

### Augmentation-Safety Classification

The sweep classifies each axis by whether augmented data can safely reuse the original action labels. This is **task-dependent, not universal** — the table below gives defaults, but the LLM agent (or the user via `prescription_program.md`) can override them for specific tasks.

| Axis | Default | Caveats |
|------|---------|---------|
| Background | Safe | Generally safe. Exception: tasks where the background is part of the workspace (e.g., "wipe the table" — the table texture matters). |
| Lighting | Safe | Safe for moderate shifts. Extreme changes can affect depth perception and shadow-based grasping cues. |
| Color | **Task-dependent** | Safe if the task doesn't depend on color identity ("pick up the block"). Unsafe if color is the discriminator ("pick up the *red* block" — recoloring changes which object is correct). |
| Clutter | **Task-dependent** | Safe if distractors don't physically obstruct the action. Unsafe for tasks requiring navigation around obstacles, or if the distractor is confusable with the target. |
| Language | **Task-dependent** | Paraphrases of the same instruction are safe. Swapping to a different task instruction changes the correct action entirely — only valid if training for multi-task generalization. |
| Position | Unsafe | Moving the object changes the correct reach trajectory. Synthetic relocation with original action labels is invalid. |

The key principle: **if in doubt, mark as task-dependent and let the LLM agent or the user decide.** The system should not silently generate augmented data with invalid action labels. When an axis is marked task-dependent, the LLM agent must justify in its proposal rationale why augmentation is safe for this specific task, and the human checkpoint provides a second layer of review.

#### Causal DAGs for Augmentation Safety

Each augmentation-safety classification encodes a causal assumption. Making these assumptions explicit as directed acyclic graphs (DAGs) clarifies what we're assuming and where the assumptions break. The DAGs below show the data-generating process for each axis type.

**Safe axis (background):**

```
Background ──→ Image ──→ Model ──→ Predicted Action
                ↑                        ↑
Object Position ─┘                       │
                                         │
Object Position ──→ Correct Action ──────┘ (training label)
```

Augmentation-safe because: background affects the image but has no causal path to the correct action. Swapping the background changes the image the model sees but doesn't change what the robot should do. The assumption: **no backdoor path from background to correct action.** This holds for most manipulation tasks but fails when the background is part of the workspace (e.g., "wipe the table" — the table surface is both background and task-relevant).

**Task-dependent axis (color):**

```
Object Color ──→ Image ──→ Model ──→ Predicted Action
                  ↑                        ↑
Object Position ──┘                        │
                                           │
Object Color ──→ Task Instruction ──→ Correct Action
                  (maybe)
```

The dotted path `Object Color → Task Instruction → Correct Action` exists only when the task instruction references color ("pick up the *red* block"). If it does, recoloring changes which object the instruction refers to, invalidating the action label. If it doesn't ("pick up the block"), the path is absent and augmentation is safe. **This is why the LLM must justify safety for this specific task** — it's checking whether the dotted path exists.

**Unsafe axis (position):**

```
Object Position ──→ Image ──→ Model ──→ Predicted Action
       │                                      ↑
       │                                      │
       └──────────→ Correct Action ───────────┘ (training label)
```

Unsafe because: object position has a direct causal path to the correct action (the reach trajectory changes). Augmentation that changes position while keeping old action labels creates a contradictory training signal.

These DAGs should be included in the `prescription_program.md` template so users can inspect the assumptions for their specific task. When the LLM proposes augmentation on a task-dependent axis, its safety justification should reference which causal paths exist or are absent for this task.

The `prescription_program.md` can override defaults:

```markdown
## Augmentation safety overrides
- Color: UNSAFE for this task — the instruction is "pick up the red block"
  and recoloring would invalidate the task (Object Color → Task Instruction path exists)
- Clutter: SAFE — the workspace is clear and distractors won't obstruct the grasp
  (no causal path from distractor to correct action)
- Language: UNSAFE — single-task model, do not swap instructions
```

### Compute Cost

Each sweep point is one forward pass through the model (+ optional GradCAM). For a typical sweep:

- ~8-10 points per axis
- 6 axes
- ~50-60 forward passes total
- Estimated time: **5-10 minutes on GPU**, **20-40 minutes on CPU**

---

## Stage 2: Adaptive Pilots

### Purpose

Sweeps tell you *what* is sensitive. Pilots determine *how many episodes* fix it — and **execute the fix automatically for data augmentation**. The user does not receive a recommendation to "train with 20 more episodes with different lighting." Instead, the system generates the augmented data, fine-tunes, evaluates (diagnostic metrics + behavioral eval), and feeds the result back to the GP. The user sees the outcome, not the intermediate steps.

The only prescriptions surfaced as manual action items are for interventions that **cannot be automated** — primarily physical data collection where the correct action labels change (e.g., object in new positions).

The pilot protocol uses **adaptive sequential experimentation** with a Gaussian Process model to determine N with calibrated uncertainty.

### Why Not a Parametric Fit

An earlier version of this spec proposed fitting `metric(n) = amplitude × e^(-λn) + floor` to three pilot points (N=5, 15, 30). This fails for several reasons:

- **Zero degrees of freedom.** Three parameters, three points. The fit is exact by construction. R² is meaningless and there is no way to estimate uncertainty.
- **Phase transitions.** The model might show zero improvement at small N then suddenly improve once augmentation diversity crosses a threshold. An exponential can't represent this.
- **Non-monotonic behavior.** Small amounts of augmented data can temporarily worsen performance before the model learns to generalize. An exponential assumes monotonic decay.
- **Diversity ≠ count.** 30 identical augmented episodes ≠ 30 diverse ones. Improvement depends on what's *in* the episodes, not just how many.
- **No uncertainty.** The user gets "N=30 will give you 35%" with no indication that it could be 25% or 45%.

### Estimand: What the GP Is Estimating

Before specifying the GP machinery, define the estimand precisely using potential outcomes notation:

```
τ(N, X, S) = E[Y(N) - Y(0) | X, S]
```

Where:
- **Y(N)** = diagnostic metric Y after fine-tuning with N augmented episodes of type X
- **Y(0)** = diagnostic metric Y with no augmented episodes (the baseline)
- **X** = augmentation characteristics (e.g., "textured backgrounds: wood, tile, fabric")
- **S** = target slice for behavioral eval (e.g., "novel_background" slice)
- **τ** = the average treatment effect of augmentation dose N on metric Y for slice S

The GP posterior is a model of `E[Y(N) | X]` as a function of N. The acquisition function seeks the N where the posterior crosses the target value of Y. The prescription is the N (with confidence interval) that achieves the desired τ.

Being explicit about the estimand clarifies several design choices:

- **Why pilots start from the same checkpoint**: each pilot measures `Y(N)` independently, not the cumulative effect of sequential fine-tunes. This gives the GP clean treatment-effect estimates at each dose.
- **Why the eval pool must be separate from the training pool**: we want `E[Y(N)]` over the *data distribution*, not over the specific augmented instances. Evaluating on training instances measures overfitting, not the treatment effect.
- **Why behavioral eval matters**: the diagnostic metric Y is a proxy for the real outcome of interest (task success). The behavioral eval checks that τ on the proxy correlates with τ on the true outcome.
- **What "augmentation ceiling" means**: the floor of the GP is the limit of `τ(N)` as N → ∞. When this limit is above the target, no amount of this augmentation type will reach the goal — the treatment has bounded effect.

### Adaptive Pilot Protocol

Instead of fixed pilot sizes, the protocol runs sequentially and decides after each measurement whether to continue:

```
1. Anchor at N=0 (current metric, known exactly)
2. Run first pilot at N=5 (small, cheap)
3. Update GP posterior
4. Ask acquisition function: "is another measurement worth it?"
   → If yes: run next pilot at the N suggested by the acquisition function
   → If no: emit prescription with confidence interval
5. Repeat until stopping criterion is met or budget exhausted
```

The key difference: the system places measurements *where they're most informative* rather than at predetermined sizes. If N=5 already shows dramatic improvement, it might probe N=8 and N=12 to pin down the target crossing precisely. If N=5 shows zero improvement, it might jump to N=25 to test whether augmentation works at all before wasting time on intermediate values.

### Gaussian Process Model

A GP over (episode_count, metric_value) pairs provides the posterior:

```python
@dataclass
class PilotPoint:
    """Result of one pilot fine-tune at a specific scale."""
    num_augmented_episodes: int     # how many augmented episodes were used
    finetune_steps: int             # how many gradient steps
    replay_ratio: float             # fraction of original data in training mix
    metric_before: float            # target metric before fine-tune
    metric_after: float             # target metric after fine-tune
    improvement: float              # metric_before - metric_after
    regression_metrics: dict[str, float]  # all other diagnostic metrics after fine-tune
    sweep_after: dict[str, dict]    # re-sweep results after fine-tune

@dataclass
class LearningCurveModel:
    """GP-based model of metric improvement vs. episode count."""
    axis: str
    target_metric: float            # what "healthy" looks like
    pilot_points: list[PilotPoint]  # observed measurements

    # GP posterior (updated after each observation)
    # Callable signatures: int → float
    posterior_mean: Callable[[int], float]
    posterior_std: Callable[[int], float]

    # Derived from posterior
    predicted_floor: float          # posterior mean as N → large
    floor_uncertainty: float        # std of floor estimate
    predicted_n_for_target: int | None      # N where posterior mean crosses target
    confidence_interval_n: tuple[int, int] | None  # 80% CI for N needed
    probability_target_reachable: float     # P(floor < target) under the posterior

    # Decision signals
    expected_improvement_at_n: Callable[[int], float]   # posterior mean improvement at any N
    marginal_value_at_n: Callable[[int], float]         # d(improvement)/d(n) — when this is near zero, stop
```

#### Kernel Choice

The GP kernel encodes structural priors about learning curves:

- **Matérn 5/2** as the base kernel — smooth but allows local variation, unlike the overly smooth RBF
- **Negative linear mean function** — soft prior that metrics decrease (improve) with more data
- **Anchor observation at N=0** — pin the starting metric with near-zero noise, so the GP is calibrated at the origin
- **Observation noise σ²** — accounts for fine-tune stochasticity (different random seeds, batch ordering). Estimated from pilot variance or set conservatively at ~5% of the metric range

```python
def build_gp(pilot_points: list[PilotPoint], metric_before: float):
    """Construct a GP posterior from pilot observations."""
    # Observations: [(0, metric_before), (n1, m1), (n2, m2), ...]
    X = np.array([0] + [p.num_augmented_episodes for p in pilot_points]).reshape(-1, 1)
    y = np.array([metric_before] + [p.metric_after for p in pilot_points])

    kernel = Matern(nu=2.5, length_scale=15.0, length_scale_bounds=(5.0, 100.0))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.01,         # observation noise (small for N=0 anchor, larger for pilots)
        n_restarts_optimizer=5,
        normalize_y=True,
    )
    # Use heteroscedastic noise: near-zero for the anchor, estimated for pilots
    gp.fit(X, y)
    return gp
```

For a more principled approach, a **monotonic GP** (constrained to non-increasing predictions, reflecting the prior that more diverse data shouldn't make the metric worse in expectation) can be implemented via virtual derivative observations. This is optional — the standard GP with a negative-trend mean function handles most cases.

#### Acquisition Function

After each pilot measurement, decide what N to try next:

```python
def next_batch_size(
    model: LearningCurveModel,
    budget_remaining_episodes: int,
    min_batch: int = 5,
    batch_step: int = 5,
) -> int | None:
    """Decide the next episode count to try, or None to stop.

    Uses expected information gain near the target crossing point.
    """
    target = model.target_metric
    current_n = max(p.num_augmented_episodes for p in model.pilot_points) if model.pilot_points else 0

    # ── Stopping criteria ────────────────────────────────────

    # 1. Confident the target is already reached
    mean_at_current = model.posterior_mean(current_n)
    std_at_current = model.posterior_std(current_n)
    if mean_at_current < target and std_at_current < (target * 0.10):
        return None  # confident, prescribe current N

    # 2. Confident augmentation can't reach target (ceiling above target)
    if model.probability_target_reachable < 0.10:
        return None  # augmentation ceiling, stop

    # 3. Marginal improvement is negligible
    if current_n > 0:
        marginal = abs(model.marginal_value_at_n(current_n))
        if marginal < 0.005:  # <0.5% improvement per additional episode
            return None  # diminishing returns, not worth more pilots

    # 4. Budget exhausted
    if budget_remaining_episodes < min_batch:
        return None

    # ── Acquisition: maximize information near target crossing ─

    candidates = list(range(current_n + min_batch, budget_remaining_episodes + 1, batch_step))
    if not candidates:
        return None

    scores = []
    for n in candidates:
        std_n = model.posterior_std(n)
        distance_to_target = abs(model.posterior_mean(n) - target)
        # High score = high uncertainty AND close to the decision boundary
        # (this is essentially Expected Improvement near the target)
        score = std_n * np.exp(-0.5 * (distance_to_target / max(std_n, 1e-6)) ** 2)
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return candidates[best_idx]
```

The acquisition function concentrates measurements where **uncertainty is high** AND **the posterior is near the target**. This is the region where knowing more would most change the prescription. If the GP is already confident the target is met at N=12, it doesn't waste compute trying N=20.

### Adaptive Protocol Walkthrough

A concrete example with background attribution:

```
Target: background_attribution < 30%
Starting: background_attribution = 78%

Step 1: Pilot at N=5
  → metric_after = 62%
  → GP posterior: mean(30)=38%, std(30)=12%
  → Acquisition says: try N=18 (high uncertainty near target crossing)

Step 2: Pilot at N=18
  → metric_after = 36%
  → GP posterior: mean(25)=33%, std(25)=4%, mean(30)=31%, std(30)=3%
  → Acquisition says: try N=22 (refine the crossing point)

Step 3: Pilot at N=22
  → metric_after = 33%
  → GP posterior: mean(28)=31%, std(28)=2%
  → probability_target_reachable = 0.72
  → Stopping: std at current N < target * 10%, marginal < 0.5%/episode

Prescription:
  → 25-30 episodes (80% CI)
  → At 25 episodes: 32% ± 3% background attribution
  → At 30 episodes: 31% ± 2% background attribution
  → Augmentation ceiling: 29% ± 4% (P(reaches 30% target) = 72%)
```

Three pilots instead of the fixed three, but placed more informatively. The prescription includes **confidence intervals** instead of point estimates.

Compare with the fixed protocol:
- Fixed: tries N=5, 15, 30 — wastes a measurement at N=15 when the interesting region is 18-28
- Adaptive: tries N=5, 18, 22 — zooms into the region that matters

### Prescription Output with Uncertainty

The GP posterior transforms the prescription from a point estimate to a distribution:

```python
@dataclass
class PrescriptionEstimate:
    """GP-backed episode count estimate with calibrated uncertainty."""
    point_estimate: int                     # posterior mean crossing point
    confidence_interval_80: tuple[int, int] # 80% CI for N needed
    confidence_interval_95: tuple[int, int] # 95% CI for N needed

    # Metric predictions at key N values
    predictions: list[dict]                 # [{"n": 20, "mean": 0.35, "std": 0.04}, ...]

    # Ceiling analysis
    predicted_ceiling: float                # posterior mean at large N
    ceiling_uncertainty: float              # std of ceiling estimate
    probability_target_reachable: float     # P(ceiling < target)

    # Decision summary
    recommendation: str                     # "augment" | "augment_with_residual" | "ceiling_reached"
    residual_gap: float | None              # ceiling - target, if positive
```

### Fine-Tune Configuration

Each pilot fine-tune is intentionally short — enough to measure the direction of improvement, not enough to fully converge:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Steps per pilot | 50-100 | Enough to see directional improvement |
| Learning rate | 1e-5 | Conservative to avoid catastrophic forgetting |
| Replay ratio | 0.7 | 70% original data, 30% augmented — protects existing capabilities |
| Batch size | Same as original training | Consistency |
| Optimizer | AdamW (same as original) | Consistency |
| Checkpoint | Saved after each pilot | Enables rollback |

Each pilot starts from the original model checkpoint (not incrementally from the previous pilot). This ensures each pilot point is an independent measurement, which is important for GP posterior calibration — correlated measurements would understate uncertainty.

### Compute Cost

The adaptive protocol typically uses 2-4 pilots per axis (vs. the fixed 3), but places them more efficiently:

Per axis:
- Augmented data generation: ~2-8 minutes (adaptive — generates only what's needed, not a fixed 30)
- 2-4 pilot fine-tunes (50-100 steps each): ~2-5 minutes each on GPU
- 2-4 re-sweeps: ~3-5 minutes each
- GP fitting: negligible (~milliseconds on 2-4 data points)

Total per axis: **~15-30 minutes on GPU**. For 2-3 flagged axes: **~45 min-1.5 hours**.

In the best case (first pilot already shows clear result), the adaptive protocol finishes faster than fixed pilots. In the worst case (noisy measurements, need more probes), it takes slightly longer but provides calibrated uncertainty that the fixed protocol can't.

### Fallback for Degenerate Cases

| Case | GP behavior | Action |
|------|-------------|--------|
| First pilot shows zero improvement | Low posterior uncertainty, flat prediction | Stop early: "augmentation doesn't help on this axis" |
| First pilot shows huge improvement | Posterior already confident target is met at small N | Stop early: prescribe small N with tight CI |
| Pilots show non-monotonic behavior (dip then recovery) | GP captures the shape, widens uncertainty | Continue probing — may need more data to resolve |
| Pilots are very noisy (high variance) | GP widens uncertainty, acquisition function asks for more points | Continue probing if budget allows, else emit wide CI |
| Only 1-2 pilots possible (budget constrained) | GP gives wide uncertainty bands | Emit prescription with explicit "low confidence" flag |

---

## Behavioral Evaluation

### The Proxy Optimization Problem

The diagnostic metrics (attribution mass, sweep sensitivity, positional baseline ratio, etc.) are **proxies** for what actually matters: does the robot perform the task better? A model can have clean diagnostics — low background attribution, robust to lighting changes — and still fail at the task. Conversely, a model with messy diagnostics might work fine in practice.

If the improvement loop only measures diagnostic metrics, it risks optimizing for cleaner diagnostics rather than better robot performance. To guard against this, the core loop includes a **behavioral evaluation gate** alongside diagnostic metrics.

### Behavioral Eval in the Core Loop

After each iteration's fine-tune, the system runs a behavioral evaluation before declaring success:

```python
@dataclass
class BehavioralEval:
    """Task-level performance measurement, not just diagnostic metrics."""

    # Episode rollouts (if simulator is available)
    success_rate: float | None          # fraction of episodes where task completes
    mean_return: float | None           # mean reward across rollout episodes
    num_rollout_episodes: int           # how many episodes were evaluated

    # Action quality — global (always available, no simulator needed)
    action_mse_global: float            # MSE on all held-out episodes
    action_smoothness: float            # mean L2 between consecutive predicted actions
    action_diversity: float             # std of predicted actions across episodes

    # Action quality — sliced by diagnostic failure mode
    action_mse_slices: dict[str, float] # keyed by slice name, e.g.:
                                        #   "target_object_visible": 0.012
                                        #   "target_object_occluded": 0.089
                                        #   "novel_background": 0.034
                                        #   "near_grasp_phase": 0.008
                                        #   "transport_phase": 0.021
    slice_regressions: list[str]        # slices where MSE worsened by >10%

    # Comparison
    improved_vs_baseline: bool
    behavioral_regression: bool         # True if global OR any slice regressed significantly
```

The evaluation has three tiers:

**Tier 1: Global action quality (always runs, no simulator needed).** Evaluate the fine-tuned model on held-out episodes from the dataset. Compare predicted actions to ground-truth actions. This catches gross regressions where augmentation caused forgetting on the core task.

**Tier 2: Slice-based action quality (always runs, aligned to diagnostic findings).** The same held-out evaluation, but **sliced by the failure mode the current iteration is targeting**. Slices are derived from the diagnostic report:

| Diagnostic finding | Behavioral eval slices |
|-------------------|----------------------|
| `high_background_attribution` | "default_background" vs. "novel_background" (augmented episodes) |
| `low_object_attribution` | "target_visible" vs. "target_partially_occluded" |
| `spatial_shortcut` | "training_position" vs. "novel_position" (from sweep variations) |
| `gripper_fixation` | "gripper_visible" vs. "gripper_masked" |
| `dead_state_pathway` | "state_matches_vision" vs. "state_contradicts_vision" |
| `language_insensitivity` | "original_instruction" vs. "paraphrased_instruction" |

Slicing matters because global MSE can hide localized regressions. A model that improves on novel backgrounds but degrades on the original background has a behavioral regression — even if global MSE is flat. The slice-based eval catches this.

**Anti-leakage rule:** No iteration may be declared successful using evaluation slices constructed from the same exact perturbation instances used in training. Eval slices must use **held-out perturbations** — different random seeds, different texture images, different placement offsets — from the same perturbation *family* but never the same instances. This prevents the loop from overfitting to its own augmentation generator.

In practice, the system maintains two perturbation pools per axis:
- **Training pool**: used to generate augmented episodes for fine-tuning
- **Eval pool**: reserved for behavioral eval slices, never seen during training

Both pools are drawn from the same sweep axis (e.g., "textured backgrounds") but with different seeds or source images. The eval pool is generated once at the start of the loop and held constant across iterations.

Episodes from the original dataset become the "baseline slice"; episodes generated from the eval pool become the "intervention slice."

**Tier 3: Rollout evaluation (opt-in, requires simulator).** Execute full task rollouts and measure success rate, mean return, or task completion. This is the gold-standard measurement but requires a simulation environment, which not all users have.

### Randomization Inference for Slice Improvements

The GP provides uncertainty estimates based on its kernel assumptions. But if the kernel is misspecified, those uncertainty estimates are unreliable. Since we control the perturbation generation, we can use a stronger, assumption-free approach: **Fisher-style randomization inference**.

The idea: instead of asking "is this improvement statistically significant under the GP's model?", ask "would we see this much improvement if we randomly shuffled which episodes were augmented vs. original?"

#### Placebo test protocol

After each pilot measurement shows improvement on a slice, run a placebo test:

```python
def placebo_test(
    model_checkpoint,
    original_episodes: list,
    augmented_episodes: list,
    eval_pool: list,
    metric_fn: Callable,
    n_permutations: int = 100,
) -> float:
    """Fisher-style randomization test for slice improvement.

    Returns a p-value: probability of seeing this much improvement
    under random assignment of episodes to augmented vs. original.
    """
    # Observed effect: fine-tune with augmented data, measure on eval pool
    observed_effect = measure_effect(model_checkpoint, augmented_episodes, eval_pool, metric_fn)

    # Null distribution: randomly assign episodes to "augmented" vs "original"
    # and measure the effect each time
    null_effects = []
    all_episodes = original_episodes + augmented_episodes
    for _ in range(n_permutations):
        shuffled = random.sample(all_episodes, len(augmented_episodes))
        remaining = [e for e in all_episodes if e not in shuffled]
        effect = measure_effect(model_checkpoint, shuffled, eval_pool, metric_fn)
        null_effects.append(effect)

    # p-value: fraction of null effects >= observed effect
    p_value = sum(1 for e in null_effects if e >= observed_effect) / n_permutations
    return p_value
```

If the p-value is high (e.g., > 0.10), the improvement could have occurred by chance under random episode assignment — the augmented data may not be contributing meaningfully. The system flags this: "Improvement on this slice is not distinguishable from random episode selection (p=0.23). The augmented data may not be the cause."

#### When to run placebo tests

Placebo tests are more expensive than a single eval (they require multiple short fine-tunes). Use them selectively:

- **Always run** when declaring an axis resolved (before emitting "RESOLVED" in the report)
- **Always run** when the GP is near its stopping criterion (marginal improvement is small — is it real or noise?)
- **Skip** when the improvement is large and obvious (e.g., 78% → 35% background attribution — no need for a permutation test)

The threshold for skipping: if the observed effect is > 3× the standard deviation of the GP posterior at that point, the improvement is large enough that a placebo test is unnecessary.

#### Compute cost

Each permutation requires a short fine-tune (~50 steps) + eval. With 100 permutations, that's ~100 fine-tunes. Too expensive to run every iteration.

Practical compromise: use **20-50 permutations** (sufficient for p < 0.05 detection) and only at decision points (axis resolved, loop stopping). At 50 steps per permutation, 20 permutations ≈ 10-15 minutes on GPU. This is comparable to one additional pilot measurement.

### Integration with the Loop

Behavioral eval runs after the diagnostic re-evaluation, as an additional gate:

```
Fine-tune → Re-diagnostic (diagnostic metrics) → Behavioral eval (action quality)
                                                       ↓
                                               Did actions improve or hold?
                                                  Yes → keep iteration
                                                  No  → flag behavioral regression
```

A **behavioral regression** triggers the same rollback as a diagnostic regression — even if the diagnostic metrics improved. Regression is detected when:

- Global action MSE increased by >10%, OR
- Any behavioral eval slice worsened by >15% (localized regression on the baseline data), OR
- Rollout success rate dropped (if simulator is available)

The slice-based check is critical: it catches cases where augmentation improved the intervention slice but degraded the baseline slice (the model "forgot" how to handle the original data distribution). This prevents the system from optimizing proxy metrics at the expense of actual task performance.

### Configuration

```bash
# Enable rollout evaluation (requires simulator)
smolvla-prescribe run \
    --model your/model_id \
    --dataset your/dataset_id \
    --eval-rollouts 20 \
    --eval-env your_gym_env_id

# Action-quality evaluation only (default, no simulator needed)
smolvla-prescribe run \
    --model your/model_id \
    --dataset your/dataset_id \
    --eval-episodes 50           # held-out episodes for action MSE
```

Action-quality evaluation (tiers 1 and 2) is **on by default**. It uses held-out episodes from the dataset — no extra configuration. Rollout evaluation (tier 3) is opt-in via `--eval-rollouts` and requires a simulator environment.

### Partial Identification for Task-Dependent Axes

When augmentation safety is uncertain (task-dependent axes), point estimates of improvement are misleading — the true effect depends on an assumption (action labels are valid) that may not hold. Instead of reporting a single estimate, report **bounds** under different assumptions about label corruption.

```python
@dataclass
class PartialBounds:
    """Improvement bounds under varying augmentation safety assumptions."""
    best_case: float            # improvement if augmentation is fully safe (0% label corruption)
    worst_case: float           # improvement if augmentation partially corrupts labels
    corruption_pct: float       # assumed corruption rate for worst case (e.g., 20%)
    width: float                # best_case - worst_case
    narrow: bool                # True if width < 0.05 (bounds are tight enough to act on)
```

The best-case bound is the GP's standard estimate (assuming all augmented labels are correct). The worst-case bound assumes a fraction of augmented episodes have corrupted labels, which act as noise in the training data:

```python
def compute_partial_bounds(
    proposal: ExperimentProposal,
    gp_model: LearningCurveModel | None,
    corruption_rates: list[float] = [0.0, 0.1, 0.2],
) -> PartialBounds:
    """Compute improvement bounds under varying label-corruption assumptions.

    corruption_rate = 0.0: all augmented labels are correct (best case)
    corruption_rate = 0.2: 20% of augmented labels are wrong (worst case)

    Under corruption, the effective number of useful augmented episodes
    is reduced: N_effective = N * (1 - corruption_rate). The corrupted
    episodes act as noise, partially offsetting the improvement.
    """
    if gp_model is None:
        return PartialBounds(best_case=0.0, worst_case=0.0, corruption_pct=0.2, width=0.0, narrow=True)

    N = proposal.num_episodes
    improvements = []
    for rate in corruption_rates:
        n_effective = int(N * (1 - rate))
        n_noisy = N - n_effective
        # Effective improvement: improvement from n_effective good episodes,
        # minus degradation from n_noisy bad episodes (modeled as noise)
        improvement = gp_model.posterior_mean(n_effective)
        noise_penalty = n_noisy * estimated_noise_impact_per_episode
        improvements.append(improvement - noise_penalty)

    return PartialBounds(
        best_case=improvements[0],
        worst_case=improvements[-1],
        corruption_pct=corruption_rates[-1],
        width=improvements[0] - improvements[-1],
        narrow=(improvements[0] - improvements[-1]) < 0.05,
    )
```

When the bounds are **narrow** (best-case and worst-case are close), the augmentation is worth trying regardless of safety uncertainty — even in the worst case, the improvement is meaningful. When the bounds are **wide**, the user should be cautious — the improvement depends heavily on an unverified assumption.

This is presented to the user at the task-dependent approval checkpoint, alongside the LLM's safety justification and the causal DAG. The user sees: "If your task doesn't depend on color, we expect 12-18% improvement. If it does, improvement could be as low as 3%." This is much more informative than a binary safe/unsafe classification.

---

## Stage 3: Prescription Report

### Two Categories of Output

The prescription report distinguishes between work the system **already did** and work the user **needs to do manually**:

| Category | What happened | User action required |
|----------|-------------|---------------------|
| **Completed interventions** | Data augmentation on safe axes ran automatically; task-dependent axes were human-reviewed | None — review the improvement trajectory and merge if satisfied |
| **Remaining prescriptions** | System identified the issue and measured sensitivity, but the fix requires physical data collection or manual intervention | Follow the collection protocol or apply the code change |

For augmentation-safe axes, the system does not recommend "generate 20 episodes with varied lighting." It generates them, fine-tunes, measures, and reports: "Lighting is fixed. Background attribution dropped from 78% to 31% over 3 iterations, 25 augmented episodes total."

For physical-collection axes, the system emits a detailed collection protocol because it genuinely cannot automate the fix.

### Prescription Data Model

```python
@dataclass
class DataPrescription:
    """A single prescription for one improvement axis."""
    axis: str                               # "background" | "lighting" | "position" | ...
    priority: int                           # 1 = highest, ordered by max_sensitivity × severity
    category: str                           # "augmentation" | "physical_collection"

    # Current state
    current_metric: float                   # e.g., background_attribution = 0.78
    current_dataset_stat: float | None      # e.g., background_diversity_score = 0.03

    # Empirical evidence
    sweep: SensitivitySweep                 # full response curve
    gp_model: LearningCurveModel | None     # GP posterior from adaptive pilots (None if physical)
    estimate: PrescriptionEstimate | None   # GP-derived N with confidence intervals

    # The prescription
    recommended_episodes: int               # point estimate from GP posterior
    episode_range_80: tuple[int, int] | None  # 80% CI for N needed
    target_metric: float                    # what "healthy" looks like
    characteristics: list[str]              # specific data characteristics to include
    augmentation_ceiling: float | None      # GP posterior mean at large N
    ceiling_uncertainty: float | None       # std of ceiling estimate
    probability_target_reachable: float | None  # P(ceiling < target)
    residual_gap: float | None              # ceiling - target (positive = augmentation won't suffice)

    # Confidence
    confidence: str                         # "empirical" (pilot-backed) | "sweep-only" | "heuristic"
    num_pilots_run: int                     # how many adaptive pilot measurements were taken
    rationale: str                          # human-readable evidence chain

@dataclass
class CompletedIntervention:
    """An intervention that the system executed (auto for data, approved for code)."""
    axis: str
    intervention_type: str                  # "data_augmentation" | "training_config" | "architecture"
    iterations: list[ImprovementStep]       # full history of pilot/apply steps
    metric_before: float
    metric_after: float
    target_metric: float
    resolved: bool                          # did we reach the target?
    total_augmented_episodes: int           # 0 for pure code changes
    code_diff_branch: str | None            # git branch with code changes (if any)
    checkpoint_path: str                    # model checkpoint after this intervention

@dataclass
class CollectionProtocol:
    """Detailed physical data collection guide for non-augmentable axes."""
    axis: str
    reason_not_automatable: str             # why this can't be done synthetically
    current_distribution: dict              # e.g., {"mean_x": 312, "std_x": 12, "range": [288, 336]}
    target_distribution: dict               # e.g., {"std_x": ">80px", "num_positions": "5x4 grid"}
    workspace_bounds: dict                  # estimated from existing detections
    episode_count: int                      # estimated episodes needed
    setup_instructions: list[str]           # step-by-step for lab setup
    consistency_requirements: list[str]     # what to keep constant (lighting, background, etc.)
    validation_criteria: dict[str, float]   # re-diagnostic targets to confirm sufficiency

@dataclass
class PrescriptionReport:
    """Top-level output of the prescription engine."""

    # What the system already did (v1a: data augmentation only)
    completed: list[CompletedIntervention]   # automated interventions, ordered by execution
    total_augmented_episodes: int            # sum across all data augmentation interventions
    final_checkpoint: str                    # model checkpoint after all completed interventions

    # What the user needs to do manually
    remaining: list[CollectionProtocol]      # physical collection protocols
    remaining_suggestions: list[DataPrescription]  # issues v1a couldn't fix (code changes, unsafe axes)

    # Summary
    axes_resolved: list[str]
    axes_unresolved: list[str]
    total_wall_clock: str
    summary: str                             # human-readable overview
```

### Report Format

The prescription report is emitted as both structured JSON and human-readable markdown:

```markdown
## Prescription Report

### Completed Interventions (automated)

Data augmentation on safe axes ran automatically. Task-dependent axes were human-reviewed.
Review the improvement trajectory and merge the model checkpoint if satisfied.

#### Background Diversity — RESOLVED (3 iterations, 22 augmented episodes)

| Iteration | Action | Background Attribution |
|-----------|--------|----------------------|
| Before    | —      | 78%                  |
| 1         | 5 textured background episodes (wood, tile) | 62% |
| 2         | 13 more episodes (fabric, concrete, kitchen surfaces) | 36% |
| 3         | 4 more episodes (GP recommended targeting high-frequency textures) | 31% |

Result: background_attribution 78% → 31% ✓ (target: < 30%, within ceiling)
Augmentation ceiling: 29% ± 4%. Remaining 1% gap is within noise.
Model checkpoint: `checkpoints/prescribe_iter_003/`

#### Lighting Range — RESOLVED (2 iterations, 12 augmented episodes)

| Iteration | Action | Lighting Breakpoint |
|-----------|--------|-------------------|
| Before    | —      | -20% brightness   |
| 4         | 8 episodes (30-50% dimmer) | -35% |
| 5         | 4 episodes (directional lighting, specular variation) | -42% |

Result: robust to ±42% brightness shifts ✓ (target: ±40%)
GP stopped: marginal improvement < 0.5%/episode beyond N=12.

---

### Remaining Prescriptions (manual action required)

These require physical data collection or code-level changes that v1a does not automate.

#### Object Position Variety — UNRESOLVED

**Why this can't be automated:**
Moving the object changes the correct action trajectory.
Augmented relocation with original action labels would be invalid.

**Current state:**
- lego_block positions: mean=(312, 245), std=(12px, 9px)
- Sweep: model fails beyond 60px from cluster center

**Collection Protocol:**
1. Set up a 5×4 grid across the workspace (bounds: x=[100,540], y=[80,400])
2. Place the target object at each grid cell with ~30px natural variation
3. Record 1 episode per position (20 episodes total)
4. Keep lighting, background, and camera angle consistent with existing data
5. Prioritize positions far from (312, 245): corners and edges first

**Validation criteria (re-run diagnostic after collection):**
- object_position std_x > 80px, std_y > 60px
- positional_baseline_ratio < 0.30

#### Dead State Pathway — UNRESOLVED (requires code change, phase 1b)

**Why this can't be automated in v1a:**
The proprioceptive state input is architecturally bypassed (vision_share = 99.7%).
Data augmentation cannot fix this — it requires a training config change
(e.g., add vision dropout, state prediction auxiliary loss).

**Recommended intervention (phase 1b):**
- Add `vision_dropout: 0.1` to training config
- Add state prediction auxiliary loss (`state_pred_loss_weight: 0.05`)
- Re-train with these config changes and re-run diagnostic

**Target:** vision_share < 99.5%
```

---

## Checkpoint Loop

### Architecture

The prescription engine operates as an **LLM-driven research loop** with a clear autonomy model:

The autonomy model has three tiers, matching the augmentation-safety classification:

1. **Data augmentation on explicitly safe axes** (background, lighting) — **auto-runs**. The model architecture is unchanged, action labels are validated as safe, and automated guardrails (regression rollback, budget enforcement, behavioral eval, replay buffer) protect against damage.

2. **Data augmentation on task-dependent axes** (color, clutter, language) — **human reviews**. The LLM must justify why augmentation is safe for this specific task, and the human approves before execution. This prevents silently generating augmented data with invalid action labels.

3. **Training config and architecture changes** — **human approves**. The system proposes the change, shows the diff, and waits. These are higher-risk: bad hyperparameters can cause divergence, architecture changes can break weight loading, and the consequences are harder to reverse.

The rule is simple: **auto-run only when both the intervention type AND the axis are explicitly safe. Everything else gets human review.**

When no LLM is available, the system falls back to greedy rule-based mode (highest-sensitivity axis, predefined augmentation). The loop structure is identical — only the experiment proposal step changes. Code-level interventions are unavailable in fallback mode.

```
Input:
  - model checkpoint
  - dataset
  - prescription_program.md (research agenda, human-authored)
  - budget: {max_iterations, max_finetune_steps_total, max_wall_clock}

Loop:
  for round in range(max_iterations):

      # ── DIAGNOSE ──────────────────────────────────────
      report = run_diagnostic(model, dataset)

      # ── AUTOMATED GUARDRAILS ──────────────────────────
      # These run unconditionally, regardless of LLM or V1 mode.

      # Regression check
      if round > 0:
          regressions = detect_regressions(report, previous_report)
          if any(r.severity == "critical" for r in regressions):
              # Critical: automatic rollback, no human input needed.
              rollback(model, previous_checkpoint)
              emit("Critical regression detected on {axes}.
                    Rolled back to round {round-1} checkpoint.
                    Remaining issues require manual intervention.")
              break
          if regressions:
              # Minor: safe data-only iterations may continue if behavioral
              # eval is non-regressive. Surface to user before the next
              # non-safe intervention (task-dependent augmentation or code change).
              minor_regression_pending = True
              emit_warning("Minor regressions on {axes}. Will surface for
                           review before any non-safe intervention.")

      # Budget enforcement (automatic)
      if budget_exceeded(budget, elapsed):
          emit("Budget exhausted. {rounds_completed} rounds completed.
                Current state: {summary}")
          break

      # Stopping condition (automatic)
      if no_actionable_symptoms(report):
          emit("All augmentable axes are healthy. Done.
                Remaining issues: {physical_collection_axes}")
          break

      # ── SENSITIVITY SWEEP ─────────────────────────────
      sweeps = run_sweeps(model, dataset, report.symptoms)

      # Task-dependent axes are valid candidates — they just require
      # human review before execution, not auto-approval.
      # Only "unsafe" axes (e.g., position) are excluded from augmentation.
      augmentable = [s for s in sweeps
                     if s.axis_safety in ("safe", "task_dependent") and s.is_sensitive]
      if not augmentable:
          emit("No sensitive augmentable axes remain.
                Physical collection protocols: {protocols}")
          emit_collection_protocols(sweeps)
          break

      # ── EXPERIMENT PROPOSAL ────────────────────────────
      #
      # Fallback (no LLM): greedy axis selection + predefined augmentation
      # Default (LLM):     agent reads everything and proposes a targeted experiment
      # Default + retrieval (phase 2): agent also retrieves similar past experiments
      #
      if llm_available:
          proposal = llm_agent.propose_experiment(
              program=load_program_md(),
              report=report,
              sweeps=sweeps,
              history=improvement_trajectory,
              retrieval_context=past_trajectories if v3 else None,
          )
          # proposal contains:
          #   .axis: str                   — which axis to target
          #   .augmentation_spec: dict     — specific characteristics to generate
          #   .num_episodes: int           — how many (informed by GP if pilot ran)
          #   .rationale: str              — why this experiment (logged for audit)
          #   .run_pilot_first: bool       — whether to run adaptive pilot before committing
      else:
          # V1 fallback: greedy
          target_axis = max(augmentable, key=lambda s: s.max_sensitivity)
          proposal = ExperimentProposal.from_greedy(target_axis)

      # ── ADAPTIVE PILOT (if requested) ──────────────────
      # The LLM may request a pilot to estimate N before committing,
      # or it may skip the pilot if it has enough signal from the sweep
      # (e.g., sweep already shows dramatic sensitivity and budget is tight).
      #
      gp_model = None
      if proposal.run_pilot_first:
          gp_model = run_adaptive_pilot(model, dataset, proposal)

      # ── APPROVAL GATE ─────────────────────────────────
      #
      # Three approval tiers:
      #   1. Data augmentation on SAFE axis     → auto-run
      #   2. Data augmentation on TASK-DEPENDENT axis → human reviews
      #   3. Training config / architecture      → human approves
      #
      axis_safety = get_axis_safety(proposal.axis, program)  # "safe" | "task_dependent" | "unsafe"

      if proposal.intervention_type == "data_augmentation" and axis_safety == "safe":
          # Explicitly safe axis — auto-run, log for audit.
          log_auto_approved(proposal)
          choice = "A"

      elif proposal.intervention_type == "data_augmentation" and axis_safety == "task_dependent":
          # Task-dependent axis — LLM must justify safety, human reviews.
          # Report partial identification bounds: best-case (augmentation is safe)
          # vs. worst-case (augmentation partially corrupts action labels).
          bounds = compute_partial_bounds(proposal, gp_model)
          emit("""
            Round {round} — data augmentation: {proposal.axis} (TASK-DEPENDENT)

            This axis is not universally safe. The LLM's safety justification:
              {proposal.safety_justification}

            Causal DAG assumption:
              {proposal.dag_assumption}

            Partial identification bounds:
              If augmentation is fully safe:  improvement = {bounds.best_case}
              If augmentation is partially unsafe (corrupts {bounds.corruption_pct}% of labels):
                improvement = {bounds.worst_case}
              Bound width: {bounds.width} — {'narrow (safe to proceed)' if bounds.narrow else 'wide (proceed with caution)'}

            Proposed augmentation:
              {proposal.augmentation_spec}
              Episodes: {proposal.num_episodes}
              {gp_summary if gp_model else ""}

            Options:
              [A] Approve — augmentation is safe for this task
              [B] Adjust the proposal
              [C] Skip — this axis is not safe to augment
              [D] Stop
          """)
          choice = await_human_input()

      else:
          # Code changes (config or architecture) — always require approval.
          emit("""
            Round {round} — {proposal.intervention_type}: {proposal.axis}

            LLM rationale:
              {proposal.rationale}

            Proposed change:
              {proposal.code_change_spec}
              {gp_summary if gp_model else ""}

            Options:
              [A] Approve and run this experiment
              [B] Adjust the proposal
              [C] Skip this intervention
              [D] Stop — emit final report with remaining prescriptions
          """)
          choice = await_human_input()

      # ── EXECUTE ─────────────────────────────────────────
      if choice in ("A", "B"):
          if choice == "B":
              proposal = apply_human_adjustments(proposal, user_input)

          previous_checkpoint = save_checkpoint(model)

          if proposal.intervention_type == "data_augmentation":
              augmented_data = generate_augmented_episodes(
                  dataset, proposal.axis,
                  num_episodes=proposal.num_episodes,
                  spec=proposal.augmentation_spec,
              )
              model = finetune(
                  model,
                  dataset=merge(dataset, augmented_data),
                  steps=budget.steps_per_round,
                  replay_ratio=max(0.7, program.min_replay_ratio),
              )
          else:
              # Code change: coding agent applies the change in a worktree,
              # verification pipeline runs, then pilot fine-tune.
              coding_result = coding_agent.apply_change(
                  worktree_path=create_worktree(),
                  change_spec=proposal.code_change_spec,
              )
              verification = verify_code_change(coding_result.worktree_path)
              if not verification.training_step_ok:
                  log_verification_failure(verification)
                  continue  # skip to next iteration
              model = finetune_from_worktree(
                  coding_result.worktree_path, model, dataset,
                  steps=budget.steps_per_round,
              )

          # ── BEHAVIORAL EVAL ─────────────────────────────
          behavioral = run_behavioral_eval(model, dataset, config)
          if behavioral.behavioral_regression:
              rollback(model, previous_checkpoint)
              log_behavioral_regression(behavioral)
              continue  # skip to next iteration

          # ── LLM EVALUATES OUTCOME ───────────────────────
          if llm_available:
              evaluation = llm_agent.evaluate_outcome(
                  proposal=proposal,
                  metrics_before=previous_report,
                  metrics_after=report,
                  behavioral=behavioral,
                  regressions=regressions,
              )
              log(evaluation)

          previous_report = report

      elif choice == "C":
          continue
      elif choice == "D":
          break

  # ── FINAL OUTPUT ──────────────────────────────────────
  # LLM synthesizes the full trajectory into a narrative report
  # (falls back to template-based report in V1)
  emit_final_report(
      improvement_trajectory,
      remaining_prescriptions,
      collection_protocols,
      model_checkpoints,
      llm_synthesis=llm_agent.synthesize_trajectory(improvement_trajectory) if llm_available else None,
  )
```

### LLM Agent Prompt Structure

The LLM agent receives a structured prompt at each iteration containing everything it needs to propose an experiment:

```
SYSTEM: You are a research agent optimizing a robot policy. You propose
one experiment per iteration. Experiments can be data augmentation,
training config changes, or architecture modifications. Your goal is to
fix the diagnosed issues efficiently within the given budget.

Rules:
- One intervention per iteration (no combined changes).
- Data augmentation on safe axes auto-runs. Task-dependent axes and code
  changes require human approval — justify safety when proposing.
- Prioritize data augmentation first. Propose config/architecture changes
  only when data cannot fix the issue or has hit its augmentation ceiling.
- Always consider behavioral impact, not just diagnostic metric improvement.

USER:
## Research Agenda
{prescription_program.md contents, including allowed interventions and safety overrides}

## Current Diagnostic State
{diagnostic report summary: symptoms, attribution mass, key metrics}
{behavioral eval: action MSE on held-out episodes, sliced by failure mode}

## Sensitivity Sweeps
{per-axis sweep results: which variations caused which deltas}

## Augmentation Safety Classification
{per-axis: safe | task_dependent | unsafe, with overrides from program.md}

## Experiment History
Round 0: {intervention_type, axis, spec, outcome, behavioral_eval, regressions}
Round 1: ...

## Budget Remaining
{iterations left, wall-clock left, episodes budget left}
{allowed intervention types: data_augmentation, training_config, architecture}

## Available Interventions
Data augmentation:
- Augmentation axes: background (safe), lighting (safe), color (task_dependent),
  clutter (task_dependent), language (task_dependent)
- For each axis: the specific variations from the sweep and their measured deltas
- Adaptive pilot: can request a GP-based pilot to estimate N before committing

Training config changes (if --allow-config-changes):
- Training codebase at: {path}
- Current config: {summary of key hyperparameters}
- Examples: learning rate, dropout, loss weights, augmentation pipeline

Architecture changes (if --allow-architecture-changes):
- Current architecture: {summary}
- Examples: connector output tokens, attention temperature, auxiliary losses

Physical collection protocol:
- For non-augmentable axes (position, and any axis marked unsafe)

## Task
Propose the next experiment. Specify:
1. Intervention type (data_augmentation | training_config | architecture)
2. Which axis or component to target and why
3. For data: specific characteristics + episode count (or request a pilot)
   For code: specific change to make + files to modify
4. If axis is task-dependent: justify why augmentation is safe for this task
5. Expected outcome (diagnostic metrics AND behavioral impact)
```

The LLM's response is parsed into an `ExperimentProposal` — structured enough for the system to execute, but flexible enough for the LLM to propose strategies outside the predefined axis taxonomy.

### ExperimentProposal Data Model

```python
@dataclass
class ExperimentProposal:
    """An experiment proposed by the LLM agent (or V1 greedy fallback)."""

    # What to do
    axis: str                           # "background" | "lighting" | "color" | "clutter" | "language"
    augmentation_spec: dict             # specific characteristics to generate
                                        # e.g., {"textures": ["wood_grain", "tile"], "brightness_range": [-0.4, -0.2]}
    num_episodes: int                   # how many episodes to generate
    run_pilot_first: bool               # whether to run GP pilot before committing

    # Why (for logging and audit)
    rationale: str                      # LLM's reasoning (or "greedy: max sensitivity" for V1)
    expected_outcome: str               # what improvement the LLM expects
    success_criterion: str              # how to evaluate (e.g., "background_attribution < 40%")

    # Safety (for task-dependent axes)
    safety_justification: str | None    # LLM's argument for why augmentation is action-label-safe
                                        # Required when axis_safety == "task_dependent"

    # Source
    source: str                         # "llm" | "llm_rag" | "greedy_fallback"
    sweep_evidence: list[str]           # which sweep points informed this proposal

    @classmethod
    def from_greedy(cls, sweep: SensitivitySweep) -> "ExperimentProposal":
        """V1 fallback: pick highest-sensitivity axis with default characteristics."""
        return cls(
            axis=sweep.axis,
            augmentation_spec=DEFAULT_AUGMENTATION_SPECS[sweep.axis],
            num_episodes=15,  # conservative default
            run_pilot_first=True,
            rationale=f"Greedy: {sweep.axis} has highest sensitivity ({sweep.max_sensitivity:.3f})",
            expected_outcome="Reduce sensitivity on this axis",
            success_criterion=f"{sweep.axis} sensitivity < 50% of current",
            source="greedy_fallback",
            sweep_evidence=[],
        )
```

### What the LLM Can Do That V1 Can't

The LLM agent enables experiment designs that no fixed policy can produce:

**Targeted augmentation from sweep data.** The greedy fallback says "generate 15 background episodes." The LLM reads the sweep and says: "Only textured backgrounds caused large deltas (wood: 0.42, tile: 0.38). Solid colors had minimal effect (gray: 0.08, blue: 0.06). Generate 10 episodes using only wood, tile, and fabric textures — skip solid colors."

**Domain-aware design.** From `prescription_program.md`: "This robot operates in a kitchen." The LLM designs augmentations using kitchen-relevant backgrounds (granite counters, wood cutting boards) instead of abstract textures, and prioritizes overhead lighting variations over lateral.

**Adaptive pilot decisions.** The LLM can decide when a pilot is worth the compute: "The sweep shows overwhelming sensitivity (delta > 0.5) on background. A pilot to estimate N precisely is overkill — we know augmentation will help. Generate 10 targeted episodes and measure directly." Saves 15-20 minutes of pilot compute.

**Cross-iteration reasoning.** After round 1 reduced background attribution from 78% to 45%, the LLM reads the new diagnostic and notices: "Background is better but now lighting became the top symptom. However, the sweep shows lighting sensitivity is moderate (max delta 0.18). Before running a full lighting experiment, let me check if the remaining background attribution is concentrated in dark regions — if so, one more round of background augmentation targeting dark scenes might also fix lighting."

**Safety justification for task-dependent axes.** When proposing augmentation on a color/clutter/language axis, the LLM explains *why* the action labels remain valid for this specific task: "The task is 'pick up the block' with no color qualifier. Recoloring the block doesn't change the correct action. Safe to augment." The greedy fallback cannot make this judgment.

### Automated Guardrails (No Human Input Required)

These run automatically every iteration. The user does not need to approve them:

| Guardrail | Trigger | Action |
|-----------|---------|--------|
| **Critical regression** | Any previously-healthy metric crosses its critical threshold | Automatic rollback to previous checkpoint. Loop stops. |
| **Minor regression** | Non-critical metric degrades | Safe data iterations continue if behavioral eval passes. Surfaced to user before next non-safe intervention. |
| **Budget enforcement** | Wall-clock, total fine-tune steps, or iteration count exceeded | Loop stops. Emit partial results. |
| **Augmentation ceiling** | Learning curve floor is above target with no improvement trend | Flag axis as ceiling-reached. Skip to next axis. |
| **No sensitive axes** | All sweeps show flat response curves | Loop stops. Emit "model is robust" or "remaining issues need physical data." |
| **Checkpoint management** | Every round | Save model checkpoint automatically. Keep all checkpoints. |
| **Replay buffer** | Every fine-tune | Always mix ≥70% original data. Non-negotiable. |

### Human Decision Points

The system only surfaces decisions when human judgment is genuinely needed. Safe data augmentation runs without interruption.

| Decision | When | Why the system can't decide alone |
|----------|------|----------------------------------|
| Approve task-dependent augmentation? | Data augmentation proposed on color/clutter/language axis | System can't determine if action labels remain valid for this specific task |
| Approve training config change? | Config change proposed (phase 1b) | Wrong hyperparameters can cause divergence — user should review the diff |
| Approve architecture change? | Architecture change proposed (phase 1c) | Can break weight loading, change capacity — user must review diff + weight report |
| Continue after minor regression? | Minor regression is pending AND next proposal is non-safe (task-dependent or code change) | Safe data iterations auto-continue, but the user should review before escalating to riskier interventions |
| Accept augmentation ceiling? | GP floor is above target | User decides whether remaining gap matters for their deployment |
| Proceed to physical collection? | Augmentation-safe axes exhausted | Physical data collection is expensive — user decides timing |

Decisions that are **never surfaced** (handled automatically):
- Whether to run the next safe data augmentation iteration (auto-runs within budget)
- Whether to roll back on critical regression (always rolls back)
- Whether to stop on budget exhaustion (always stops)
- Whether to use replay buffer (always ≥70%, non-negotiable)

### Improvement Trajectory Output

After the loop completes, emit a full history:

```python
@dataclass
class ImprovementStep:
    """One iteration of the improvement loop."""
    round: int
    axis: str
    sweep: SensitivitySweep
    gp_model_summary: dict                  # serialized GP posterior (predictions at key N values, ceiling, uncertainty)
    pilot_points: list[PilotPoint]          # raw adaptive pilot measurements
    prescription_applied: DataPrescription
    num_augmented_episodes: int
    finetune_steps: int
    metrics_before: dict[str, float]        # all diagnostic metrics pre-iteration
    metrics_after: dict[str, float]         # all diagnostic metrics post-iteration
    regressions: list[str]                  # any metrics that worsened
    checkpoint_path: str                    # path to saved model

@dataclass
class ImprovementTrajectory:
    """Full history of the prescription loop."""
    steps: list[ImprovementStep]
    final_report: DiagnosticReport          # diagnostic after last iteration
    remaining_prescriptions: list[DataPrescription]  # unfixed axes
    collection_protocols: list[CollectionProtocol]    # physical collection needed
    total_augmented_episodes: int
    total_finetune_steps: int
    total_wall_clock: str
    model_checkpoints: list[str]            # all saved checkpoints in order
```

---

## Experiment Design

### DiagnosticState

The state representation is shared across the LLM prompt, structured logs, and the greedy fallback. It must be comparable across models and datasets, so raw metric values are normalized:

```python
@dataclass
class DiagnosticState:
    """Normalized state vector — used by LLM prompts, retrieval, fallback policy, and logging."""

    # ── Per-axis features (one entry per candidate axis) ──────
    axis_sensitivities: dict[str, float]        # max_sensitivity / baseline_action_norm
    axis_mean_sensitivities: dict[str, float]   # mean_sensitivity / baseline_action_norm
    axis_sweep_variance: dict[str, float]       # variance across sweep points (stability)

    # ── Global diagnostic features ────────────────────────────
    num_critical_symptoms: int
    num_warning_symptoms: int
    symptom_types: list[str]                    # active symptom type names
    background_attribution: float               # 0-1, from matrix
    foreground_attribution: float               # 0-1, from matrix
    positional_baseline_ratio: float            # 0-1, spatial shortcut indicator
    vision_share: float                         # 0-1, vision vs. state balance

    # ── Dataset features ──────────────────────────────────────
    dataset_size: int                           # total episodes
    background_diversity_score: float           # 0-1
    position_diversity_score: float             # min(std_x, std_y) / image_size, normalized
    lighting_diversity_score: float             # contrast_variance / max_expected_variance
    task_diversity_score: float                 # embedding_spread, normalized

    # ── Model architecture features ───────────────────────────
    num_vision_layers: int
    num_expert_layers: int
    num_vision_heads: int
    num_expert_heads: int
    connector_ratio: int                        # input_patches / output_tokens (e.g., 1024/64 = 16)

    # ── History features ──────────────────────────────────────
    rounds_completed: int
    axes_already_fixed: list[str]
    cumulative_improvement: dict[str, float]    # metric deltas from all prior rounds
    regressions_observed: int                   # count of regressions across all rounds

    @classmethod
    def from_report(
        cls,
        report: DiagnosticReport,
        sweeps: list[SensitivitySweep],
        history: list[ImprovementStep] | None = None,
    ) -> "DiagnosticState":
        """Build normalized state from a diagnostic report and sweep results."""
        ...
```

### Fallback: Greedy Rule-Based (No LLM)

Used only when no LLM API key is configured. Picks the most sensitive augmentable axis and uses predefined augmentation characteristics. Limited to data augmentation — cannot propose training config or architecture changes.

```python
class GreedyExperimentDesigner:
    """Fallback: Pick the most sensitive axis, use default augmentation spec.

    Only considers explicitly safe axes — task-dependent axes are excluded
    because there is no LLM to justify action-label safety.
    """

    def propose(self, state, sweeps, history) -> ExperimentProposal:
        safe_only = [s for s in sweeps if s.axis_safety == "safe" and s.is_sensitive]
        target = max(safe_only, key=lambda s: s.max_sensitivity)
        return ExperimentProposal.from_greedy(target)
```

Works when axes are independent. Can't design targeted augmentations, combine axes, or adapt to domain knowledge. Still useful as a baseline and for environments where LLM API access is unavailable.

### LLM Research Agent (Default)

The default and primary mode. The LLM reads the full diagnostic state, sweep data, research agenda (`prescription_program.md`), and experiment history. It proposes a specific, targeted experiment — choosing not just *which axis* but *what characteristics*, *how many episodes*, *whether to pilot first*, and *which intervention type* (data augmentation, training config, or architecture change).

The LLM handles experiment design end-to-end: axis selection, augmentation design, code change specification, and pilot decisions. It produces an `ExperimentProposal` that the system executes. This works from day one with no training data — the LLM has general knowledge about ML and the sweep data gives it specific evidence about this model.

### LLM + Retrieval Over Past Experiments (Phase 2)

In phase 2, the LLM is augmented with a **retrieval system** over logged trajectories. Before proposing an experiment, it queries a database of past (state, action, outcome) tuples from prior runs — both the user's own and (with opt-in) other users' anonymized trajectories. Retrieval is opt-in (`--enable-retrieval`) and should only be enabled after the core loop is validated.

```python
class RAGExperimentDesigner:
    """V3: LLM with retrieval over past experiments."""

    def propose(self, state, sweeps, history, program_md) -> ExperimentProposal:
        # 1. Retrieve similar past experiments
        similar = self.retriever.query(
            state=DiagnosticState.from_report(report, sweeps),
            top_k=5,
        )
        # Returns: past experiments with similar symptom profiles
        # and their outcomes (what worked, what didn't, what regressed)

        # 2. Format retrieval context for the LLM
        retrieval_context = format_similar_experiments(similar)

        # 3. LLM proposes with retrieval context
        return self.llm.propose(
            program_md=program_md,
            state=state,
            sweeps=sweeps,
            history=history,
            similar_experiments=retrieval_context,
        )
```

The retrieval context gives the LLM evidence like:

```
## Similar past experiments (from trajectory database)

1. Model with similar symptom profile (background=72%, position_std=18px):
   - Round 1: 12 textured background episodes → background dropped to 38% (success)
   - Round 2: 8 lighting episodes → lighting fixed, but gripper fixation increased (partial regression)
   - Lesson: lighting augmentation interacted negatively with gripper tracking in this architecture

2. Same model architecture, different dataset:
   - Background augmentation with only solid colors had minimal effect
   - Switching to textured backgrounds (wood, tile) was the breakthrough
   - 10 episodes was sufficient — 20 episodes showed no further improvement
```

The LLM uses this as empirical evidence alongside the current sweep data. It can reason: "Past experiments show textured backgrounds work better than solid colors for models with this architecture. Start there."

### Retrieval Index

The trajectory database indexes experiments by a normalized `DiagnosticState` vector. Similarity is computed as cosine distance over the state features. The top-K most similar past experiments are returned.

```python
@dataclass
class ExperimentRecord:
    """One indexed experiment from the trajectory database."""
    state_vector: list[float]           # normalized DiagnosticState, flattened
    axis: str                           # which axis was targeted
    augmentation_spec: dict             # what was generated
    num_episodes: int                   # how many
    outcome: dict[str, float]           # metric deltas (target + collateral)
    regressions: dict[str, float]       # any metrics that worsened
    success: bool                       # did the target metric improve?
    model_architecture: dict            # for architecture-specific matching
    source: str                         # "local" | "anonymous_shared"
```

The retrieval system is lightweight — a numpy array of state vectors with brute-force cosine search. At the scale we expect (hundreds to low thousands of experiments), no approximate nearest neighbor infrastructure is needed.

### Why Not a Bandit?

An earlier version of this spec proposed a contextual bandit as an intermediate step. The LLM-driven approach is superior because:

| Dimension | Bandit | LLM agent |
|-----------|--------|-----------|
| Cold start | Needs 50-100 trajectories | Works immediately |
| Action space | Fixed menu of axes | Open-ended (can combine axes, target specific variations) |
| Domain knowledge | Cannot incorporate | Reads prescription_program.md |
| Interpretability | Outputs a score | Explains its reasoning |
| Novel strategies | Limited to training distribution | Can propose strategies not seen in training data |
| Data efficiency | Needs many examples per symptom profile | Generalizes from few examples via language understanding |

The bandit's advantage — it discovers interaction patterns from data — is subsumed by the retrieval approach, which gives the LLM empirical evidence about interactions without requiring a separate learned model.

If a bandit is still desired (e.g., for offline environments where LLM API calls are too expensive to run in the loop), it can be layered on top of the greedy fallback using the same logged trajectories and outcome scoring described in the Retrieval Pipeline section.

---

## Structured Logging

### Purpose

Every iteration of the improvement loop produces a structured log entry. These logs serve three purposes:

1. **Audit trail** — the user can review exactly what happened and why
2. **Retrieval corpus** — past (state, action, outcome) tuples that the LLM retrieves from when proposing experiments
3. **Cross-user analytics** — aggregate patterns across models and datasets

### Log Schema

Each loop iteration produces one `IterationLog` entry. A complete run produces a `TrajectoryLog`.

```python
@dataclass
class IterationLog:
    """One iteration of the improvement loop, fully logged."""

    # ── Identity ──────────────────────────────────────────────
    trajectory_id: str              # unique ID for this full loop run
    iteration: int                  # 0-indexed round number
    timestamp: str                  # ISO 8601

    # ── State (before action) ─────────────────────────────────
    state: DiagnosticState          # normalized state vector
    symptoms: list[str]             # active symptom types
    sweep_results: dict[str, dict]  # per-axis sweep summary (sensitivity, breakpoint, robust_range)
    dataset_stats: dict             # from DatasetDiversityReport

    # ── Action taken ──────────────────────────────────────────
    axis_selected: str              # which axis the policy chose
    selection_reason: str           # "greedy_max_sensitivity" | "bandit_predicted" | "user_override"
    num_augmented_episodes: int     # how many episodes generated
    augmentation_characteristics: list[str]  # what was generated
    finetune_steps: int             # gradient steps applied
    finetune_config: dict           # lr, replay_ratio, batch_size, etc.

    # ── Outcome (after action) ────────────────────────────────
    metrics_before: dict[str, float]  # all diagnostic metrics pre-iteration
    metrics_after: dict[str, float]   # all diagnostic metrics post-iteration
    metric_deltas: dict[str, float]   # after - before for each metric
    target_metric_improvement: float  # improvement on the target axis metric
    collateral_improvements: dict[str, float]  # improvements on OTHER axes (interaction effects)
    regressions: dict[str, float]     # metrics that worsened (negative deltas)
    gp_posterior: dict                # serialized GP model: pilot points, predictions at key N values, ceiling ± std, P(target)
    num_pilots: int                   # how many adaptive pilot measurements were taken
    acquisition_trace: list[dict]     # [{n_tried, metric_after, gp_mean, gp_std, acquisition_score}, ...]
    sweep_after: dict[str, dict]      # per-axis sweep summary after fine-tune
    behavioral_eval: dict             # serialized BehavioralEval: global MSE, slice MSEs, regressions

    # ── Human decision ────────────────────────────────────────
    human_choice: str | None        # "apply" | "adjust" | "skip" | "stop" | None (auto-approved data augmentation)
    human_adjusted_n: int | None    # if user changed episode count
    human_notes: str | None         # freeform user annotation

    # ── Guardrail events ──────────────────────────────────────
    guardrail_triggered: str | None   # "regression_rollback" | "budget_exceeded" | "ceiling_reached" | None
    rollback_performed: bool

@dataclass
class TrajectoryLog:
    """Complete log of one improvement loop run."""

    # ── Identity ──────────────────────────────────────────────
    trajectory_id: str
    start_time: str
    end_time: str

    # ── Context ───────────────────────────────────────────────
    model_id: str                   # HuggingFace model ID or path
    model_architecture: dict        # layer counts, head counts, connector type
    dataset_id: str                 # dataset ID or path
    dataset_size: int               # total episodes
    initial_symptoms: list[str]     # symptoms at start
    budget: dict                    # max_iterations, max_steps, max_hours

    # ── Iterations ────────────────────────────────────────────
    iterations: list[IterationLog]

    # ── Final state ───────────────────────────────────────────
    final_symptoms: list[str]       # symptoms remaining at end
    total_improvement: dict[str, float]  # cumulative metric deltas
    total_augmented_episodes: int
    total_finetune_steps: int
    termination_reason: str         # "healthy" | "budget" | "ceiling" | "regression" | "user_stop"
    physical_collection_needed: list[str]  # axes that augmentation couldn't fix

    # ── Derived labels (for training) ─────────────────────────
    success_score: float            # 0-1, weighted metric improvement minus regressions
    axes_resolved: list[str]        # axes that went from sensitive to healthy
    axes_failed: list[str]          # axes that augmentation couldn't fix
```

### Storage

Logs are stored as JSON files alongside the improvement trajectory output:

```
outputs/prescribe_run_YYYY-MM-DD_HH-MM-SS/
  trajectory_log.json               # full TrajectoryLog
  iterations/
    iter_000/
      state.json                    # DiagnosticState snapshot
      sweep_results.json            # full sweep data
      learning_curve.json           # pilot results
      report_before.json            # diagnostic report pre-iteration
      report_after.json             # diagnostic report post-iteration
      augmented_data/               # generated episodes (if retained)
      checkpoint/                   # model checkpoint
    iter_001/
      ...
```

### Privacy and Opt-In

Trajectory logs are **local by default**. Cross-user learning (V2+) requires explicit opt-in:

```bash
# Opt in to anonymous telemetry for cross-user learning
smolvla-prescribe run --share-outcomes \
    --model your/model_id --dataset your/dataset_id

# Or set globally
export SMOLVLA_SHARE_OUTCOMES=1
```

When shared, logs are anonymized:
- Model and dataset IDs are hashed (not sent in clear)
- Architecture features, metrics, and outcomes are sent as-is (these are not PII)
- Human notes and file paths are stripped
- User identity is not collected

---

## Retrieval Pipeline (Phase 2)

**This section describes phase 2 functionality.** It should not be implemented until the core loop (phase 1) is validated end-to-end. Retrieval is opt-in via `--enable-retrieval` and requires a non-empty trajectory database.

### From Logs to Retrieval Corpus

Each `IterationLog` is indexed as an `ExperimentRecord` (defined in the V3 section above). The indexing pipeline:

```python
def index_trajectory(trajectory: TrajectoryLog) -> list[ExperimentRecord]:
    """Convert a logged trajectory into indexable experiment records."""
    records = []
    for iteration in trajectory.iterations:
        records.append(ExperimentRecord(
            state_vector=flatten_and_normalize(iteration.state),
            axis=iteration.axis_selected,
            augmentation_spec=iteration.augmentation_spec,
            num_episodes=iteration.num_augmented_episodes,
            outcome=iteration.metric_deltas,
            regressions=iteration.regressions,
            success=iteration.target_metric_improvement > 0,
            model_architecture=trajectory.model_architecture,
            source="local",
        ))
    return records
```

### Outcome Scoring

Each experiment record is annotated with a scalar outcome score for ranking retrieved results by quality:

```python
def compute_outcome_score(record: ExperimentRecord, weights: dict | None = None) -> float:
    """Score an experiment outcome for retrieval ranking.

    High scores = experiments worth emulating.
    Negative scores = experiments to learn from (what NOT to do).
    """
    if weights is None:
        weights = {
            "target_improvement": 1.0,
            "collateral_improvement": 0.5,
            "regression_penalty": -2.0,
            "efficiency_bonus": 0.1,
        }

    target_imp = record.outcome.get(record.axis, 0.0)
    collateral_imp = sum(v for k, v in record.outcome.items() if k != record.axis and v > 0)
    regression_cost = sum(abs(v) for v in record.regressions.values())
    efficiency = 1.0 / max(record.num_episodes, 1)

    return (
        weights["target_improvement"] * target_imp
        + weights["collateral_improvement"] * collateral_imp
        + weights["regression_penalty"] * regression_cost
        + weights["efficiency_bonus"] * efficiency
    )
```

The asymmetric regression penalty (2x) encodes the principle that protecting existing capabilities matters more than incremental gains.

### Retrieval Query

When the V3 LLM agent prepares a proposal, it retrieves the top-K most relevant past experiments:

```python
def retrieve_similar_experiments(
    current_state: DiagnosticState,
    index: list[ExperimentRecord],
    top_k: int = 5,
    min_similarity: float = 0.5,
) -> list[tuple[ExperimentRecord, float]]:
    """Find past experiments with similar diagnostic states."""
    current_vector = flatten_and_normalize(current_state)
    scored = []
    for record in index:
        sim = cosine_similarity(current_vector, record.state_vector)
        if sim >= min_similarity:
            scored.append((record, sim))

    # Rank by: similarity × outcome quality (surface both successes and failures)
    scored.sort(key=lambda r: r[1] * abs(compute_outcome_score(r[0])), reverse=True)
    return scored[:top_k]
```

Both successful and failed experiments are surfaced — the LLM needs to know what worked AND what didn't. Failed experiments with high similarity are especially valuable ("a model just like yours tried lighting augmentation and it caused a regression on gripper tracking").

### What the LLM Retrieves

The retrieval context is formatted as natural language for the LLM prompt:

```markdown
## Similar Past Experiments

### Experiment 1 (similarity: 0.87, outcome: positive)
- **State**: background_attribution=72%, position_std=18px, 2 critical symptoms
- **Action**: 12 textured background episodes (wood, tile, fabric)
- **Result**: background_attribution dropped to 38%. Collateral: position sensitivity
  also decreased by 15% (interaction effect).
- **Lesson**: Textured backgrounds worked well. Solid colors had minimal effect.

### Experiment 2 (similarity: 0.81, outcome: negative)
- **State**: background_attribution=65%, lighting_variance=3.2
- **Action**: 8 lighting augmentation episodes (brightness ±40%)
- **Result**: Lighting robustness improved, BUT gripper fixation increased from 25% to 38%.
- **Lesson**: Lighting augmentation can interact negatively with gripper tracking.
  Consider combining with gripper masking.

### Experiment 3 (similarity: 0.79, outcome: positive)
- **State**: Same model architecture (12L vision, 16L expert, 8H cross-attn)
- **Action**: 10 background episodes — ONLY kitchen-relevant textures (granite, wood)
- **Result**: background_attribution dropped to 31%. 10 episodes was sufficient.
- **Lesson**: Domain-targeted augmentation is more efficient than generic diversity.
```

The LLM synthesizes these into its proposal: "Past experiments show textured backgrounds work, solid colors don't. Lighting augmentation risks gripper regression. Domain-targeted textures are more efficient. I'll propose 10 kitchen-relevant textured background episodes and skip lighting for now."

### Bootstrapping

| Local trajectories | Shared trajectories | Behavior |
|-------------------:|--------------------:|----------|
| 0 | 0 | V2 (LLM without retrieval) or V1 (greedy) |
| 1-5 | any | V3 with limited local retrieval + shared corpus |
| 5+ | any | V3 with full retrieval |

The system automatically includes shared trajectories (if opted in) to bootstrap the retrieval corpus. Even one local trajectory is useful — the LLM can compare the current state to the user's own prior run.

### Patterns That Emerge From Retrieval

Over time, the retrieval corpus accumulates patterns that neither V1 nor V2 (without retrieval) can surface:

- **Interaction effects**: "When background sensitivity > 0.3 AND position sensitivity > 0.2, fixing background first also reduces position sensitivity by ~20%."
- **Diminishing returns by axis type**: "Lighting augmentation rarely improves beyond 15 episodes regardless of sensitivity."
- **Architecture-dependent patterns**: "Models with 16-head experts respond better to color augmentation than models with 8-head experts."
- **Dataset-size effects**: "For datasets with <50 episodes, fixing background diversity first is almost always correct."
- **Domain-specific strategies**: "Kitchen environments benefit from granite/wood textures. Lab environments benefit from varied table surfaces."

These patterns are not hand-coded — they emerge from the data and are surfaced to the LLM through retrieval context.

---

## Code-Level Interventions (Phase 1b/1c — not required for v1a)

> **Scope note:** Everything in this section is exploratory direction for phases 1b and 1c. It is not part of the v1a acceptance bar.

### Motivation

The diagnostic agent already identifies issues that data augmentation cannot fix:

| Symptom | Why data can't fix it | Code intervention needed |
|---------|----------------------|------------------------|
| `dead_state_pathway` | Proprioceptive input is architecturally ignored | Add dropout to vision pathway, add state prediction aux loss |
| `connector_bottleneck` | Pixel shuffle compresses 1024→64 tokens, losing information | Increase connector output tokens, change pooling strategy |
| `cross_attention_diffuse` | Action expert uses near-uniform attention | Add temperature scaling, adjust attention dropout |
| `gripper_fixation` | Model over-attends to gripper regardless of data | Add gripper masking augmentation to training pipeline |
| `language_insensitivity` (persistent) | Model ignores language even after task paraphrasing | Add contrastive language loss, language dropout in training |
| Spectral alpha out of range | Weight matrices are over/under-correlated | Adjust learning rate, add/modify weight decay |

These require changes to the **training config** (hyperparameters, loss functions, augmentation pipeline) or **model architecture** (connector size, auxiliary modules, attention mechanisms). The prescription engine handles these through a headless coding agent.

### Headless Coding Agent

The coding agent is an LLM (Claude Code in headless mode, Codex, or similar) that receives a precise modification spec and makes code changes in an isolated environment, verifying its own work before the research loop evaluates the result.

#### Execution Model

```
Research agent (proposes experiment)
  → Coding agent (makes code changes in isolated worktree)
    → Verification pipeline (compilation, shape check, smoke test)
      → Research loop (pilot fine-tune, re-diagnostic)
        → Keep or discard (merge branch or delete worktree)
```

#### Isolation

Every code intervention runs in a **git worktree** — a separate working copy of the training codebase on a dedicated branch:

```bash
# System creates an isolated worktree for each code experiment
git worktree add /tmp/prescribe_iter_003 -b prescribe/iter-003-connector-256

# Coding agent modifies files in the worktree
# Training runs from the worktree
# If successful: branch available for user review and merge
# If failed: worktree and branch are cleaned up
```

The user's main branch is never modified. Every code change is a reviewable diff on an experiment branch.

#### Coding Agent Prompt

The coding agent receives a structured prompt with the modification spec, relevant file contents, and verification requirements:

```
SYSTEM: You are a coding agent modifying a robot policy training codebase.
You will receive a specific modification to make. Make ONLY the requested change.
Verify your work compiles and passes shape checks before returning.

USER:
## Modification Request
{from the research agent's ExperimentProposal}

Example: "Increase the connector output tokens from 64 to 128.
The connector uses pixel shuffle in smolvla/modeling.py.
Change the output_tokens parameter and update any downstream
layers that depend on the token count (attention dimensions,
positional embeddings)."

## Relevant Files
{file contents of files that need modification, read by the system}

## Training Config
{current config YAML}

## Verification Requirements
1. Code compiles without errors
2. Model instantiates with the new config
3. A forward pass with dummy input produces the correct output shape
4. Training loop runs for 1 step without errors

## Constraints
- Do not modify the vision encoder (SigLIP weights are frozen)
- Do not change the VLM base model architecture
- Preserve all existing model checkpoint loading paths
- Add comments explaining what was changed and why
```

#### Verification Pipeline

Before the research loop evaluates a code change, the system runs an automatic verification pipeline. This is **not optional** — it gates all code changes regardless of the coding agent's confidence:

```python
@dataclass
class CodeVerification:
    """Result of the automated verification pipeline."""
    compilation_ok: bool            # does the modified code import without errors?
    model_instantiation_ok: bool    # does the model construct with new config?
    forward_pass_ok: bool           # does a forward pass produce correct shapes?
    training_step_ok: bool          # does one training step complete?
    shape_report: dict | None       # input/output shapes at key layers
    error_message: str | None       # first error encountered, if any
    diff: str                       # git diff of changes made

def verify_code_change(worktree_path: str, config: dict) -> CodeVerification:
    """Run the verification pipeline on a code change.

    Runs in the isolated worktree. Failures at any stage stop the pipeline.
    """
    # Stage 1: Import check
    result = run_in_worktree(worktree_path, "python -c 'import train'")
    if result.returncode != 0:
        return CodeVerification(compilation_ok=False, ...)

    # Stage 2: Model instantiation
    result = run_in_worktree(worktree_path,
        "python -c 'from train import build_model; m = build_model(config)'")
    if result.returncode != 0:
        return CodeVerification(compilation_ok=True, model_instantiation_ok=False, ...)

    # Stage 3: Forward pass with dummy input
    result = run_in_worktree(worktree_path,
        "python verify_shapes.py --config {config}")
    if result.returncode != 0:
        return CodeVerification(..., forward_pass_ok=False, ...)

    # Stage 4: One training step
    result = run_in_worktree(worktree_path,
        "python train.py --config {config} --max-steps 1 --no-save")
    if result.returncode != 0:
        return CodeVerification(..., training_step_ok=False, ...)

    return CodeVerification(
        compilation_ok=True,
        model_instantiation_ok=True,
        forward_pass_ok=True,
        training_step_ok=True,
        shape_report=parse_shape_report(result),
        diff=get_diff(worktree_path),
    )
```

If verification fails, the coding agent gets one retry with the error message. If it fails again, the experiment is marked as failed and the research loop moves on.

### Intervention Types in Detail

#### Training Config Changes

The safest code intervention. The model architecture stays the same — only training dynamics change.

**What the coding agent modifies:**
- Training config YAML (learning rate, weight decay, dropout rates, batch size)
- Augmentation pipeline (add color jitter, random erasing, gripper masking to the training dataloader)
- Loss function weights (add auxiliary losses, adjust action loss weighting)
- Optimizer settings (switch optimizer, adjust momentum, add gradient clipping)

**Example experiment proposals:**

| Diagnostic finding | Proposed config change |
|-------------------|----------------------|
| `dead_state_pathway` (vision_share > 99.5%) | Add `vision_dropout: 0.1` to force reliance on state input |
| `gripper_fixation` (gripper > 40% GradCAM) | Add `gripper_mask_augmentation: {probability: 0.3}` to training pipeline |
| `language_insensitivity` (persistent after data) | Add `language_dropout: 0.1` + `contrastive_language_loss_weight: 0.05` |
| Spectral alpha > 6.0 (undertrained) | Increase `learning_rate` by 2x, add `warmup_steps: 500` |
| `unstable_gradcam` (high variance) | Add `gradient_clip_norm: 1.0`, reduce `learning_rate` by 0.5x |

**Verification:** standard pipeline (import → instantiate → forward → one step). Low risk because model weights load unchanged.

#### Architecture Changes

Higher risk. The model structure changes, which may break pretrained weight loading or change model capacity.

**What the coding agent modifies:**
- Model config (connector output tokens, number of attention heads, hidden dimensions)
- Model code (add new modules, modify forward pass, add skip connections)
- Weight initialization for new parameters (random init for new layers, preserve existing weights)

**Example experiment proposals:**

| Diagnostic finding | Proposed architecture change |
|-------------------|-----------------------------|
| `connector_bottleneck` (information loss > 0.3) | Increase connector output from 64 to 128 tokens |
| `cross_attention_diffuse` (entropy > 5.0 bits) | Add learnable temperature parameter to cross-attention softmax |
| `dead_state_pathway` (persistent after config change) | Add state prediction auxiliary head (MLP that predicts next state from current) |
| High head redundancy (> 0.9) | Prune redundant heads, fine-tune remaining |

**Verification:** extended pipeline — also checks that pretrained weights load for unchanged layers and that new parameters are correctly initialized.

**Weight loading strategy for architecture changes:**

```python
def load_with_architecture_change(
    new_model,
    pretrained_checkpoint: str,
    strict: bool = False,
):
    """Load pretrained weights into a modified architecture.

    - Matching layers: load pretrained weights exactly
    - New layers: initialize randomly (or from a specified init scheme)
    - Removed layers: skip (warn)
    - Shape-changed layers: skip and warn (user must decide)
    """
    pretrained_state = torch.load(pretrained_checkpoint)
    new_state = new_model.state_dict()

    loaded, skipped, new = [], [], []
    for key in new_state:
        if key in pretrained_state and pretrained_state[key].shape == new_state[key].shape:
            new_state[key] = pretrained_state[key]
            loaded.append(key)
        elif key in pretrained_state:
            skipped.append((key, pretrained_state[key].shape, new_state[key].shape))
        else:
            new.append(key)

    new_model.load_state_dict(new_state)
    return {"loaded": loaded, "skipped": skipped, "new_parameters": new}
```

### Intervention Risk Tiers and Guardrails

The guardrail system matches the three-tier approval model:

| Tier | Intervention | Approval | Phase | Extra guardrails |
|------|-------------|----------|-------|------------------|
| 1a | Data augmentation (safe axis) | Auto-run | 1a | Replay buffer, behavioral eval (global + sliced), regression rollback |
| 1b | Data augmentation (task-dependent axis) | Human reviews LLM safety justification | 1a | Same as 1a + LLM must justify action-label validity |
| 2 | Training config | Human approves diff | 1b | Verification pipeline, config snapshot, behavioral eval, regression rollback |
| 3 | Architecture | Human approves diff + weight loading report | 1c | Verification pipeline, weight loading audit, extended pilot (more steps), behavioral eval, regression rollback |

For tier 3, the human checkpoint shows:

```
Round 4 — Architecture: increase connector tokens (64 → 128)

Coding agent diff:
  configs/smolvla_base.yaml: connector_output_tokens: 64 → 128
  smolvla/modeling.py: +3 lines (update positional embedding size)

Verification: ✓ compile ✓ instantiate ✓ forward pass ✓ training step
Weight loading: 847/850 parameters loaded. 3 new parameters (connector pos_embed).

LLM rationale:
  "Connector information loss is 0.34 — significant detail is dropped during
   pixel shuffle. Doubling output tokens from 64 to 128 preserves more spatial
   information. The pilot will tell us if the extra tokens improve object
   attribution without slowing inference significantly."

Estimated pilot time: 8 minutes (100 steps, extended for architecture change)

Options:
  [A] Run pilot with this architecture change
  [B] View full diff
  [C] Skip — try a different approach
  [D] Stop
```

### ExperimentProposal Extensions

The `ExperimentProposal` model gains fields for code interventions:

```python
@dataclass
class ExperimentProposal:
    # ... existing fields (axis, augmentation_spec, num_episodes, rationale, etc.) ...

    # Intervention type
    intervention_type: str          # "data_augmentation" | "training_config" | "architecture"

    # Code change spec (for training_config and architecture interventions)
    code_change_spec: dict | None   # structured description of what to change
                                    # e.g., {"file": "configs/smolvla.yaml",
                                    #        "change": "connector_output_tokens: 64 → 128",
                                    #        "rationale": "reduce connector information loss"}
    files_to_modify: list[str] | None  # paths the coding agent should read/modify
    verification_requirements: list[str] | None  # any extra checks beyond standard pipeline

    # NOTE: Combined interventions (modifying code AND data in one iteration)
    # are not supported in v1a. Propose them as sequential iterations instead.
    # This field is reserved for future use when combined interventions are
    # explicitly opted in via prescription_program.md.
    # combined_with: list[str] | None  # RESERVED — not active in v1a
```

### Coding Agent Integration

The coding agent is invoked through a standard interface that abstracts the specific LLM backend:

```python
class CodingAgent(Protocol):
    """Interface for headless coding agents that modify training code."""

    def apply_change(
        self,
        worktree_path: str,
        change_spec: dict,
        files_to_read: list[str],
        constraints: list[str],
    ) -> CodingResult:
        """Make the specified code change in the worktree.

        Returns the result including the diff and any warnings.
        """
        ...

@dataclass
class CodingResult:
    success: bool
    diff: str                       # git diff of changes made
    files_modified: list[str]
    warnings: list[str]             # any concerns the coding agent flagged
    explanation: str                 # what was changed and why

# Implementations
class ClaudeCodeHeadless(CodingAgent):
    """Claude Code running in headless mode (--print flag or SDK)."""
    ...

class CodexAgent(CodingAgent):
    """OpenAI Codex / GPT-4 with code execution."""
    ...

class LocalCodingAgent(CodingAgent):
    """Local model (e.g., DeepSeek Coder via Ollama) for offline environments."""
    ...
```

### One Intervention Per Iteration (Default)

By default, the system applies **one intervention per iteration**. This is a deliberate constraint: when you change one thing at a time, you know exactly what caused the improvement (or regression). Combined interventions make causal attribution murky — if you add dropout AND augment data simultaneously, and the metric improves, you don't know which change helped. And if it regresses, you don't know which to roll back.

The LLM agent should propose the single most impactful intervention for each round. If it believes a code change and data augmentation are both needed (e.g., dead state pathway requires dropout AND state-diverse data), it should propose them as **sequential iterations**, not a combined one:

```
Round 5: Add vision_dropout=0.1 (training config change)
  → Re-diagnostic shows vision_share dropped from 99.7% to 95.2%
  → Partial improvement, but state pathway still underutilized

Round 6: Generate 8 episodes with varied gripper positions (data augmentation)
  → With dropout active, the model now uses state input
  → vision_share drops to 91.2% ✓
```

This way, the log clearly shows that dropout alone got partial credit and augmented data completed the fix. The retrieval corpus learns this two-step pattern for future models.

> **Later phases:** Combined interventions (modifying code AND data in one iteration) may be supported as an opt-in exception in phase 1b+. They would require explicit opt-in via `prescription_program.md`, and logged with a `causal_attribution: "ambiguous"` flag. Not part of v1a.

---

## Integration With smolvla-inspect

### Consumed Interfaces

`smolvla-prescribe` imports from `smolvla-inspect` without modifying it. The integration points:

| What prescribe consumes | Where it lives in inspect | How it's used |
|------------------------|--------------------------|---------------|
| `DiagnosticReport` | `diagnostic.models` | Input to the research loop — symptoms, metrics, findings drive experiment proposals |
| `DiagnosticMatrix` | `diagnostic.models` | Attribution mass per region per signal — feeds sweep axis selection |
| `DatasetDiversityReport` | `diagnostic.models` | Current dataset stats — informs collection protocols |
| Counterfactual primitives | `diagnostic.counterfactual` | Reused by sweeps (called in parameterized loops instead of once) |
| `compare_diagnostic_runs()` | `diagnostic.comparison` | Regression detection between iterations |
| `run_diagnostic()` | `diagnostic.__init__` | Full re-diagnostic after each iteration |

### Counterfactual Primitives in Sweep Mode

The sweep calls existing counterfactual primitives in a loop with varied parameters:

```python
# In smolvla-inspect (unchanged): single test
result = background_substitution(policy, sample, ..., replacement="gray")

# In smolvla-prescribe: parameterized sweep
from smolvla_inspect.diagnostic.counterfactual import background_substitution

for replacement in ["gray", "noise", "blur", "wood_texture", "tile_texture", ...]:
    point = background_substitution(policy, sample, ..., replacement=replacement)
    sweep.points.append(point)
```

This requires extending `background_substitution` in smolvla-inspect to accept texture image paths in addition to `"gray"` / `"noise"` / `"blur"`. This is the only change needed in smolvla-inspect — a backward-compatible extension to the existing primitives.

### What Does NOT Change in smolvla-inspect

- No new files added to smolvla-inspect
- No new dependencies
- The `Finding` dataclass is not modified (prescriptions are a separate output in smolvla-prescribe, not a field on `Finding`)
- The MCP server, web viewer, CLI, and all existing functionality are unaffected

### CLI (smolvla-prescribe)

```bash
# Run prescription engine (includes diagnostic + sweeps + pilots + loop)
smolvla-prescribe run \
    --model your/model_id \
    --dataset your/dataset_id \
    --episode 0 \
    --device cuda \
    --max-iterations 5 \
    --max-finetune-steps 500 \
    --budget-hours 2

# With a research agenda
smolvla-prescribe run \
    --model your/model_id \
    --dataset your/dataset_id \
    --program prescription_program.md

# Run sweeps only (no fine-tuning, just sensitivity analysis)
smolvla-prescribe sweep \
    --model your/model_id \
    --dataset your/dataset_id

# Allow code-level interventions (training config + architecture changes)
smolvla-prescribe run \
    --model your/model_id \
    --dataset your/dataset_id \
    --training-codebase /path/to/lerobot-training \
    --allow-config-changes \
    --allow-architecture-changes \
    --coding-agent claude-code       # or: codex, local

# Data-only mode (no code changes, only augmentation)
smolvla-prescribe run \
    --model your/model_id \
    --dataset your/dataset_id \
    --data-only

# Generate augmented data from a completed prescription report
smolvla-prescribe generate \
    --from-report outputs/prescribe_run/prescription_report.json \
    --source-dataset your/dataset_id \
    --output-dataset ./augmented_data
```

---

## Open Questions

> **Scope note:** Items in the "Sweep & Pilot Design" and "LLM Research Agent" sections are relevant to v1a. Items in "Code-Level Interventions" and "Retrieval Pipeline" are exploratory for later phases and not required for v1a freeze.

### Sweep & Pilot Design

1. **Texture library for background sweeps.** Ship a small built-in set of textures? Let users provide a folder? Generate procedurally (Perlin noise, checkerboard, gradients)?

2. **Fine-tune integration.** SmolVLA training happens through LeRobot's training infrastructure, not this repo. The pilot fine-tune needs either: (a) a thin wrapper around LeRobot's trainer, or (b) a standalone fine-tune loop for short runs. Option (b) is simpler but may diverge from the user's actual training setup.

3. **Action validity for position augmentation.** Could we support limited position augmentation (small shifts where action interpolation is approximately valid)? The sweep would reveal the safe range. Risky but potentially useful for cases where physical collection is very expensive.

4. **Multi-episode sweeps.** Current design sweeps one frame. Should we sweep across multiple episodes and aggregate? More robust but proportionally more expensive.

5. **Pilot independence.** Current design starts each pilot from the original checkpoint (independent measurements). An alternative is incremental pilots (each builds on the previous). Incremental is cheaper but measurements are correlated and harder to interpret.

6. **GP kernel selection.** Matérn 5/2 with a negative linear mean is the default. For models that show phase transitions (sudden improvement after enough diversity), a changepoint kernel might be more appropriate. Worth auto-detecting from pilot data, or always use the simpler kernel and let more pilot points capture the shape?

6b. **Monotonic GP.** Enforcing monotonicity (more data shouldn't make things worse *in expectation*) is principled but adds implementation complexity. The soft approach (negative-trend mean function) is simpler. If a pilot measurement violates monotonicity, is that noise (GP handles naturally) or a real signal (non-monotonic kernel needed)?

7. **Physical collection protocols.** How specific can we get? We know the workspace bounds from detected object positions, but we don't know physical constraints (table edges, robot reach limits, obstacles). Should we ask the user for workspace parameters?

### LLM Research Agent

8. **Proposal parsing robustness.** The LLM proposes experiments in natural language, which must be parsed into a structured `ExperimentProposal`. How strict should the parsing be? Options: (a) structured output format (JSON) enforced via system prompt, (b) flexible natural language parsed with a second LLM call, (c) hybrid — structured fields with a freeform rationale. Option (a) is most reliable for execution.

9. **LLM cost management.** Each iteration requires at least one LLM call (proposal) and optionally a second (evaluation). With 3-5 iterations per run, that's 6-10 LLM calls. At current API prices this is negligible, but should we support local models (Ollama/vLLM) like the diagnostic agent already does?

10. **Program.md versioning.** The user may refine `prescription_program.md` between iterations of the same loop run. Should the system detect changes and re-read it each iteration, or snapshot it at loop start? Re-reading is more flexible; snapshotting is more reproducible.

11. **LLM proposal guardrails.** The LLM might propose something the system can't execute (e.g., "generate episodes with a different robot arm"). Should the system validate proposals against available augmentation primitives before presenting to the user? Or trust the LLM + human checkpoint to catch errors?

12. **Outcome scoring weights.** The regression penalty (2x) in `compute_outcome_score` is a judgment call. Too high and retrieval surfaces only conservative experiments. Too low and risky strategies appear equally good. User checkpoint decisions (apply/skip/rollback) are implicit feedback — could tune weights from these signals.

### Code-Level Interventions

13. **Training codebase coupling.** The coding agent needs to understand the user's training codebase (file structure, config format, model class hierarchy). Should we support only LeRobot's standard layout, or be codebase-agnostic? LeRobot-specific is simpler and covers the primary use case. Codebase-agnostic requires the coding agent to explore and understand arbitrary training setups.

14. **Coding agent verification depth.** The current verification pipeline is 4 stages (compile → instantiate → forward → one step). Should we add: (a) backward pass check (gradients flow correctly), (b) multi-step smoke test (10 steps, check loss decreases), (c) memory profiling (architecture change doesn't OOM)? More stages = safer but slower.

15. **Coding agent retry policy.** If the coding agent's change fails verification, it gets one retry with the error message. Should this be configurable? Some failures (import error) are easy to fix in one retry. Others (shape mismatch deep in the model) may need the research agent to reformulate the spec entirely.

16. **Worktree lifecycle.** Successful experiment branches accumulate. Should the system auto-clean branches older than N days? Merge them into a single "prescription" branch? Or leave all branch management to the user?

17. **Architecture change + pretrained weight compatibility.** When the coding agent changes model architecture, some pretrained weights may not load. The weight loading strategy (load matching, skip changed, init new) is a heuristic. Should the system support more sophisticated approaches like distillation from the original model into the modified architecture?

### Retrieval Pipeline

18. **Retrieval similarity metric.** Cosine similarity over flattened `DiagnosticState` treats all features equally. Some features (symptom types, architecture) may matter more for relevance than others (exact metric values). Learned distance metrics or feature weighting could improve retrieval quality.

19. **Shared corpus governance.** With opt-in anonymous sharing, who hosts the shared trajectory database? Options: (a) centralized service (simplest but requires infrastructure), (b) peer-to-peer (complex), (c) bundled with releases (static snapshots, no infra needed). Option (c) is pragmatic for launch.

20. **Negative experiment surfacing.** Failed experiments are valuable ("don't do X, it caused a regression"). But how many failures should we surface vs. successes? Too many failures and the LLM becomes overly cautious. Current approach: surface by absolute outcome score (both high positive and high negative), letting the LLM judge.

21. **Cross-architecture retrieval.** Should retrieval match on model architecture features, or just on symptom profiles? A background shortcut in a 500M model may require different treatment than in a 2B model. Architecture-weighted similarity could help, but reduces the effective corpus size.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | CLEAR | 6 proposals, 6 accepted, 0 deferred |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 5 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |

- **OUTSIDE VOICE:** 10 findings from independent Claude subagent. 2 acted on (placebo test → sanity check, LLM benchmark added). Remaining findings addressed by existing review decisions or are philosophical scope disagreements.
- **ENG REVIEW AMENDMENTS:** LeRobot integration spike (Phase 0), --resume flag, segmentation mask caching, axis selection DRY utility, undefined variable fix, dual-layer test strategy (mocks + fixtures).
- **UNRESOLVED:** 0 across all reviews
- **VERDICT:** CEO + ENG CLEARED — ready to implement.
