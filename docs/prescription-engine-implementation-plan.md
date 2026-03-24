# Prescription Engine — Implementation Plan

**Status:** Reviewed — CEO + Eng cleared, ready to implement
**Date:** 2026-03-23
**Spec:** [`docs/prescription-engine-spec.md`](prescription-engine-spec.md) (~2500 lines — full architecture, data models, protocols)
**Repo:** `smolvla-prescribe` (new, separate from `smolvla-inspect`)

---

## 1. What We're Building

Transform diagnostic findings from smolvla-inspect into precise, empirically-validated data prescriptions. The system measures how sensitive a model is to each data axis (background, lighting, position, etc.), runs GP-backed pilot fine-tunes to determine exact episode counts, auto-augments safe axes, and produces validated prescriptions with calibrated uncertainty.

**v1a success contract:** An intervention is successful only if (1) the diagnostic target metric improves, (2) no behavioral regression is detected (global or slice-based), and (3) the change survives automated rollback checks.

### Core Pipeline

```
Model + Dataset
    → Diagnostic (via smolvla-inspect)
    → Sensitivity Sweeps (per axis: ~60 forward passes)
    → Pairwise Interaction Detection (top 2-3 axes, ~9 passes/pair)
    → GP Adaptive Pilots (2-4 fine-tunes per axis, ~5 min each)
    → Auto-Augment Safe Axes (generate data + fine-tune + eval)
    → Behavioral Eval (global + slice-based MSE + sanity check)
    → Prescription Report (completed interventions + collection protocols)
```

---

## 2. Scope — What's In and What's Out

### In Scope (v1a + accepted expansions)

| Component | Module | Description |
|-----------|--------|-------------|
| Sensitivity sweeps | `sweep.py` | Parameterized counterfactual loops, 6 axes, safety classification |
| Pairwise interaction detection | `sweep.py` | 3x3 factorial sweep on top 2-3 sensitive axis pairs |
| GP adaptive pilots | `pilot.py` | sklearn GP, Matern 5/2 kernel, acquisition function |
| Loop orchestrator | `loop.py` | Iteration management, approval gates, guardrails, `--dry-run`, `--resume` |
| LLM research agent | `experiment_agent.py` | Proposal generation, outcome evaluation, prompt construction |
| Greedy fallback | `experiment_agent.py` | No-LLM mode: max sensitivity safe axis + default augmentation |
| Behavioral eval | `verification.py` | Global MSE, slice-based MSE, sanity check (10 permutations) |
| Structured logging | `logging.py` | IterationLog, TrajectoryLog, JSON serialization, resume deserialization |
| Collection protocols | `collection_protocol.py` | Physical data collection guides for non-augmentable axes |
| Data models | `models.py` | 15 dataclasses — one per domain concept |
| CLI | `run_prescribe.py` | Entry point, argument parsing |
| Config | `configs/` | `prescription_targets.yaml`, default `prescription_program.md` |
| Web dashboard | `dashboard/` | Interactive GP curves, improvement trajectories, behavioral eval charts |
| MCP server | `mcp_server.py` | Tools: run_sweep, get_prescription_report, list_trajectories, propose_experiment |
| Checkpoint comparison | `comparison.py` | Multi-checkpoint table with Pareto frontier analysis |
| Sweep caching | `loop.py` | Cache unchanged sweep results, invalidate on detected interactions |
| Segmentation mask caching | augmentation pipeline | Cache source episode masks, reuse across iterations |

### NOT in Scope

| Item | Reason |
|------|--------|
| Phase 1b — training config changes via coding agent | Separate phase, ships after v1a stable |
| Phase 1c — architecture changes via coding agent | Highest risk, ships last |
| Phase 2 — retrieval over logged trajectories | Requires trajectory corpus, enabled after core loop validated |
| Combined interventions (data + code in one iteration) | Excluded for causal attribution clarity |
| Cross-user sharing infrastructure | Phase 2+, local-only for v1a |
| Monotonic GP constraint | Optional enhancement, standard GP sufficient |
| Bandit-based experiment selection | Superseded by LLM agent |
| Simulator rollout evaluation (Tier 3) | Opt-in, not required for v1a |

---

## 3. Tech Decisions (Resolved)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| GP library | scikit-learn `GaussianProcessRegressor` | Sufficient for 2-10 data points. Simpler than GPyTorch. CPU-only is fine at this scale. |
| Fine-tune integration | Thin wrapper around LeRobot's trainer | Matches user's actual training setup. Validates with Phase 0 spike. |
| Texture library | Built-in set + user-provided folder | Both. Built-in covers common cases, user folder enables domain-specific textures. |
| Placebo test | Renamed to "sanity check" — 10 permutations, no p-value claims | 20 permutations can't detect p<0.05. Honest framing: "is this obviously not noise?" |
| Dataclass design | 15 classes, one per domain concept | Explicit over consolidated. Worth the extra 30 lines for domain clarity. |
| Test strategy | Dual-layer: mocks for unit tests, fixtures for integration | Mocks for speed (business logic), fixtures for truth (fine-tune, GP, eval) |

---

## 4. Implementation Order

### Phase 0: LeRobot Integration Spike (do this FIRST)

**Goal:** Validate the fine-tune wrapper before building anything else. If fine-tuning doesn't work, none of the rest matters.

**What to prove:**
1. LeRobot's trainer supports short runs (50-100 steps) with custom stop
2. Mixed dataset (original + augmented) works with replay ratio
3. Checkpoint saving at arbitrary points
4. Checkpoint format is compatible between LeRobot and smolvla-inspect's model loading
5. Augmented data format matches LeRobot's expected episode structure

**Deliverable:** `prescribe/finetune_wrapper.py` with a working integration test that fine-tunes a tiny SmolVLA checkpoint for 50 steps on a 5-episode dataset and saves a loadable checkpoint.

**Effort:** human: ~2 days / CC: ~30 min

### Phase 1: Core Data Models + Sweep Infrastructure

```
prescribe/
├── models.py              # All 15 dataclasses
├── sweep.py               # Single-axis sweeps + pairwise interaction detection
└── configs/
    └── prescription_targets.yaml
```

**Key implementation details:**
- Reuse smolvla-inspect counterfactual primitives in parameterized loops
- Extend `background_substitution` in smolvla-inspect to accept texture image paths (backward-compatible)
- Axis safety classification: defaults + `prescription_program.md` overrides
- `get_augmentable_axes(sweeps, program)` utility (DRY — used by greedy, loop, and prompt construction)
- Pairwise interaction: 3x3 factorial on top 2-3 sensitive axes after single-axis sweeps

### Phase 2: GP Pilot Infrastructure

```
prescribe/
└── pilot.py               # build_gp(), next_batch_size(), run_adaptive_pilot()
```

**Key implementation details:**
- sklearn `GaussianProcessRegressor` with Matern 5/2, lengthscale=15.0, bounds=(5.0, 100.0)
- Anchor at N=0 with near-zero noise (alpha=0.001 for anchor, 0.01 for pilots)
- `estimated_noise_impact_per_episode = gp.kernel_.k2.noise_level / len(pilot_points)` (derived from GP observation noise)
- Acquisition function: maximize `std_n * exp(-0.5 * (distance_to_target / std_n)^2)`
- Fallbacks: LinAlgError → heuristic (15 episodes), ConvergenceWarning → check posterior variance → heuristic if unreasonable, NaN variance → clamp to metric range

### Phase 3: LLM Agent + Greedy Fallback

```
prescribe/
└── experiment_agent.py    # LLMExperimentDesigner, GreedyExperimentDesigner
```

**Key implementation details:**
- LLM prompt: system message + research agenda + diagnostic state + sweeps + history + budget + available interventions
- Response format: structured JSON (enforced via system prompt) parsed into `ExperimentProposal`
- Retry logic: malformed JSON → retry with "respond in JSON" hint, refusal → greedy fallback, timeout → retry 1x then greedy, invalid axis → retry with clarification
- Greedy: `max(safe_only, key=lambda s: s.max_sensitivity)` with `run_pilot_first=True`
- Support local models via existing Ollama/vLLM pattern from smolvla-inspect diagnostic agent

### Phase 4: Behavioral Eval + Sanity Check

```
prescribe/
└── verification.py        # run_behavioral_eval(), sanity_check()
```

**Key implementation details:**
- Two perturbation pools per axis: training pool (augmentation) and eval pool (held-out, never seen in training)
- Eval pool generated once at loop start, held constant across iterations
- Tier 1: Global action MSE on all held-out episodes
- Tier 2: Slice-based MSE keyed by diagnostic failure mode (novel_background, target_occluded, etc.)
- Regression thresholds: global MSE >10%, any slice >15%
- Sanity check: 10 permutations (shuffled augmented vs. original), no p-value claims, just "is improvement obviously not noise?"
- Skip sanity check if improvement > 3x GP posterior std

### Phase 5: Loop Orchestrator

```
prescribe/
└── loop.py                # run_prescription_loop(), signal handlers, state serialization
```

**Key implementation details:**
- Three approval tiers: safe auto-run, task-dependent human review, code change human approve
- Automated guardrails (no human input): critical regression → rollback + stop, budget → stop, no axes → stop, replay ≥ 70%
- `--dry-run`: run sweeps + LLM proposals + cost estimation, skip fine-tune/augment
- `--resume`: deserialize `interrupted_state.json` → restore checkpoint, GP posteriors, sweep cache, budget counters, iteration index
- Signal handling: SIGINT/SIGTERM → wait for current training step → save state → exit gracefully
- File locking on checkpoint directory during a run (prevents concurrent runs)
- Sweep caching: if no augmented episodes touched axis X and no interaction detected between X and the modified axis, cache X's sweep results
- Human checkpoint timeout: 30 min configurable, auto-skip with warning

### Phase 6: Logging + Collection Protocols + Report

```
prescribe/
├── logging.py             # IterationLog, TrajectoryLog, serialization
├── collection_protocol.py # Physical collection guides
└── comparison.py          # Checkpoint Pareto analysis
```

**Key implementation details:**
- IterationLog captures: state (DiagnosticState), action (ExperimentProposal), outcome (metric deltas + behavioral eval), human decision, guardrail events
- TrajectoryLog wraps all iterations + final state + termination reason
- Real-time progress events: `{"event": "sweep_complete", "axis": "background", "sensitivity": 0.42, "elapsed": "3m12s"}`
- GP posterior visualization logging: posterior mean/std at N=0,5,10,15,20,25,30 after each pilot update
- LLM full prompt logged for debugging unexpected proposals
- Per-stage timing (sweep, pilot, fine-tune, eval) for cost estimation
- Checkpoint comparison: table of all checkpoints × all metrics, best value highlighted per column, Pareto frontier identification
- Collection protocols: workspace bounds from detected positions, step-by-step setup instructions, validation criteria

### Phase 7: CLI Entry Point

```
prescribe/
└── run_prescribe.py       # CLI argument parsing, orchestration
```

**CLI interface:**
```bash
# Full run
smolvla-prescribe run --model X --dataset Y --device cuda --max-iterations 5

# With research agenda
smolvla-prescribe run --model X --dataset Y --program prescription_program.md

# Sweep only (no fine-tuning)
smolvla-prescribe sweep --model X --dataset Y

# Dry run (preview plan)
smolvla-prescribe run --model X --dataset Y --dry-run

# Resume interrupted run
smolvla-prescribe run --resume outputs/prescribe_run_X/

# Data-only mode
smolvla-prescribe run --model X --dataset Y --data-only
```

### Phase 8: Expansions (can parallelize)

```
prescribe/
├── dashboard/             # Web viewer (React + Vite, pattern from smolvla-inspect)
└── mcp_server.py          # MCP tools
```

**Web dashboard:** Consumes TrajectoryLog from disk. Interactive charts: GP posterior narrowing over pilots, improvement trajectory per axis, behavioral eval heatmap, sweep response curves. Reuse smolvla-inspect's web viewer architecture (FastAPI backend + React frontend).

**MCP server:** Tools: `run_sweep` (launches sweep subprocess), `get_prescription_report` (reads from disk), `list_trajectories` (lists completed runs), `propose_experiment` (calls LLM agent). Read-only except `run_sweep`.

---

## 5. Repository Structure (Final)

```
smolvla-prescribe/
├── prescribe/
│   ├── __init__.py
│   ├── models.py                  # 15 dataclasses
│   ├── sweep.py                   # Sensitivity sweeps + pairwise interaction
│   ├── pilot.py                   # GP fitting, acquisition function, adaptive protocol
│   ├── loop.py                    # Orchestrator, guardrails, --dry-run, --resume, signal handling
│   ├── experiment_agent.py        # LLM agent + greedy fallback
│   ├── finetune_wrapper.py        # LeRobot trainer integration (Phase 0 spike)
│   ├── verification.py            # Behavioral eval (global + sliced) + sanity check
│   ├── logging.py                 # IterationLog, TrajectoryLog, serialization
│   ├── collection_protocol.py     # Physical collection guides
│   ├── comparison.py              # Checkpoint Pareto analysis
│   ├── mcp_server.py              # MCP tools for AI assistant integration
│   └── utils.py                   # get_augmentable_axes(), axis safety helpers
├── dashboard/
│   ├── backend/                   # FastAPI (reads TrajectoryLog from disk)
│   └── frontend/                  # React + Vite + TypeScript (Recharts for GP curves)
├── configs/
│   ├── prescription_targets.yaml  # Sensitivity thresholds, eval thresholds
│   └── prescription_program.md    # Default research agenda template
├── textures/                      # Built-in texture library (wood, tile, fabric, etc.)
├── tests/
│   ├── unit/                      # Mock-based (business logic, fast)
│   │   ├── test_sweep.py
│   │   ├── test_pilot.py
│   │   ├── test_loop.py
│   │   ├── test_agent.py
│   │   ├── test_verification.py
│   │   ├── test_logging.py
│   │   └── test_models.py
│   ├── integration/               # Fixture-based (real tiny model, slower)
│   │   ├── test_finetune_wrapper.py
│   │   ├── test_gp_real_data.py
│   │   ├── test_full_loop.py
│   │   └── test_resume.py
│   ├── benchmark/                 # LLM vs. greedy comparison
│   │   └── test_llm_benchmark.py
│   └── fixtures/                  # Tiny model checkpoint + dataset
│       ├── tiny_checkpoint/
│       └── tiny_dataset/
├── run_prescribe.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## 6. Dependencies

```
# Core
smolvla-inspect>=0.1.0,<1.0.0     # Diagnostic pipeline, counterfactual primitives
scikit-learn>=1.3.0                # GaussianProcessRegressor
numpy>=1.24.0
torch>=2.0.0

# Fine-tuning
lerobot>=0.1.0                     # Training infrastructure (pin version!)

# LLM agent
anthropic>=0.25.0                  # Optional — Claude API
openai>=1.0.0                      # Optional — GPT API

# Web dashboard (optional)
fastapi>=0.100.0
uvicorn>=0.23.0

# Testing
pytest>=7.0.0
pytest-timeout>=2.1.0
```

---

## 7. Error Handling — Complete Rescue Map

### GP Failures

| Exception | Trigger | Rescue | User Sees |
|-----------|---------|--------|-----------|
| `LinAlgError` | Degenerate pilot data (all same N or metric) | Fall back to heuristic (15 episodes) | "GP fit failed — using heuristic estimate of 15 episodes" |
| `ConvergenceWarning` | sklearn optimizer warning | Check posterior variance; if >10x metric range, fall back to heuristic | "GP uncertainty is very high — using heuristic estimate" |
| NaN variance | Extrapolation beyond observed range | Clamp predictions to valid range | Warning: "GP prediction clamped at N={n}" |

### LLM Failures

| Exception | Trigger | Rescue | User Sees |
|-----------|---------|--------|-----------|
| `TimeoutError` | API timeout | Retry 1x, then greedy fallback | "LLM timed out. Falling back to greedy mode." |
| `RateLimitError` | API rate limit (429) | Backoff + retry 2x | "Rate limited. Retrying in {N}s..." |
| `JSONParseError` | Malformed response | Retry with "respond in JSON" hint | Warning: "LLM response malformed. Retrying..." |
| `RefusalError` | Content filter | Greedy fallback | "LLM refused. Falling back to greedy mode." |
| `InvalidProposalError` | Proposes invalid axis | Retry with clarification | Warning: "LLM proposed invalid axis. Retrying..." |

### Fine-Tune Failures

| Exception | Trigger | Rescue | User Sees |
|-----------|---------|--------|-----------|
| `TrainingDivergenceError` | Loss → NaN | Rollback to previous checkpoint | "Training diverged. Rolling back." |
| `RuntimeError(CUDA OOM)` | GPU out of memory | Retry on CPU if possible | "CUDA OOM. Trying CPU fallback..." |
| `IOError` (checkpoint) | Disk full | Pre-run disk check; catch with message | "Checkpoint save failed. Free space: {X}GB." |

### Data/Eval Failures

| Exception | Trigger | Rescue | User Sees |
|-----------|---------|--------|-----------|
| `NoEvalDataError` | No held-out episodes | Pre-run validation (before loop starts) | "No held-out episodes found. Configure --eval-episodes." |
| `EmptySliceWarning` | Slice matches 0 episodes | Skip slice + warning | "Eval slice '{name}' matched 0 episodes. Skipping." |
| `SegmentationError` | SAM/OWL-ViT fails on an episode | Skip episode + warning | "Segmentation failed for episode X. Skipping." |
| All episodes fail segmentation | >80% episodes fail | Abort with message | "Segmentation failed on {N}/{total} episodes. Check source data." |

### System Failures

| Exception | Trigger | Rescue | User Sees |
|-----------|---------|--------|-----------|
| SIGINT (Ctrl+C) | User interrupts | Wait for current step → save state → exit | "Interrupted. State saved to interrupted_state.json." |
| SIGTERM | Terminal close | Same as SIGINT | (graceful shutdown, state saved) |
| Concurrent run | Lock file exists | Check PID; if stale, remove; if active, error | "Another run is active on this checkpoint. Use --force to override." |
| Human checkpoint timeout | User doesn't respond for 30 min | Auto-skip with warning | "Checkpoint timed out. Skipping this proposal." |

---

## 8. Behavioral Eval — Anti-Leakage Protocol

```
                    TRAINING POOL                    EVAL POOL
                  (augmentation)                 (held-out, never trained)
                ┌─────────────────┐           ┌─────────────────┐
                │ seed=42          │           │ seed=99          │
                │ texture_wood_01  │           │ texture_wood_07  │
                │ texture_tile_03  │           │ texture_tile_09  │
                │ brightness=-0.3  │           │ brightness=-0.35 │
                └────────┬────────┘           └────────┬────────┘
                         │                             │
                    Fine-tune on                  Evaluate on
                    these episodes               these episodes
                         │                             │
                         ▼                             ▼
                    Model learns              Slice-based MSE
                    from training pool        on eval pool
                                              (independent signal)
```

Both pools are from the same perturbation *family* but with different seeds/source images. Eval pool generated once at loop start, held constant across all iterations.

---

## 9. Autonomy Model

```
┌─────────────────────────────────────────────────────────┐
│              WHAT RUNS AUTOMATICALLY                     │
├─────────────────────────────────────────────────────────┤
│ Data augmentation on safe axes (background, lighting)    │
│ Regression rollback on critical threshold                │
│ Budget enforcement (stop on limit)                       │
│ Replay buffer enforcement (≥70%, non-negotiable)         │
│ Checkpoint save every iteration                          │
│ Sweep caching + invalidation                             │
│ Segmentation mask caching                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              WHAT REQUIRES HUMAN REVIEW                   │
├─────────────────────────────────────────────────────────┤
│ Task-dependent augmentation (color, clutter, language)    │
│ → LLM must justify safety; human reviews justification   │
│ → Partial identification bounds shown (best/worst case)  │
│ Continue after minor regression + next proposal is risky │
│ Accept augmentation ceiling (gap is within noise?)       │
│ Proceed to physical collection (timing decision)         │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Test Plan

### Strategy
- **Unit tests (mocks):** Business logic — approval gates, acquisition function, axis classification, sweep analysis, serialization. Fast (milliseconds).
- **Integration tests (fixtures):** Fine-tune wrapper, GP fitting, behavioral eval, full loop with tiny model. Real data, slower (~30s each).
- **Benchmark tests:** LLM agent vs. greedy on 3-5 standard scenarios. Validates that LLM adds value.

### Coverage Requirements (50+ paths)

**sweep.py (8 paths):**
- Single-axis sweep → valid SensitivitySweep
- NaN action → skip point + warning
- Zero variation points → EmptySweepError
- Perturbation fails → skip + warning
- Default safety classification
- User override in program.md
- Pairwise interaction detected
- No interaction (independent axes)

**pilot.py (8 paths):**
- GP fit on 2-4 normal observations
- GP LinAlgError → heuristic fallback
- GP ConvergenceWarning → check + fallback
- Single observation → wide posterior
- Acquisition: target reached → None
- Acquisition: ceiling reached → None
- Acquisition: budget exhausted → None
- Acquisition: normal → best candidate N

**loop.py (12 paths):**
- Safe axis auto-approve
- Task-dependent → human review
- Code change → human approve
- Critical regression → rollback + stop
- Minor regression → warn + continue
- Budget exceeded → stop
- No sensitive axes → stop
- `--dry-run` → sweeps + proposals, no fine-tune
- `--resume` → deserialize + continue
- SIGINT → save state + exit
- SIGTERM → save state + exit
- Concurrent run → lock file error

**experiment_agent.py (8 paths):**
- LLM → valid JSON → ExperimentProposal
- LLM → malformed JSON → retry → fallback
- LLM → refuses → fallback
- LLM → timeout → retry → fallback
- LLM → invalid axis → retry
- Greedy → max sensitivity safe axis
- Greedy → no safe axes → None
- Greedy → tie → deterministic selection

**verification.py (7 paths):**
- Global MSE computation
- Slice-based MSE per failure mode
- Empty slice → skip + warning
- Regression detected (global >10%)
- Regression detected (slice >15%)
- Sanity check passes (improvement real)
- Sanity check fails (improvement is noise)

**logging.py (4 paths):**
- Serialize IterationLog → JSON
- Serialize TrajectoryLog → JSON
- Deserialize for --resume
- Corrupted log → graceful error

**Other (5 paths):**
- Collection protocol from sweep + workspace bounds
- Axis with no workspace bounds → fallback
- DataPrescription serialization roundtrip
- PrescriptionReport construction
- DiagnosticState normalization

### LLM Benchmark Scenarios

| Scenario | What it tests | Metric |
|----------|--------------|--------|
| Background shortcut (known fix) | Does LLM pick textured backgrounds over solid colors? | Episode count efficiency |
| Multi-axis (background + lighting) | Does LLM sequence axes intelligently? | Total improvement + compute cost |
| Task-dependent (color w/ color task) | Does LLM correctly identify unsafe axis? | Safety justification quality |

---

## 11. Performance Optimizations

| Optimization | Where | Impact | Details |
|-------------|-------|--------|---------|
| Sweep caching | `loop.py` | ~5 min saved per cached axis | Cache unchanged sweep results; invalidate on detected cross-axis interaction |
| Segmentation mask caching | augmentation pipeline | ~5 min saved per iteration after first | Source episode masks don't change; cache and reuse for different augmentations |
| Interaction-aware invalidation | `sweep.py` + `loop.py` | Correct cache invalidation | Pairwise interaction detection tells us which cached sweeps are stale |

---

## 12. Security Considerations

| Concern | Mitigation |
|---------|-----------|
| Texture file path traversal | Validate user-provided texture files are valid images before processing |
| Checkpoint tampering | Checksum verification on checkpoint load (SHA-256) |
| LLM prompt injection via program.md | Guardrails prevent damage; user is attacking their own system (low risk for solo use) |
| Trajectory log PII | Opt-in sharing only; model IDs hashed; human notes stripped; no user identity collected |
| LLM API keys | Standard env var pattern. No hardcoded secrets. |

---

## 13. Observability

| Signal | What to log | Format |
|--------|-------------|--------|
| Progress events | `{"event": "sweep_complete", "axis": "...", "sensitivity": 0.42, "elapsed": "3m12s"}` | Structured JSON to stdout |
| GP posterior | Posterior mean/std at N=0,5,10,15,20,25,30 after each pilot | In IterationLog.gp_posterior |
| LLM prompts | Full prompt text sent to LLM | In IterationLog (debug only) |
| Per-stage timing | Wall-clock for sweep, pilot, fine-tune, eval | In IterationLog |
| Guardrail events | Which guardrail triggered, what action taken | In IterationLog.guardrail_triggered |

---

## 14. Existing Code Reused from smolvla-inspect

| Component | Import path | How it's used |
|-----------|-------------|---------------|
| `DiagnosticReport` | `smolvla_inspect.diagnostic.models` | Input to research loop |
| `DiagnosticMatrix` | `smolvla_inspect.diagnostic.models` | Attribution mass per region |
| `DatasetDiversityReport` | `smolvla_inspect.diagnostic.models` | Current dataset stats for collection protocols |
| `background_substitution` | `smolvla_inspect.diagnostic.counterfactual` | Called in parameterized sweep loops |
| `object_relocation` | `smolvla_inspect.diagnostic.counterfactual` | Position sweep |
| `lighting_perturbation` | `smolvla_inspect.diagnostic.counterfactual` | Lighting sweep |
| `object_recolor` | `smolvla_inspect.diagnostic.counterfactual` | Color sweep |
| `distractor_insertion` | `smolvla_inspect.diagnostic.counterfactual` | Clutter sweep |
| `task_string_swap` | `smolvla_inspect.diagnostic.counterfactual` | Language sweep |
| `compare_diagnostic_runs` | `smolvla_inspect.diagnostic.comparison` | Regression detection between iterations |
| `run_diagnostic` | `smolvla_inspect.diagnostic` | Full re-diagnostic after each iteration |

**One change needed in smolvla-inspect:** extend `background_substitution` to accept texture image paths in addition to `"gray"` / `"noise"` / `"blur"`. Backward-compatible.

---

## 15. Open Questions (from spec — to resolve during implementation)

| # | Question | Current best answer |
|---|----------|-------------------|
| 1 | Texture library: ship built-in + user folder? | **Resolved:** Both. Built-in textures in `textures/` dir + `--texture-dir` flag. |
| 2 | Fine-tune: LeRobot wrapper or standalone? | **Resolved:** LeRobot wrapper. Validated by Phase 0 spike. |
| 3 | Action validity for limited position augmentation? | Defer to post-v1a. Risky. |
| 4 | Multi-episode sweeps? | Start with single-episode. Add multi-episode as follow-up if needed. |
| 5 | Pilot independence (fresh checkpoint each time)? | Yes — independent measurements, important for GP calibration. |
| 6 | GP kernel selection? | **Resolved:** Matern 5/2 + negative linear mean. Auto-detect changepoint only if >6 pilots. |
| 7 | Physical collection: ask for workspace params? | Start with inferred bounds from detected positions. Add explicit workspace config as follow-up. |
| 8 | LLM proposal parsing? | **Resolved:** Structured JSON via system prompt (option a). |
| 9 | LLM cost management? | Support local models via Ollama/vLLM (same pattern as diagnostic agent). |
| 10 | Program.md versioning? | Re-read each iteration (more flexible). |
| 11 | LLM proposal guardrails? | Validate proposals against available primitives before presenting to user. |
| 12 | Outcome scoring weights? | Start with 2x regression penalty. Tune from user checkpoint decisions over time. |

---

## 16. Review Trail

| Review | Date | Status | Key Findings |
|--------|------|--------|-------------|
| CEO Review (`/plan-ceo-review`) | 2026-03-23 | CLEAR | SELECTIVE EXPANSION. 6 expansions accepted. 20 amendments (error handling, edge cases, security, observability, performance). |
| Outside Voice (Claude subagent) | 2026-03-23 | 10 findings | 2 acted on: placebo → sanity check, LLM benchmark added. Rest: philosophical scope disagreements or already addressed. |
| Eng Review (`/plan-eng-review`) | 2026-03-23 | CLEAR | Phase 0 spike, --resume, seg mask caching, DRY utility, undefined var fix, dual-layer test strategy. |

**Artifacts:**
- CEO plan: `~/.gstack/projects/ddl-subir-m-smolvla-inspect/ceo-plans/2026-03-23-prescription-engine.md`
- Test plan: `~/.gstack/projects/smolvla-inspect/subirmansukhani-main-eng-review-test-plan-20260323.md`
- Review log: `~/.gstack/projects/ddl-subir-m-smolvla-inspect/main-reviews.jsonl`
- Full spec: `docs/prescription-engine-spec.md`
