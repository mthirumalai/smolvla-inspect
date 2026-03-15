# AGENTS.md — Guide for Coding Assistants

> Primary reference for any AI coding assistant working on smolvla-inspect.
> For user-facing docs, see [README.md](README.md). For high-level explanation, see [docs/ELI5.md](docs/ELI5.md).

## What This Project Does

SmolVLA-Inspect is a diagnostic toolkit for **vision-language-action (VLA) robot policies**. It extracts attention maps, gradient attribution, and runs an AI-driven diagnostic agent that answers: *"Why does this robot policy fail, and what should I fix first?"*

The diagnostic agent runs a 9-phase pipeline: scene understanding, dataset diversity analysis, signal extraction, diagnostic matrix construction, representation probes, LLM hypothesis formation, counterfactual verification, spatial-vs-object diagnosis, and report synthesis.

## Architecture Overview

```
inspect_attention.py          ← thin entry point
smolvla_inspect/
├── __init__.py               ← main() dispatcher: CLI | serve | diagnose | compare
├── cli.py                    ← inspection pipeline (attention, gradient, export)
├── serve.py                  ← web viewer launcher
├── capture.py                ← attention hook capture
├── gradient.py               ← saliency, GradCAM attribution
├── heatmap.py                ← attention → heatmap conversion
├── viz.py                    ← visualization grids and overlays
├── export.py                 ← structured data export (npz + JSON)
├── data.py                   ← dataset helpers, batch building
├── internals.py              ← model health report (WeightWatcher)
└── diagnostic/               ← THE DIAGNOSTIC AGENT
    ├── __init__.py            ← run_diagnostic() entry, imports all modules
    ├── agent.py               ← DiagnosticAgent class + 8 signal runners
    ├── registry.py            ← all 5 registries + hypothesis templates
    ├── models.py              ← 25+ dataclasses (DiagnosticReport, Symptom, Finding, etc.)
    ├── matrix.py              ← diagnostic matrix builder + 13 symptom detectors
    ├── scene.py               ← OWL-ViT detection, SAM segmentation
    ├── counterfactual.py      ← 8 perturbation test primitives
    ├── prompts.py             ← LLM prompt templates (auto-generated from registry)
    ├── semantic_probe.py      ← patch↔text cosine similarity probe
    ├── spatial_object.py      ← spatial vs object grounding diagnosis
    ├── regions.py             ← region-wise attribution helpers
    ├── occlusion.py           ← occlusion sensitivity
    ├── temporal.py            ← temporal attention trajectories
    ├── comparison.py          ← cross-run comparison
    ├── report.py              ← report serialization
    └── diagnostic_cli.py      ← diagnose + compare subcommands

web/
├── backend/                   ← FastAPI (port 8080)
│   ├── main.py                ← create_app(settings)
│   ├── routers/               ← runs, visualizations, diagnostic, llm, compare
│   └── services/              ← data_loader, run_scanner, llm_service
└── frontend/                  ← React + Vite + TypeScript (port 5173 dev)
    ├── src/components/        ← ~25 components (HeatmapCanvas, DiagnosticPanel, etc.)
    ├── src/stores/            ← Zustand state (appStore.ts)
    └── src/services/          ← API client (api.ts)

configs/
├── defaults.yaml              ← default CLI flags for inspection pipeline
├── diagnostic.yaml            ← diagnostic agent config (thresholds, models, budgets)
├── diagnostic_quick.yaml      ← fast variant for iteration
└── gpu.yaml                   ← GPU-specific defaults
```

## The Registry Pattern (How to Extend)

The diagnostic system uses **5 pluggable registries**. Adding a new capability is typically 1 decorated function + 0 plumbing. All registries live in `smolvla_inspect/diagnostic/registry.py`.

### 1. Primitives — `REGISTRY`

Registered functions that the agent can call. Categories: `scene`, `counterfactual`, `composite`, `dataset`.

```python
from smolvla_inspect.diagnostic.registry import register_primitive

@register_primitive(
    "counterfactual.my_test",
    category="counterfactual",
    cost="expensive",
    requires_gpu=True,
    requires_model=True,
    description="What it does.",
    prompt_description="For LLM: when/why to use this test.",
    param_schema='target_object: str, intensity: float (0-1)',
)
def my_test(policy, sample, dataset, image_key, device, target_object, intensity, **kw):
    # Return a CounterfactualResult
    ...
```

The `prompt_description` and `param_schema` are injected into the LLM hypothesis prompt automatically via `format_counterfactual_prompt_section()`. The LLM sees available tests with their schemas and chooses which to run.

**Lookup:** `get_primitive(name)`, `list_primitives(category=None)`

Currently registered: 18 primitives (4 scene, 8 counterfactual, 4 regions, 1 occlusion, 1 temporal).

### 2. Signals — `SIGNAL_REGISTRY`

Expensive analysis functions the LLM can selectively enable (budget: max 4-5 per run).

```python
from smolvla_inspect.diagnostic.registry import register_signal

@register_signal(
    "my_signal",
    cost_description="~30s/frame",
    prompt_description="What it measures and when it's useful.",
    requires_model=True,
)
def _signal_my_signal(agent, sample, signals, scene=None):
    signals["my_signal"] = computed_heatmap  # store result in signals dict
```

Currently registered: 8 signals in `agent.py` (gradcam_siglip, saliency, vision_vs_state, per_action_dim_gradcam, connector_analysis, occlusion_sensitivity, temporal_trajectory, language_diff).

### 3. Symptom Detectors — `SYMPTOM_DETECTORS`

Functions that check the diagnostic matrix for known failure patterns.

```python
from smolvla_inspect.diagnostic.matrix import register_symptom_detector
from smolvla_inspect.diagnostic.models import Symptom

@register_symptom_detector
def detect_my_issue(matrix, scene=None):
    if matrix.scalars.get("my_metric", 0) > threshold:
        return Symptom(type="my_issue", severity="warning",
                       description="...", evidence={"my_metric": value})
    return None
```

Currently registered: 13 detectors in `matrix.py`.

### 4. Hypothesis Templates — `HYPOTHESIS_TEMPLATES`

Rule-based fallback (no LLM needed). Maps symptom type → counterfactual test.

```python
from smolvla_inspect.diagnostic.registry import register_hypothesis_template

register_hypothesis_template(
    "my_issue",
    description_template="Model fails because of {target}",
    confidence=0.7,
    test_type="background_substitution",
    test_params_template={"replacement": "gray"},
    expected_if_true="Action delta > 0.05",
    expected_if_false="Action delta < 0.01",
    confirms_on_change=True,
)
```

`{target}` placeholders are resolved at runtime to the detected target object name.

Currently registered: 9 templates in `registry.py`.

### 5. LLM Providers — `LLM_PROVIDERS`

```python
from smolvla_inspect.diagnostic.registry import register_llm_provider

@register_llm_provider("my_provider")
async def my_provider(agent, prompt, model, api_key, base_url):
    # async generator yielding string tokens
    yield "token"
```

### Adding a Complete New Diagnostic Capability

To add (for example) a "texture bias" detector:

1. **Symptom detector** in `matrix.py` — `@register_symptom_detector` that checks attribution patterns
2. **Hypothesis template** in `registry.py` — `register_hypothesis_template("texture_bias", ...)` mapping to a counterfactual
3. **Counterfactual primitive** in `counterfactual.py` — `@register_primitive("counterfactual.texture_scramble", ...)` that scrambles texture and measures action change

The LLM prompt, matrix, report, and comparison images all update automatically. Three registrations, zero plumbing.

## Key Data Model

All dataclasses in `smolvla_inspect/diagnostic/models.py`. The root entity is `DiagnosticReport`:

```
DiagnosticReport (serializable root — to_json() / from_dict())
├── scene: SceneSegmentation
│   └── objects: list[DetectedObject]      (label, box, score, mask)
├── dataset_diversity: DatasetDiversityReport?
├── matrix: DiagnosticMatrix               (signal × region → attribution mass)
│   ├── temporal_trajectories: list[TemporalTrajectory]
│   ├── occlusion: OcclusionMap?
│   └── connector_analysis: ConnectorAnalysis?
├── semantic_probe: SemanticProbeReport?
│   └── frames: list[SemanticFrameSummary]
├── qk_probe: QKProbeReport?
│   └── top_heads: list[QKHeadSummary]
├── spatial_object_diagnosis: SpatialObjectDiagnosis?
│   └── spatial_evidence / object_evidence: list[DiagnosticEvidence]
├── symptoms: list[Symptom]                (type, severity, description, evidence)
├── hypotheses: list[Hypothesis]           (test_type, test_params, confirms_on_change)
├── counterfactual_results: list[CounterfactualResult]  (action_delta_l2, confirmed)
├── findings: list[Finding]                (severity, title, fix, evidence_refs)
└── llm_synthesis: str                     (narrative summary)
```

Every `Finding.evidence_refs` traces back: Finding → Hypothesis → Symptom → matrix cell.

## SmolVLA Model Architecture (Need-to-Know)

Understanding these facts is critical when modifying attention capture or gradient code:

| Fact | Detail |
|------|--------|
| Vision encoder | SigLIP: 512×512 input, 16×16 patches → 32×32 = 1024 patches, 12 heads, 12 layers, **no CLS token** |
| Connector | Pixel-shuffle: 1024 patches → 64 tokens (93.75% compression) |
| VLM | 16 layers, 15 heads — fuses vision + language |
| Action expert | 16 layers, 8 heads — predicts robot actions |
| Cross-attention | Implicit: VLM builds KV cache from prefix, expert queries it. Detected by Q seq_len ≠ K seq_len |

**Critical gotcha:** SmolVLA **bypasses `self_attn.forward()` entirely**. The method `forward_attn_layer()` manually calls q/k/v projections, concatenates VLM+Expert tensors, then calls `eager_attention_forward()`. Hooks on `self_attn` will never fire — you must hook or patch `eager_attention_forward` instead.

## Gradient Attribution Constraints

These will trip you up if you don't know them:

1. **`select_action()` has `@torch.no_grad()`** — blocks all gradient flow. Bypass by calling `prepare_images()`, `prepare_state()`, `model.sample_actions()` directly under `torch.enable_grad()`.

2. **`eager_attention_forward` mask dtype** — under autograd, `attention_mask` may be `Long` not `bool`. Must cast with `.bool()` before `torch.where`. The gradient module monkey-patches this.

3. **SigLIP activations are BFloat16** — must `.float()` before `.numpy()` for GradCAM.

4. **Action caching** — `select_action()` caches in `_queues[ACTION]`. Call `policy.reset()` between frames to force a new forward pass.

5. **Backprop target** — `actions[:, 0, :].sum()` (immediate next action step).

6. **Memory** — ~176 transformer layer passes stored per frame (prefill + 10 denoising × 16 layers). GPU strongly recommended.

7. **Vision token position** — `add_image_special_tokens` defaults to False, so vision tokens start at position 0 in the prefix sequence.

## Entry Points

| Command | Entry | What it does |
|---------|-------|-------------|
| `python inspect_attention.py` | `cli.main()` | Full inspection pipeline (attention + gradient + export) |
| `python inspect_attention.py serve` | `serve.serve_main()` | Launch web viewer on pre-computed runs |
| `python inspect_attention.py diagnose` | `diagnostic_cli.diagnose_main()` | Run diagnostic agent (integrated or post-hoc) |
| `python inspect_attention.py compare` | `diagnostic_cli.compare_main()` | Compare multiple diagnostic runs |

### Diagnose modes

- **Integrated:** `--model X --dataset Y` — loads model, runs inspection + diagnostic
- **Post-hoc:** `--run-dir ./outputs/run_folder --dataset Y` — analyzes existing run data, optionally loads model for counterfactuals

## Config Reference

**`configs/defaults.yaml`** — inspection pipeline defaults (model, dataset, attention method, gradient options, export settings).

**`configs/diagnostic.yaml`** — diagnostic agent settings:
- `scene_model` — OWL-ViT v2 model ID (default: `google/owlv2-base-patch16-ensemble`)
- `segmentation_model` — SAM model ID (default: `facebook/sam-vit-base`)
- `detection_confidence` — detection threshold (default: 0.1)
- `max_hypotheses` — max LLM hypotheses (default: 5)
- `max_counterfactuals` — total test budget (default: 7)
- `max_expensive_signals` — LLM signal triage budget (default: 5)
- `mandatory_counterfactual_tests` — always-run tests for cross-run comparability: `background_substitution`, `object_relocation`, `task_string_swap`, `occlusion_targeted`

## Web Viewer

- **Backend:** FastAPI at `web/backend/main.py` — reads pre-computed results from disk
- **Frontend:** React + Vite at `web/frontend/` — interactive heatmaps (canvas), charts (Recharts), LLM streaming (SSE)
- **Dev:** `npm run dev` in `web/frontend/` (port 5173, proxies `/api` to backend at 8080)
- **Prod:** `npm run build` → `dist/` served by FastAPI as static files
- **Launch:** `python inspect_attention.py serve --port 8080 --base-dir ./outputs`

## Output Structure

Each run produces:
```
outputs/run_YYYY-MM-DD_HH-MM-SS/
├── run_manifest.json          ← metadata + available visualizations
├── frames/                    ← per-frame heatmap PNGs
├── data/                      ← structured .npz files
├── images/                    ← original camera frames
├── grids/                     ← multi-panel visualization grids
└── diagnostic/                ← diagnostic agent output
    ├── report.json            ← structured DiagnosticReport
    ├── report.md              ← human-readable narrative
    ├── matrix.json            ← attribution mass matrix
    ├── scene/                 ← detection + segmentation results
    ├── counterfactuals/       ← perturbation test images + results
    └── evidence_chain.json    ← full evidence log
```

## Conventions

- **Terminology:** "symptom" (not "anomaly") for detected failure patterns. Recently renamed — legacy JSON may use "anomaly" keys; `from_dict()` handles both.
- **Imports:** All `@register_primitive` modules must be imported in `diagnostic/__init__.py` to trigger registration at load time.
- **LLM prompts:** Auto-generated from registry metadata (`prompt_description`, `param_schema`). Don't hardcode test lists in prompts.
- **Serialization:** `DiagnosticReport.to_json()` / `DiagnosticReport.from_dict()` is the canonical serialization boundary. All fields must be JSON-serializable (numpy arrays → lists).
