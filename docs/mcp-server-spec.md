# SmolVLA-Inspect MCP Server — Comprehensive Spec

## Overview

An MCP (Model Context Protocol) server that exposes the full smolvla-inspect diagnostic pipeline as tools callable by any MCP client (Claude Code, Claude Desktop, custom agents). The server enables natural-language experiment management, cross-product comparisons, and autonomous diagnostic workflows.

**Target query examples:**
- "Compare model v2 vs v3 across the bridge and aloha datasets"
- "Run diagnostics on the latest 3 checkpoints and tell me which one has the best object grounding"
- "What symptoms appeared after fine-tuning on episode 5?"
- "Show me the attention difference between the base model and the 50-epoch checkpoint on the pick-and-place task"
- "Which heads in layer 8 are spatial vs semantic?"
- "Is the connector losing information about the target object?"
- "Regenerate the comparison report for the last two fine-tuning runs"

---

## 1. Run Index & Discovery

### 1.1 Run Index Store

The current system discovers runs by walking a directory tree on every request. The MCP server needs a persistent, queryable index.

**What to build:**
- A lightweight SQLite database (or JSON index file) at `{base_dir}/.smolvla_index.db`
- Populated by scanning manifests (`run_manifest.json`) from all run directories
- Auto-refreshes on startup and exposes a manual rescan tool
- Handles legacy runs (pre-manifest) via `_build_legacy_manifest()` heuristic

**Schema (indexed fields):**

| Column | Type | Source |
|--------|------|--------|
| `run_id` | TEXT PK | SHA256(path)[:12] (matches existing ID scheme) |
| `run_name` | TEXT | Directory basename |
| `run_dir` | TEXT | Absolute path |
| `created_at` | TIMESTAMP | manifest.created_at |
| `model_id` | TEXT | manifest.model_info.model_id |
| `dataset_id` | TEXT | manifest.dataset_info.dataset_id |
| `episode_idx` | INT | manifest.dataset_info.episode_idx |
| `task_string` | TEXT | manifest.dataset_info.task_string |
| `num_frames` | INT | manifest.dataset_info.num_frames |
| `method` | TEXT | manifest.cli_args.method |
| `has_gradient` | BOOL | available_visualizations.saliency or gradcam_siglip |
| `has_cross_attention` | BOOL | available_visualizations.cross_attention |
| `has_diagnostic` | BOOL | diagnostic/report.json exists |
| `has_model_health` | BOOL | available_visualizations.model_internals |
| `has_semantic_probe` | BOOL | diagnostic report contains semantic_probe |
| `has_qk_probe` | BOOL | diagnostic report contains qk_probe |
| `tags` | TEXT (JSON array) | User-supplied tags (new field in manifest) |
| `notes` | TEXT | User notes (from run_notes.json sidecar) |
| `is_legacy` | BOOL | Pre-manifest run |

**Tagging system:**
- Runs can be tagged with arbitrary strings (e.g., `["baseline", "v2", "experiment-A"]`)
- Tags stored in `run_manifest.json` under a new `tags` field
- Supports add/remove operations via MCP tool

### 1.2 MCP Tools — Discovery

| Tool | Parameters | Returns |
|------|-----------|---------|
| `list_runs` | `model_id?: str`, `dataset_id?: str`, `tag?: str`, `has_diagnostic?: bool`, `created_after?: str`, `created_before?: str`, `limit?: int`, `sort_by?: "created_at"\|"model_id"\|"dataset_id"` | `RunSummary[]` |
| `search_runs` | `query: str` (free-text search across model_id, dataset_id, task_string, tags, run_name, notes) | `RunSummary[]` |
| `get_run_detail` | `run_id: str` | Full manifest + diagnostic availability + notes |
| `tag_run` | `run_id: str`, `tags: str[]` | Updated tags |
| `untag_run` | `run_id: str`, `tags: str[]` | Updated tags |
| `rescan_runs` | `base_dir?: str` | Count of new/updated runs |
| `validate_run` | `run_id: str` | Data integrity check — which files exist, which are missing or corrupt |

---

## 2. Single-Run Inspection

### 2.1 Visualization Data Access

Expose the existing data loader functions as MCP tools. The LLM client can request specific visualization data and reason over it.

| Tool | Parameters | Returns |
|------|-----------|---------|
| `get_attention_heatmaps` | `run_id: str`, `type: "self"\|"cross"`, `frame_indices?: int[]` | Heatmap arrays (patch-level, float[][]) + frame metadata |
| `get_gradient_attribution` | `run_id: str`, `type: "saliency"\|"gradcam_siglip"\|"gradcam_connector"`, `frame_indices?: int[]` | Heatmap arrays |
| `get_per_head_attention` | `run_id: str`, `frame_idx?: int` | Per-head heatmaps + entropy values |
| `get_per_step_cross_attention` | `run_id: str`, `frame_idx?: int` | Per-denoising-step cross-attention maps (shows how attention evolves during action generation) |
| `get_vlm_layer_gradcam` | `run_id: str`, `frame_indices?: int[]` | Per-VLM-layer GradCAM — vision and language modality attribution at each layer |
| `get_per_action_dim` | `run_id: str`, `frame_indices?: int[]` | Per action dimension GradCAM maps (x, y, z, roll, pitch, yaw, gripper) + magnitude values |
| `get_language_diff` | `run_id: str`, `frame_indices?: int[]` | Language sensitivity — GradCAM diff between original vs alternative task string per frame |
| `get_vision_vs_state` | `run_id: str` | Per-frame vision/state contribution ratios (vision_norm, state_norm, vision_share) |
| `get_model_internals` | `run_id: str`, `component?: str` | WeightWatcher alphas, entropy, redundancy — optionally filtered to a specific component (siglip, vlm, expert) |
| `get_frame_image` | `run_id: str`, `frame_idx: int` | Base64-encoded PNG of original frame |
| `get_heatmap_image` | `run_id: str`, `viz_type: str`, `frame_idx: int` | Base64-encoded heatmap overlay PNG |
| `list_run_images` | `run_id: str` | All available image paths in the run (dashboards, per-frame overlays, grids) |

### 2.2 Statistics

| Tool | Parameters | Returns |
|------|-----------|---------|
| `get_viz_stats` | `run_id: str`, `viz_type: str` | Per-frame + aggregate stats (entropy_ratio, coverage_pct, centroid, gini, max/mean_intensity, temporal_stability, centroid_drift) |
| `get_run_summary_stats` | `run_id: str` | Aggregated stats across all available viz types |
| `format_stats_for_prompt` | `run_id: str`, `viz_type: str` | Pre-formatted text summary of stats suitable for LLM consumption |

---

## 3. Diagnostic Agent

### 3.1 Running Diagnostics

The diagnostic agent is the most valuable capability to expose. It runs a 9-phase pipeline:

1. **Scene Understanding** — OWL-ViT object detection + SAM segmentation
2. **Dataset Diversity Analysis** — object position variance, background diversity, task string diversity
3. **Signal Triage** — cheap signals always run; LLM selects which expensive signals to add (budget-aware)
4. **Diagnostic Matrix** — attribution mass per (signal × region) with 13 symptom detectors
5. **Representation Probes** — semantic probe (text↔patch similarity) + QK probe (per-head decomposition)
6. **Hypothesis Generation** — LLM-generated or rule-based hypotheses from symptoms
7. **Counterfactual Testing** — causal interventions to confirm/deny hypotheses
8. **Spatial vs Object Diagnosis** — separates positional prior bias from genuine object grounding
9. **LLM Synthesis** — compiles findings into narrative with severity-ranked recommendations

| Tool | Parameters | Returns |
|------|-----------|---------|
| `run_diagnostic` | `run_id: str`, `max_counterfactuals?: int`, `skip_counterfactuals?: bool`, `max_hypotheses?: int`, `max_expensive_signals?: int`, `max_hypothesis_iterations?: int`, `config_path?: str` (e.g. "configs/diagnostic_quick.yaml" for fast iteration) | `DiagnosticReport` (full structured report) |
| `run_diagnostic_on_model` | `model_id: str`, `dataset_id: str`, `episode?: int`, `num_frames?: int`, `config_path?: str`, `run_name?: str`, `tags?: str[]` | Runs full pipeline (inspection + diagnostic), returns `{run_id, report}` |
| `run_diagnostic_phase` | `run_id: str`, `phase: str` | Run a single phase in isolation. Phases: `"scene"`, `"matrix"`, `"probes"`, `"hypothesize"`, `"test"`, `"synthesize"`. Returns phase-specific structured result |
| `get_diagnostic_report` | `run_id: str` | Saved `DiagnosticReport` if exists |
| `get_diagnostic_matrix` | `run_id: str`, `signals?: str[]` | Attribution matrix (signal × region → mass), optionally filtered to specific signals |
| `get_scene_data` | `run_id: str` | Detected objects, segmentation regions, annotated frame (base64 PNG) |
| `get_symptoms` | `run_id: str` | `Symptom[]` with type, severity (critical/warning/info), description, evidence |
| `get_hypotheses` | `run_id: str` | `Hypothesis[]` with confidence, supporting_symptoms, test plan, expected outcomes |
| `get_counterfactual_result` | `run_id: str`, `test_name: str` | `CounterfactualResult` with action_delta_l2, per-dim deltas, gradcam_shift, confirmed status, comparison image |
| `get_findings` | `run_id: str` | `Finding[]` — severity, title, observation, test_description, test_result, interpretation, fix suggestion, expected_impact, evidence_refs |
| `get_evidence_chain` | `run_id: str` | Ordered list of `EvidenceEntry` — the full audit trail of the diagnostic agent's reasoning (phase, type, data, timestamp) |

### 3.2 Representation Probes

These are critical for understanding *why* the model attends where it does — spatial memorisation vs genuine object recognition.

| Tool | Parameters | Returns |
|------|-----------|---------|
| `get_semantic_probe` | `run_id: str` | `SemanticProbeReport` — per-frame text↔image similarity for target object vs candidates. Reveals whether the model's spatial focus aligns with actual object recognition or just positional bias |
| `get_qk_probe` | `run_id: str` | `QKProbeReport` — per-head decomposition into spatial vs semantic attention patterns. Shows which heads are positional-prior heads vs learned-association heads |
| `get_spatial_object_diagnosis` | `run_id: str` | `SpatialObjectDiagnosis` — verdict (spatial_prior / object_grounded / mixed / inconclusive), confidence, spatial_score, object_score, summary, key_metrics |
| `get_connector_analysis` | `run_id: str` | `ConnectorAnalysis` — pre/post connector attribution shares per region, information loss per region, total information loss. Shows what the pixel-shuffle bottleneck (1024→64 patches) drops |

### 3.3 Temporal & Spatial Analysis

| Tool | Parameters | Returns |
|------|-----------|---------|
| `get_temporal_trajectory` | `run_id: str`, `signal_type?: str` | `TemporalTrajectory` — attention centroid path across frames, smoothness metric, object-tracking correlation. Reveals if attention is stable or erratically jumping |
| `get_occlusion_map` | `run_id: str` | `OcclusionMap` — spatial sensitivity grid computed by sliding a gray patch. Ground-truth causal attribution (complementary to gradient methods) |
| `run_occlusion_analysis` | `run_id: str`, `patch_size?: int`, `stride?: int` | Compute occlusion sensitivity from scratch (GPU, ~5min). Returns `OcclusionMap` |
| `compute_region_attribution` | `run_id: str`, `signal_type: str` | Per-region attribution shares from heatmap + segmentation (foreground_ratio, spatial_prior_ratio, per-object mass) |

### 3.4 Running Individual Primitives

For fine-grained control, expose individual diagnostic primitives and registry metadata.

| Tool | Parameters | Returns |
|------|-----------|---------|
| `list_primitives` | `category?: "scene"\|"model"\|"counterfactual"\|"composite"\|"dataset"` | `PrimitiveSpec[]` (name, cost, description, GPU requirements, param_schema) |
| `get_primitive_spec` | `name: str` | Full spec for a single primitive including param_schema and prompt_description |
| `run_counterfactual` | `run_id: str`, `test_type: str`, `params?: dict` | `CounterfactualResult` |
| `get_counterfactual_param_schema` | `test_type: str` | JSON schema of parameters for a specific test type |
| `list_counterfactual_types` | — | Available test types with descriptions, parameter schemas, and cost estimates |
| `list_signals` | — | Available expensive signals with cost descriptions (e.g., "~30s/frame") and GPU requirements |
| `list_symptom_detectors` | — | All 13 registered symptom detectors with descriptions and trigger conditions |
| `list_hypothesis_templates` | — | All symptom→hypothesis rule-based mappings (symptom_type, test_type, expected outcomes, confirms_on_change) |

Available counterfactual test types:
- `background_substitution` — Replace background with gray/random, measure action delta. Params: `{replacement: "gray"|"random"}`
- `object_relocation` — Move object by pixels, measure action change. Params: `{target_object: str, shift_pixels: [int, int]}`
- `object_recolor` — Shift hue, measure appearance sensitivity. Params: `{target_object: str, hue_shift: float}`
- `lighting_perturbation` — Adjust brightness/contrast. Params: `{brightness: float, contrast: float}`
- `distractor_insertion` — Add novel object, measure distraction. Params: `{position: [int, int], distractor_size: int}`
- `task_string_swap` — Replace instruction text. Params: `{replacement_task: str}`
- `temporal_consistency` — Apply perturbation across N frames, measure consistency. Params: `{perturbation_type: str, num_frames: int}`
- `occlusion_targeted` — Occlude specific object with fill. Params: `{target_object: str, fill: "gray"|"blur"}`

Available expensive signals (8 registered):
- `gradcam_siglip` (~30s/frame) — GradCAM on SigLIP last encoder layer
- `saliency` (~30s/frame) — Input-pixel gradient saliency
- `vision_vs_state` (~30s) — Gradient ratio between vision and proprioceptive state
- `per_action_dim_gradcam` (~2min/frame) — Separate GradCAM per action dimension
- `connector_analysis` (~1min) — Pre/post connector attribution comparison
- `occlusion_sensitivity` (~5min) — Ground-truth causal map via sliding occlusion
- `temporal_trajectory` (~2min) — Attention centroid tracking across episode
- `language_diff` (~1min) — GradCAM under original vs alternative task string

---

## 4. Cross-Run Comparison

This is the core experiment management capability. Three levels of comparison exist.

### 4.1 Structured Diagnostic Comparison

Uses `smolvla_inspect/diagnostic/comparison.py` — compares attribution deltas, counterfactual deltas, weight alphas, symptoms, and generates verdicts + recommendations.

| Tool | Parameters | Returns |
|------|-----------|---------|
| `compare_diagnostic_runs` | `run_ids: str[]`, `labels?: str[]` | `ComparisonReport` with attribution_deltas, counterfactual_deltas, weight_alpha_deltas, symptom_summary, verdict, recommendations |
| `compare_run_configs` | `run_ids: str[]` | Dict of config fields that differ across runs (flattened from cli_args, model_info, dataset_info) |
| `compare_component_weights` | `run_ids: str[]`, `component?: str` | Per-component weight spectral alpha comparison (siglip, vlm, expert — with status labels: overcorrelated/healthy/undertrained/severely undertrained) |

**ComparisonReport data model:**
- `attribution_deltas: AttributionDelta[]` — per (signal, region): values across runs, absolute delta, pct_change
- `counterfactual_deltas: CounterfactualDelta[]` — per test_type: action_delta_l2 across runs, confirmed status
- `weight_alpha_deltas: WeightAlphaDelta[]` — per component: mean alpha across runs, status labels
- `scalar_deltas: dict[str, list[float]]` — scalars like foreground_ratio, spatial_prior_ratio
- `symptom_summary: dict[str, list[str]]` — label → symptom types present
- `verdict: str` — auto-generated natural-language verdict
- `recommendations: str[]` — actionable training recommendations

### 4.2 Visualization-Level Comparison

Uses `web/backend/services/comparison.py` — element-wise heatmap diffs and side-by-side data.

| Tool | Parameters | Returns |
|------|-----------|---------|
| `compare_heatmaps` | `run_ids: str[]`, `viz_types: str[]`, `frame_indices?: int[]` | Per-run heatmap data + element-wise diff arrays |

### 4.3 Cross-Product Comparison (New)

This is the key new capability for experiment management. The LLM can orchestrate this by calling `list_runs` + `compare_diagnostic_runs` in a loop, but a dedicated tool makes it atomic and efficient.

| Tool | Parameters | Returns |
|------|-----------|---------|
| `compare_matrix` | `model_ids: str[]`, `dataset_ids: str[]`, `metric?: str`, `episode?: int` | Matrix of comparison results: `{(model, dataset) → run_id, metric_value}` + cross-product summary |

**`metric` options:**
- `"symptom_count"` — number of detected symptoms
- `"background_attribution"` — background attribution mass (lower is better)
- `"object_grounding"` — target object attribution mass (higher is better)
- `"weight_alpha"` — mean expert weight alpha (2-4 is healthy)
- `"counterfactual_sensitivity"` — mean action delta across counterfactuals
- `"spatial_vs_object"` — spatial_object_diagnosis verdict (spatial_prior / object_grounded / mixed)
- `"overall_health"` — composite score derived from the above

**How it works:**
1. For each `(model_id, dataset_id)` pair, find the matching run(s) in the index
2. If multiple runs match, use the most recent
3. Extract the requested metric from each run's diagnostic report
4. Return a structured matrix + a natural-language summary

**Example output:**
```json
{
  "matrix": {
    "models": ["smolvla-base", "smolvla-v2", "smolvla-v3"],
    "datasets": ["bridge-v2", "aloha-static"],
    "cells": {
      "smolvla-base|bridge-v2": {"run_id": "abc123", "value": 0.82, "symptoms": ["high_background_attribution"]},
      "smolvla-v2|bridge-v2": {"run_id": "def456", "value": 0.45, "symptoms": []},
      ...
    }
  },
  "summary": "v3 shows the best object grounding across both datasets. v2 resolved background dependence on bridge but regressed on aloha."
}
```

### 4.4 CLI ↔ MCP Tool Equivalence

The existing CLI subcommand `smolvla-inspect compare` provides the same comparison functionality as the MCP tools:

```bash
smolvla-inspect compare <run_dir_1> <run_dir_2> [--labels L1 L2] [--output path]
```

**MCP equivalent:**
```
compare_diagnostic_runs(run_ids=["abc123", "def456"], labels=["v1", "v2"])
```

Both import the same underlying logic from `smolvla_inspect/diagnostic/comparison.py:compare_runs()`. The CLI parses command-line arguments and writes markdown + JSON output files; the MCP tool receives structured parameters and returns a `ComparisonReport` object.

---

## 5. Inspection Pipeline Execution

For running new analyses from scratch (not just reading existing results).

| Tool | Parameters | Returns |
|------|-----------|---------|
| `run_inspection` | `model_id: str`, `dataset_id: str`, `episode?: int`, `num_frames?: int`, `method?: "last-layer"\|"rollout"\|"all-layers"`, `cross_attention?: bool`, `gradient?: "saliency"\|"gradcam"\|"both"`, `model_health?: bool`, `show_heads?: bool`, `per_step_cross_attention?: bool`, `gradcam_vlm_layers?: str` (layer indices, e.g. "4,8,12,16"), `per_action_dim?: bool`, `language_diff?: bool`, `vision_vs_state?: bool`, `gradient_seed?: int`, `raw_attention?: bool`, `attn_threshold?: float`, `smooth_grad?: int`, `smooth_grad_sigma?: float`, `skip_attention?: bool`, `internals_frames?: int`, `config?: str` (path to YAML config profile, e.g. "configs/gpu.yaml"), `run_name?: str`, `tags?: str[]` | `{run_id: str, run_dir: str, manifest: dict}` |
| `add_run` | `run_dir: str` | Add an existing run directory to the index without rescanning everything. Returns `RunSummary` |
| `get_job_status` | `job_id: str` | Status of an in-progress or completed job (applies to any long-running tool) |
| `list_jobs` | `status?: "pending"\|"running"\|"completed"\|"failed"` | All tracked jobs with progress info |

This wraps the full `smolvla_inspect/cli.py` pipeline:
1. Load model + dataset
2. Extract attention heatmaps (self, cross, per-head, per-step)
3. Compute gradient attribution (saliency, GradCAM, per-action-dim, VLM layers, language diff)
4. Compute vision vs state contribution
5. Run model health analysis (WeightWatcher alpha, entropy, redundancy)
6. Export structured data (npz + JSON) + rendered images (PNG)
7. Register in run index

---

## 6. LLM Analysis & Configuration

Expose the existing LLM analysis capabilities for AI-assisted interpretation.

### 6.1 LLM Configuration

| Tool | Parameters | Returns |
|------|-----------|---------|
| `get_llm_config` | — | Current LLM settings: `{provider, model, base_url, has_api_key}` |
| `set_llm_config` | `provider?: "anthropic"\|"openai"`, `model?: str`, `api_key?: str`, `base_url?: str` | Updated config. Supports OpenAI-compatible servers (vLLM, Ollama, LM Studio) via base_url |

**Environment variable fallback chain:**
- `SMOLVLA_LLM_API_KEY` → `ANTHROPIC_API_KEY` (if provider=anthropic) → `OPENAI_API_KEY` (if provider=openai)

### 6.2 Analysis Tools

| Tool | Parameters | Returns |
|------|-----------|---------|
| `analyze_with_llm` | `run_id: str`, `analysis_type: str`, `custom_prompt?: str`, `include_images?: bool`, `include_stats?: bool`, `compare_run_ids?: str[]`, `run_notes?: str` | LLM analysis text (streamed if transport supports it) |
| `get_cached_analyses` | `run_id: str` | Previously saved LLM analyses keyed by type, with metadata (model, provider, timestamps) |
| `delete_cached_analysis` | `run_id: str`, `analysis_type: str` | Remove a specific cached analysis to force re-run |
| `list_analysis_types` | — | Available analysis types with descriptions and required context |
| `get_prompt_template` | `analysis_type: str` | Raw prompt template with variable placeholders — useful for previewing what the LLM will see |
| `render_prompt_preview` | `run_id: str`, `analysis_type: str` | Fully rendered prompt with all context filled in (for debugging/transparency) |

**Analysis types** (from prompt templates):
- `single_viz_self_attention` — Interpret self-attention patterns
- `single_viz_per_head` — Per-head attention analysis
- `single_viz_cross_attention` — Interpret action expert cross-attention
- `single_viz_saliency` — Interpret gradient saliency
- `single_viz_gradcam_siglip` — Interpret GradCAM on vision encoder
- `single_viz_vision_vs_state` — Interpret vision vs state contribution
- `single_viz_model_internals` — Interpret weight health metrics
- `model_health_summary` — Overall model health assessment
- `per_action_dim_analysis` — Per-dimension attribution analysis
- `language_conditioning_analysis` — Language grounding assessment
- `temporal_dynamics` — Temporal attention stability analysis
- `action_expert_cross_attention` — Cross-attention expert analysis
- `language_grounding_summary` — Language sensitivity summary
- `run_insights` — Overall run interpretation
- `multi_run_comparison` — Cross-run comparison narrative

---

## 7. Run Notes & Annotations

Persistent per-run notes for collaborative annotation and experiment tracking.

| Tool | Parameters | Returns |
|------|-----------|---------|
| `get_run_notes` | `run_id: str` | `{notes: str, updated_at: str}` — Markdown-formatted notes |
| `set_run_notes` | `run_id: str`, `notes: str` | `{notes: str, updated_at: str}` — Stored in `run_notes.json` sidecar file |

---

## 8. Dataset Analysis

| Tool | Parameters | Returns |
|------|-----------|---------|
| `get_dataset_diversity` | `run_id: str` | `DatasetDiversityReport` — object_position_stats (per-object std_x/std_y, mean position, sample_count), background_diversity_score, lighting_stats (mean/std brightness/contrast), task_string_diversity (unique_count, examples), num_episodes_sampled |
| `analyze_dataset` | `dataset_id: str`, `num_episodes?: int`, `frames_per_episode?: int` | Run standalone dataset diversity analysis without full inspection |
| `parse_task_objects` | `task_string: str` | Extract object names heuristically from a natural-language task instruction — useful for previewing what the diagnostic agent will focus on |

---

## 9. Report Generation

| Tool | Parameters | Returns |
|------|-----------|---------|
| `regenerate_report` | `run_id: str` | Regenerate markdown report from existing JSON data (no GPU needed). Useful after manual edits to report.json |
| `regenerate_comparison_report` | `run_ids: str[]`, `labels?: str[]` | Regenerate comparison markdown + verdict + recommendations from existing diagnostic data |
| `export_run_data` | `run_id: str`, `include?: str[]` | Export/re-export structured data. Include options: `["frames", "self_attention", "cross_attention", "gradient", "model_internals", "diagnostic"]`. Returns `{paths: str[], size_bytes: int}` |

---

## 10. Implementation Plan

### 10.1 Architecture

```
┌─────────────────────────────────────────────┐
│                MCP Client                    │
│  (Claude Code / Claude Desktop / custom)     │
└──────────────────┬──────────────────────────┘
                   │ MCP Protocol (stdio or SSE)
┌──────────────────▼──────────────────────────┐
│           smolvla-inspect MCP Server         │
│                                              │
│  ┌──────────────┐  ┌─────────────────────┐  │
│  │  Tool Layer   │  │  Run Index (SQLite) │  │
│  │  (75 tools)   │  └─────────────────────┘  │
│  └──────┬───────┘                            │
│         │         ┌──────────────────────┐   │
│         │         │  Job Queue (async)   │   │
│         │         │  max_concurrent_gpu=1│   │
│         │         └──────────────────────┘   │
│  ┌──────▼──────────────────────────────────┐ │
│  │        Service Layer (existing)          │ │
│  │  ┌────────┐ ┌──────────┐ ┌───────────┐ │ │
│  │  │run_scan│ │data_load │ │stats_eng  │ │ │
│  │  ├────────┤ ├──────────┤ ├───────────┤ │ │
│  │  │compare │ │diagnostic│ │llm_service│ │ │
│  │  ├────────┤ ├──────────┤ ├───────────┤ │ │
│  │  │export  │ │  agent   │ │prompt_tmpl│ │ │
│  │  └────────┘ └──────────┘ └───────────┘ │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │          Core Libraries (existing)       │ │
│  │  cli.py  gradient.py  viz.py  export.py │ │
│  └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

### 10.2 Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| MCP SDK | `mcp` Python SDK | Official MCP SDK, matches existing Python codebase |
| Transport | stdio (default) + optional SSE | stdio for Claude Code/Desktop, SSE for remote clients |
| Index | SQLite via `sqlite3` stdlib | Zero-dependency, file-based, perfect for local tool |
| Async | `asyncio` | MCP SDK is async; diagnostic agent already uses `asyncio` |
| Config | Existing `configs/` YAML + env vars | Reuse defaults.yaml, diagnostic.yaml; env prefix `SMOLVLA_` |
| Job Queue | `asyncio.Queue` + background tasks | Serialize GPU work, allow concurrent reads |

### 10.3 Dependencies

The MCP server requires the `mcp` Python SDK (not currently in `requirements.txt`):

```
# New dependencies for MCP server
mcp>=1.0.0              # Official MCP SDK (pip install mcp)
```

All other dependencies (torch, transformers, numpy, Pillow, etc.) are already in the project's requirements. The web backend's service layer (`web/backend/services/`) is imported directly — no FastAPI dependency is introduced.

### 10.4 Relationship to FastAPI Backend

The project already has a FastAPI web backend (`web/backend/`) with overlapping functionality. The MCP server and FastAPI backend should **share the service layer** but run as separate processes:

```
                   ┌─────────────────────┐
                   │ Service Layer        │  ← Shared
                   │ (data_loader,        │
                   │  diagnostic_service, │
                   │  llm_service,        │
                   │  stats_engine,       │
                   │  comparison, etc.)   │
                   └──────┬──────┬───────┘
                          │      │
              ┌───────────┘      └───────────┐
              │                              │
    ┌─────────▼─────────┐        ┌──────────▼──────────┐
    │  MCP Server        │        │  FastAPI Backend     │
    │  (stdio/SSE)       │        │  (HTTP, serves UI)   │
    │  smolvla_inspect/  │        │  web/backend/        │
    │  mcp/server.py     │        │  main.py             │
    └────────────────────┘        └──────────────────────┘
```

**Implementation rules:**
1. MCP tools import from `web.backend.services.*` directly — no code duplication
2. The MCP server does NOT start the FastAPI app or import `web.backend.main`
3. Both share the same SQLite index at `{base_dir}/.smolvla_index.db`
4. Both can run concurrently against the same `base_dir` (SQLite WAL mode handles contention)
5. A future unified entry point could offer: `smolvla-inspect serve --mcp --web` to run both

### 10.5 File Structure

```
smolvla_inspect/
├── mcp/
│   ├── __init__.py
│   ├── server.py              # MCP server setup, tool registration, lifespan
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── discovery.py       # list_runs, search_runs, tag_run, rescan_runs, validate_run
│   │   ├── inspection.py      # get_attention_heatmaps, get_gradient_attribution, ...
│   │   ├── diagnostic.py      # run_diagnostic, get_symptoms, get_findings, ...
│   │   ├── probes.py          # get_semantic_probe, get_qk_probe, get_connector_analysis
│   │   ├── comparison.py      # compare_diagnostic_runs, compare_matrix, ...
│   │   ├── pipeline.py        # run_inspection, run_diagnostic_on_model
│   │   ├── analysis.py        # analyze_with_llm, get_cached_analyses, LLM config
│   │   ├── dataset.py         # get_dataset_diversity, analyze_dataset, parse_task_objects
│   │   ├── notes.py           # get_run_notes, set_run_notes
│   │   └── reports.py         # regenerate_report, export_run_data
│   ├── index.py               # SQLite run index
│   ├── jobs.py                # Async job queue for GPU-bound operations
│   └── resources.py           # MCP resources (run manifests, config, registries)
```

### 10.4 Entry Point

```bash
# As a CLI command
smolvla-inspect mcp --base-dir ./outputs

# For Claude Code config (.mcp.json in project root)
{
  "mcpServers": {
    "smolvla-inspect": {
      "command": "python",
      "args": ["-m", "smolvla_inspect.mcp.server", "--base-dir", "./outputs"],
      "env": {
        "ANTHROPIC_API_KEY": "...",
        "SMOLVLA_LLM_PROVIDER": "anthropic",
        "SMOLVLA_LLM_MODEL": "claude-sonnet-4-20250514"
      }
    }
  }
}

# For Claude Desktop config
{
  "mcpServers": {
    "smolvla-inspect": {
      "command": "python",
      "args": ["-m", "smolvla_inspect.mcp.server", "--base-dir", "/path/to/outputs"]
    }
  }
}
```

### 10.5 MCP Resources (Read-Only Context)

In addition to tools, expose MCP resources for context the LLM can read:

| Resource URI | Description |
|---|---|
| `smolvla://config/defaults` | Default CLI parameters (from defaults.yaml) |
| `smolvla://config/diagnostic` | Diagnostic agent configuration (from diagnostic.yaml) |
| `smolvla://config/gpu` | GPU-specific defaults (from gpu.yaml) |
| `smolvla://registry/primitives` | All 19 registered primitives with category, cost, GPU requirements |
| `smolvla://registry/signals` | All 8 registered expensive signals with cost estimates |
| `smolvla://registry/hypothesis-templates` | All 9 symptom→hypothesis rule-based mappings |
| `smolvla://registry/symptom-detectors` | All 13 symptom detectors with trigger conditions |
| `smolvla://run/{run_id}/manifest` | Run manifest JSON |
| `smolvla://run/{run_id}/diagnostic-report` | Full diagnostic report |
| `smolvla://run/{run_id}/notes` | Run notes |

### 10.6 MCP Prompts (Reusable Workflow Templates)

| Prompt Name | Description | Arguments |
|---|---|---|
| `diagnose-model` | Full diagnostic workflow prompt | `model_id`, `dataset_id` |
| `compare-checkpoints` | Compare training checkpoints | `model_ids[]`, `dataset_id` |
| `experiment-matrix` | Cross-product comparison | `model_ids[]`, `dataset_ids[]`, `metric` |
| `investigate-symptom` | Deep-dive into a specific symptom | `run_id`, `symptom_type` |
| `training-recommendations` | Get actionable training advice | `run_ids[]` |
| `spatial-vs-object` | Investigate whether model uses spatial priors or object features | `run_id` |
| `attention-deep-dive` | Per-head, per-layer, temporal analysis of attention | `run_id` |

---

## 11. Implementation Phases

### Phase 1 — Foundation (index + read-only tools)
1. SQLite run index with full schema (`mcp/index.py`)
2. MCP server skeleton with stdio transport (`mcp/server.py`)
3. Discovery tools: `list_runs`, `search_runs`, `get_run_detail`, `rescan_runs`, `validate_run`
4. Read tools: `get_attention_heatmaps`, `get_gradient_attribution`, `get_model_internals`, `get_viz_stats`, `get_per_step_cross_attention`, `get_vlm_layer_gradcam`, `get_per_action_dim`, `get_language_diff`, `get_vision_vs_state`
5. Image tools: `get_frame_image`, `get_heatmap_image`, `list_run_images`
6. CLI entry point: `smolvla-inspect mcp`

**Deliverable:** Can query existing runs and read all their data via Claude Code.

### Phase 2 — Diagnostics + Probes + Comparison
7. Diagnostic read tools: `get_diagnostic_report`, `get_symptoms`, `get_findings`, `get_hypotheses`, `get_counterfactual_result`, `get_evidence_chain`
8. Probe tools: `get_semantic_probe`, `get_qk_probe`, `get_spatial_object_diagnosis`, `get_connector_analysis`, `get_temporal_trajectory`, `get_occlusion_map`
9. Region tools: `compute_region_attribution`
10. Comparison tools: `compare_diagnostic_runs`, `compare_run_configs`, `compare_component_weights`, `compare_heatmaps`
11. `compare_matrix` cross-product tool
12. MCP resources for config and all registry data
13. Tagging + notes: `tag_run`, `untag_run`, `get_run_notes`, `set_run_notes`

**Deliverable:** Full experiment management and deep diagnostic inspection via natural language.

### Phase 3 — Pipeline Execution + Job Queue
14. Async job queue (`mcp/jobs.py`) with GPU serialization
15. `run_inspection` — trigger full inspection pipeline
16. `run_diagnostic` — trigger diagnostic agent on existing run
17. `run_diagnostic_on_model` — end-to-end (inspection + diagnostic)
18. `run_diagnostic_phase` — run individual diagnostic phases
19. `run_counterfactual` — run individual counterfactual test
20. `run_occlusion_analysis` — compute occlusion sensitivity
21. `get_job_status`, `list_jobs` — job monitoring
22. Progress reporting via MCP notifications

**Deliverable:** Can kick off new analyses, monitor progress, and get results.

### Phase 4 — LLM Analysis + Reports
23. `get_llm_config`, `set_llm_config` — runtime LLM configuration
24. `analyze_with_llm` with streaming support
25. `get_cached_analyses`, `delete_cached_analysis`, `list_analysis_types`
26. `get_prompt_template`, `render_prompt_preview`
27. `regenerate_report`, `regenerate_comparison_report`, `export_run_data`
28. `analyze_dataset`, `parse_task_objects`
29. MCP prompts: all 7 workflow templates

**Deliverable:** Full AI-assisted diagnostic workflow with transparent prompt management.

### Phase 5 — Registry Introspection
30. `list_primitives`, `get_primitive_spec`
31. `list_counterfactual_types`, `get_counterfactual_param_schema`
32. `list_signals`, `list_symptom_detectors`, `list_hypothesis_templates`
33. `format_stats_for_prompt`

**Deliverable:** LLM clients can discover all available capabilities and their parameters.

### Phase 6 — Production Hardening
34. SSE transport for remote access
35. Authentication (API key validation for remote transport)
36. Graceful degradation: tools that require GPU, model, or LLM return structured errors with suggestions when unavailable
37. Comprehensive error messages with recovery actions on every tool
38. Integration tests with mock MCP client
39. Documentation: tool catalog, getting started guide, example queries

---

## 12. Key Design Decisions

### 12.1 Data Size & Serialization

Heatmap arrays can be large (32x32 × N frames). For MCP tool responses:
- **Patch-level heatmaps** (32×32): Return as nested float arrays — small enough for inline JSON (~4KB per frame)
- **Pixel-level maps** (512×512): Return as base64-encoded PNG images — use lossy compression
- **Large datasets** (100+ frames): Support `frame_indices` parameter to request specific frames, never return all frames by default
- **NumPy arrays**: Always `.tolist()` before JSON serialization; BFloat16 must `.float()` first
- **Double-encoded JSON**: Deserializer must handle `json.loads(json.loads(data))` — existing diagnostic reports sometimes double-encode

### 12.2 Long-Running Operations

Inspection and diagnostic runs take minutes to hours. Strategy:
- Tools that run pipelines return immediately with a `job_id`
- Expose `get_job_status(job_id)` and `list_jobs` for monitoring
- Use MCP progress notifications where the SDK supports them
- Alternative: for MCP clients that support it, use streaming tool results

**Phase-level progress labels** (from diagnostic agent callback):
`scene_understanding` → `dataset_diversity` → `triage` → `matrix` → `representation` → `hypothesize` → `counterfactuals` → `iteration` → `disambiguation` → `synthesis` → `complete`

### 12.3 GPU Contention

Only one GPU-intensive operation should run at a time:
- Internal job queue with `max_concurrent_gpu=1`
- Non-GPU tools (reads, comparisons, stats) execute immediately
- Queue status visible via `list_jobs` tool
- Diagnostic agent runs sequentially internally (no parallel signal computation)

### 12.4 Statelessness

The MCP server is stateless between tool calls (all state in SQLite index + filesystem). This means:
- No in-memory caches that drift from disk
- Server can restart without losing state
- Multiple MCP clients can connect to the same base_dir
- Sidecar files (run_notes.json, llm_analyses.json) use atomic temp-file + rename writes

### 12.5 Error Handling & Graceful Degradation

Every tool response includes:
- `success: bool`
- `error?: str` with human-readable message
- `suggestion?: str` with recovery action

**Graceful degradation hierarchy** (matches existing agent behavior):
- No GPU available → skip expensive signals, counterfactuals; return what's possible from existing data
- No policy model loaded → post-hoc mode only (read existing data, no new counterfactuals)
- No dataset available → skip diversity analysis
- No LLM API key → fall back to rule-based hypothesis generation (from `HYPOTHESIS_TEMPLATES` registry)
- Counterfactual budget exceeded → stop after `max_counterfactuals`; confidence reduced 40% for untested hypotheses
- Individual signal failure → log warning, skip, continue with remaining signals

**Common error → suggestion mappings:**
| Error | Suggestion |
|-------|-----------|
| "No diagnostic report found" | "Run `run_diagnostic` to generate one, or check `get_run_detail` to see what data is available" |
| "No gradient data in this run" | "Rerun inspection with `--gradient both` to generate gradient attribution" |
| "Run not found in index" | "Run `rescan_runs` to refresh the index, or `validate_run` to check if the run directory exists" |
| "LLM API key not configured" | "Use `set_llm_config` to configure a provider, or set the `ANTHROPIC_API_KEY` environment variable" |
| "GPU required but not available" | "This operation requires a GPU. Run on a GPU-equipped machine, or use post-hoc analysis tools that read pre-computed data" |

### 12.6 Two Diagnostic Modes

The diagnostic agent supports two operation modes, and the MCP server must handle both:

1. **Integrated mode** (`run_diagnostic_on_model`): Provide `model_id` + `dataset_id` → runs full inspection pipeline first, then diagnostic agent. Produces the richest results but requires GPU + model weights.

2. **Post-hoc mode** (`run_diagnostic` on existing `run_id`): Reads pre-computed data from an existing run directory. Can optionally load the policy model for counterfactuals, but works without it (just skips causal interventions).

### 12.7 Config Precedence

Settings are resolved in this order (later wins):
1. YAML config file (defaults.yaml, diagnostic.yaml)
2. CLI arguments / MCP tool parameters
3. Environment variables (prefix `SMOLVLA_`)

### 12.8 Legacy Run Support

Runs created before the manifest format (`run_manifest.json`) are detected by the presence of known PNG filenames. The system constructs a synthetic manifest via `_build_legacy_manifest()`. Legacy runs:
- Have `is_legacy: true` in the index
- Support image-based visualization only (no `.npz` data)
- Cannot be diagnosed (no structured data to analyze)
- Should surface a clear message when tools that require structured data are called

### 12.9 Context Window Management

Tools that return large structured data must be context-window-aware. Strategies:

**Summary mode**: Tools like `get_diagnostic_report`, `compare_diagnostic_runs`, `get_evidence_chain` support an optional `summary: bool = true` parameter. When true, return high-level summaries and counts instead of full nested data.

**Pagination**: Array-returning tools support `limit: int` and `offset: int`. Responses include `{total_count, returned_count, has_more}`.

**Selective inclusion**: `get_diagnostic_report` supports `include_sections?: str[]` (e.g., `["findings", "symptoms"]`) to return only the requested parts.

**Size guidelines**:
- 32×32 patch heatmaps: ~4KB/frame — inline JSON is fine
- 512×512 pixel maps: return as base64 PNG or resource URI, never as float arrays
- Full diagnostic report: 20-50KB — always offer summary mode
- Evidence chain: 5-100 entries — paginate at 50

### 12.10 Image Handling

Base64-encoded images in tool responses follow these rules:

| Content | Format | Typical size |
|---------|--------|-------------|
| Original frames | JPEG 80% quality | ~50KB |
| Heatmap overlays (patch-level) | PNG 8-bit | ~20KB |
| Counterfactual comparisons | JPEG 85% quality | ~100KB |
| Annotated scene frames | PNG | ~80KB |

- Images < 50KB: inline as `data:image/png;base64,...`
- Images > 50KB: return as MCP resource URI `smolvla://run/{run_id}/images/{path}` — client fetches on demand
- Tools that include images for LLM analysis (`render_prompt_preview`, `get_scene_data`) down-sample to 512×512 max
- `get_frame_image` and `get_heatmap_image` accept optional `max_size?: int` (max dimension in pixels) and `format?: "png"|"jpeg"`

### 12.11 Concurrent Access & SQLite Configuration

The SQLite index at `{base_dir}/.smolvla_index.db` is accessed by multiple clients simultaneously.

**Recommended PRAGMA settings** (in `mcp/index.py`):
```sql
PRAGMA journal_mode=WAL;        -- Write-Ahead Logging (readers don't block writers)
PRAGMA busy_timeout=5000;       -- 5s retry before SQLITE_BUSY error
PRAGMA synchronous=NORMAL;      -- Async fsync (safe with WAL, faster)
```

**Contention patterns:**
- **Reads** (`list_runs`, `search_runs`, `get_run_detail`): Never block each other — WAL allows concurrent reads
- **Brief writes** (`tag_run`, `set_run_notes`): < 100ms lock, acceptable contention
- **Long writes** (`rescan_runs` on 1000+ runs): Up to 30s — should run as a background job via the job queue, not inline

### 12.12 Run Lifecycle & Storage Management

Runs accumulate on disk (1-10GB each). The MCP server should support lifecycle management:

| Tool | Parameters | Description |
|------|-----------|-------------|
| `delete_run` | `run_id: str`, `keep_report?: bool` | Remove run from disk + index. If `keep_report=true`, retains manifest + diagnostic report only (drops npz/images) |
| `get_storage_stats` | — | `{total_bytes, run_count, largest_run, oldest_run, disk_free}` |

**Guidelines:**
- Warn (in tool response metadata) when `disk_free / disk_total < 0.10`
- `validate_run` flags orphaned files not referenced in the manifest
- Future: `archive_run` to move to compressed cold storage

---

## 13. Complete Tool Inventory

### Discovery & Lifecycle (10 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 1 | `list_runs` | No | No | Filter runs by model, dataset, tag, date, capabilities |
| 2 | `search_runs` | No | No | Free-text search across all run metadata + notes |
| 3 | `get_run_detail` | No | No | Full manifest, diagnostic availability, notes |
| 4 | `tag_run` | No | Yes | Add tags to run manifest |
| 5 | `untag_run` | No | Yes | Remove tags from run manifest |
| 6 | `rescan_runs` | No | Yes | Refresh SQLite index from disk |
| 7 | `validate_run` | No | No | Check data integrity of a run directory |
| 8 | `add_run` | No | Yes | Add a specific run directory to the index |
| 9 | `delete_run` | No | Yes | Remove run from disk + index (optionally keep report) |
| 10 | `get_storage_stats` | No | No | Disk usage: total bytes, run count, largest/oldest run, free space |

### Inspection — Visualization Data (12 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 9 | `get_attention_heatmaps` | No | No | Self or cross attention, patch-level |
| 10 | `get_gradient_attribution` | No | No | Saliency, GradCAM SigLIP, GradCAM connector |
| 11 | `get_per_head_attention` | No | No | Per-head heatmaps + entropy values |
| 12 | `get_per_step_cross_attention` | No | No | Per-denoising-step cross-attention |
| 13 | `get_vlm_layer_gradcam` | No | No | Per-VLM-layer GradCAM (vision + language) |
| 14 | `get_per_action_dim` | No | No | Per action dimension GradCAM + magnitudes |
| 15 | `get_language_diff` | No | No | GradCAM diff: original vs alternative task string |
| 16 | `get_vision_vs_state` | No | No | Per-frame vision/state contribution ratios |
| 17 | `get_model_internals` | No | No | WeightWatcher alphas, entropy, redundancy |
| 18 | `get_frame_image` | No | No | Original frame as base64 PNG |
| 19 | `get_heatmap_image` | No | No | Heatmap overlay as base64 PNG |
| 20 | `list_run_images` | No | No | All available image paths in run |

### Statistics (3 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 20 | `get_viz_stats` | No | No | Per-frame + aggregate stats for a viz type |
| 21 | `get_run_summary_stats` | No | No | Aggregated stats across all viz types |
| 22 | `format_stats_for_prompt` | No | No | Pre-formatted text stats for LLM consumption |

### Diagnostic Agent (12 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 23 | `run_diagnostic` | Yes | Yes | Run full 9-phase diagnostic agent |
| 24 | `run_diagnostic_on_model` | Yes | Yes | End-to-end: inspection + diagnostic |
| 25 | `run_diagnostic_phase` | Varies | Yes | Run a single diagnostic phase in isolation |
| 26 | `get_diagnostic_report` | No | No | Load saved DiagnosticReport |
| 27 | `get_diagnostic_matrix` | No | No | Attribution matrix (signal × region → mass) |
| 28 | `get_scene_data` | No | No | Object detections, segmentation, annotated frame |
| 29 | `get_symptoms` | No | No | Detected symptoms with severity + evidence |
| 30 | `get_hypotheses` | No | No | Hypotheses with confidence + test plans |
| 31 | `get_counterfactual_result` | No | No | Specific test result with delta metrics |
| 32 | `get_findings` | No | No | High-level findings with fix suggestions |
| 33 | `get_evidence_chain` | No | No | Full audit trail of diagnostic reasoning |
| 34 | `run_counterfactual` | Yes | Yes | Run a single counterfactual test |

### Representation Probes (4 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 35 | `get_semantic_probe` | No | No | Text↔image similarity for target object vs candidates |
| 36 | `get_qk_probe` | No | No | Per-head spatial vs semantic decomposition |
| 37 | `get_spatial_object_diagnosis` | No | No | Spatial prior vs object grounding verdict |
| 38 | `get_connector_analysis` | No | No | Pixel-shuffle bottleneck information loss |

### Temporal & Spatial Analysis (4 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 39 | `get_temporal_trajectory` | No | No | Attention centroid path + smoothness + tracking correlation |
| 40 | `get_occlusion_map` | No | No | Pre-computed spatial sensitivity grid |
| 41 | `run_occlusion_analysis` | Yes | Yes | Compute occlusion sensitivity from scratch |
| 42 | `compute_region_attribution` | No | No | Per-region attribution shares from heatmap + segmentation |

### Comparison (5 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 43 | `compare_diagnostic_runs` | No | No | Full ComparisonReport with deltas + verdict + recommendations |
| 44 | `compare_run_configs` | No | No | Config fields that differ across runs |
| 45 | `compare_component_weights` | No | No | Per-component weight alpha comparison |
| 46 | `compare_heatmaps` | No | No | Element-wise heatmap diffs |
| 47 | `compare_matrix` | No | No | Cross-product model × dataset comparison |

### Pipeline Execution (4 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 48 | `run_inspection` | Yes | Yes | Full attention + gradient + health pipeline |
| 49 | `add_run` | No | Yes | Add existing run directory to index |
| 50 | `get_job_status` | No | No | Poll status of any long-running job |
| 51 | `list_jobs` | No | No | All tracked jobs with progress info |

### LLM Analysis & Config (8 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 51 | `get_llm_config` | No | No | Current LLM provider/model settings |
| 52 | `set_llm_config` | No | Yes | Update LLM provider/model/API key at runtime |
| 53 | `analyze_with_llm` | No | Yes | Run LLM analysis (caches result) |
| 54 | `get_cached_analyses` | No | No | Previously saved LLM analyses |
| 55 | `delete_cached_analysis` | No | Yes | Remove cached analysis to force re-run |
| 56 | `list_analysis_types` | No | No | Available analysis types with descriptions |
| 57 | `get_prompt_template` | No | No | Raw prompt template with variable placeholders |
| 58 | `render_prompt_preview` | No | No | Fully rendered prompt for debugging/transparency |

### Notes & Annotations (2 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 59 | `get_run_notes` | No | No | Load per-run markdown notes |
| 60 | `set_run_notes` | No | Yes | Save per-run notes |

### Dataset Analysis (3 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 61 | `get_dataset_diversity` | No | No | Load pre-computed diversity report |
| 62 | `analyze_dataset` | Yes | No | Run standalone dataset diversity analysis |
| 63 | `parse_task_objects` | No | No | Extract object names from task string |

### Report & Export (3 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 64 | `regenerate_report` | No | Yes | Rebuild markdown from existing JSON |
| 65 | `regenerate_comparison_report` | No | Yes | Rebuild comparison report from existing data |
| 66 | `export_run_data` | No | No | Export/re-export structured data |

### Registry Introspection (5 tools)

| # | Tool | GPU | Mutates | Description |
|---|------|-----|---------|-------------|
| 67 | `list_primitives` | No | No | All 19 registered diagnostic primitives |
| 68 | `get_primitive_spec` | No | No | Full spec with param_schema for one primitive |
| 69 | `list_counterfactual_types` | No | No | All 8 test types with param schemas |
| 70 | `get_counterfactual_param_schema` | No | No | JSON schema for a specific test type |
| 71 | `list_signals` | No | No | All 8 expensive signals with costs |
| 72 | `list_symptom_detectors` | No | No | All 13 symptom detectors |
| 73 | `list_hypothesis_templates` | No | No | All 9 symptom→hypothesis rule mappings |

---

**Summary: 77 tools** — 55 read-only, 22 mutating (of which 7 require GPU).

### Tool count by category

| Category | Read | Write | GPU | Total |
|----------|------|-------|-----|-------|
| Discovery & Lifecycle | 5 | 5 | 0 | 10 |
| Inspection | 12 | 0 | 0 | 12 |
| Statistics | 3 | 0 | 0 | 3 |
| Diagnostic Agent | 8 | 4 | 3 | 12 |
| Probes | 4 | 0 | 0 | 4 |
| Temporal/Spatial | 3 | 1 | 1 | 4 |
| Comparison | 5 | 0 | 0 | 5 |
| Pipeline | 3 | 2 | 1 | 4 |
| LLM Analysis | 5 | 3 | 0 | 8 |
| Notes | 1 | 1 | 0 | 2 |
| Dataset | 2 | 1 | 1 | 3 |
| Reports/Export | 1 | 2 | 0 | 3 |
| Registry | 7 | 0 | 0 | 7 |
| **Total** | **55** | **22** | **7** | **77** |

---

## 14. Diagnostic Agent Internals

Details on the agent's internal workflow that affect tool behavior and output interpretation.

### 14.1 Signal Triage Workflow

The agent doesn't run all 8 expensive signals — it triages based on cheap signal results:

1. **Always-run signals**: `attention` (hook-based, free), `positional_baseline` (cheap computation)
2. **Mandatory expensive signal**: `vision_vs_state` (always runs regardless of triage)
3. **LLM-selected signals**: The remaining signals are ranked by an LLM triage prompt that receives:
   - Foreground ratio from cheap attention
   - Positional baseline similarity score
   - List of available signals with cost descriptions
4. **Budget**: `max_expensive_signals` (default 5) caps how many run

If LLM is unavailable, falls back to running: `gradcam_siglip`, `saliency`, `vision_vs_state`.

### 14.2 Mandatory Counterfactual Tests

For cross-run comparability, certain counterfactual tests **always run** (if the model is loaded), regardless of LLM selections:

- `background_substitution`
- `object_relocation`
- `task_string_swap`
- `occlusion_targeted`

This list is configurable via `mandatory_counterfactual_tests` in diagnostic.yaml.

### 14.3 Iteration & Disambiguation

After the initial hypothesis→counterfactual cycle, the agent may run follow-up iterations:

- **Iteration trigger**: When counterfactual results are "surprising" — high-confidence hypothesis was unconfirmed, or a low-confidence one was strongly confirmed
- **Budget**: `max_hypothesis_iterations` (default 1) — how many follow-up rounds
- **Disambiguation**: If `spatial_object_diagnosis` returns `"mixed"` or `"inconclusive"`, OR the semantic probe margin is < 0.08, the QK probe runs to decompose per-head attention into spatial vs semantic components. The diagnosis is then recomputed with QK evidence.

### 14.4 Rule-Based Fallback System

When the LLM is unavailable (no API key, rate limit, network error), the agent falls back to:

1. **Hypothesis generation**: Uses `HYPOTHESIS_TEMPLATES` registry — 9 symptom→hypothesis mappings with pre-defined test types and expected outcomes
2. **Finding generation**: Uses an internal `_EXPERT_DB` with 11 symptom-specific expert fixes covering: high background attribution, spatial shortcuts, low object attribution, dead state pathway, gripper fixation, language insensitivity, cross-attention diffuse, temporal instability, unstable GradCAM, action-attention misalignment, low dataset diversity
3. **Narrative generation**: Produces a prioritized action plan (severity-ordered) without LLM synthesis

The rule-based path produces comparable quality for well-understood symptoms but lacks the nuanced cross-signal reasoning of the LLM path.

### 14.5 Evidence Log

The `DiagnosticAgent` maintains an `evidence_log: list[EvidenceEntry]` throughout execution. Each entry contains:

```
{
  "phase": str,        # "scene_understanding", "matrix", "counterfactual", "iteration", ...
  "type": str,         # "detection", "signal", "symptom", "hypothesis", "test_result", ...
  "data": dict,        # Phase-specific structured data
  "timestamp": str     # ISO 8601
}
```

This log is serialized to `diagnostic/evidence_chain.json` and is accessible via `get_evidence_chain`. It provides a complete audit trail of why the agent made each decision.

### 14.6 Spatial vs Object Diagnosis Scoring

The `SpatialObjectDiagnosis` verdict is computed by aggregating 15+ weighted evidence sources from `smolvla_inspect/diagnostic/spatial_object.py:build_spatial_object_diagnosis()`.

Each source contributes a `strength` value [0, 1] computed via linear interpolation between low/high thresholds. Sources are classified as either "spatial evidence" (supports spatial memorisation) or "object evidence" (supports genuine object grounding).

**Scoring:**
- `spatial_score = weighted_mean(spatial_evidence_strengths)`
- `object_score = weighted_mean(object_evidence_strengths)`

**Verdicts:**
| Verdict | Condition |
|---------|-----------|
| `spatial_prior` | spatial_score > object_score + 0.2 |
| `object_grounded` | object_score > spatial_score + 0.2 |
| `mixed` | \|spatial_score - object_score\| <= 0.2 |
| `inconclusive` | Insufficient evidence or very low confidence |

**Key evidence sources (by weight):**

| Source | Type | Weight | What it measures |
|--------|------|--------|------------------|
| Object relocation follow probe | object | 0.26 | Actions follow relocated object |
| Object relocation anchor probe | spatial | 0.26 | Actions stay at original position |
| Positional baseline ratio | spatial | 0.24 | Attention matches positional prior |
| Semantic relocation probes | varies | 0.20 | Semantic similarity shifts with relocation |
| GradCAM target share | object | 0.18 | Target object receives GradCAM mass |
| Semantic target margin | object | 0.18 | SigLIP text-patch similarity for target |
| Low target GradCAM | spatial | 0.16 | Target object has very low GradCAM |
| Dataset position variance | spatial | 0.14 | Object appears in narrow position range |
| QK probe heads | varies | 0.14 | Per-head spatial vs semantic decomposition |
| Temporal tracking | object | 0.12 | Attention centroid follows object across frames |
| Occlusion response | varies | 0.08-0.18 | Action sensitivity when target is occluded |

Confidence = `min(total_spatial_weight, total_object_weight) / max_possible_weight`. Higher confidence means more evidence sources were available and contributed.

---

## 15. Symptom Detectors Reference

All 13 registered symptom detectors in `smolvla_inspect/diagnostic/matrix.py`:

| # | Detector | Severity | What it checks |
|---|----------|----------|----------------|
| 1 | `high_background_attribution` | warning | Background region receives >60% of GradCAM or saliency mass |
| 2 | `attention_gradcam_divergence` | warning | Attention and GradCAM disagree on which regions matter (high KL divergence) |
| 3 | `dead_state_pathway` | warning | Vision contributes >95% of action gradient (proprioceptive state ignored) |
| 4 | `low_object_attribution` | warning | Target object receives <5% of GradCAM mass |
| 5 | `spatial_shortcut` | critical | Positional baseline explains >70% of attention pattern (cosine similarity) |
| 6 | `language_insensitivity` | warning | Language diff signal shows <5% GradCAM change when task string is altered |
| 7 | `unstable_gradcam` | info | GradCAM temporal stability (cosine similarity between consecutive frames) < 0.5 |
| 8 | `low_dataset_diversity` | warning | Object position std < 15px OR background diversity score < 0.05 |
| 9 | `gripper_fixation` | warning | Gripper region receives >40% of attention AND target object < 10% |
| 10 | `cross_attention_diffuse` | info | Cross-attention entropy ratio > 0.9 (near-uniform distribution) |
| 11 | `action_attention_misalignment` | info | Per-action-dim GradCAM centroids diverge significantly from attention centroid |
| 12 | `temporal_attention_instability` | warning | Attention centroid drift > 100px between consecutive frames |
| 13 | `single_region_dependency` | info | One non-background region captures >80% of attribution across all signals |

---

## 16. Configuration Reference

### 16.1 Diagnostic Agent Config (`configs/diagnostic.yaml`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `scene_model` | str | `"google/owlv2-base-patch16-ensemble"` | OWL-ViT model for object detection |
| `segmentation_model` | str | `"facebook/sam-vit-base"` | SAM model for segmentation |
| `detection_confidence` | float | `0.1` | Object detection confidence threshold |
| `diversity_episodes` | int | `10` | Episodes to sample for dataset diversity |
| `diversity_frames_per_episode` | int | `3` | Frames per episode for diversity |
| `num_frames` | int | `4` | Frames to analyze per run |
| `max_expensive_signals` | int | `5` | Budget for LLM-selected expensive signals |
| `max_hypotheses` | int | `5` | Max hypotheses to generate |
| `max_counterfactuals` | int | `7` | Max counterfactual tests to run |
| `max_hypothesis_iterations` | int | `1` | Follow-up iteration rounds |
| `mandatory_counterfactual_tests` | str[] | `["background_substitution", "object_relocation", "task_string_swap", "occlusion_targeted"]` | Tests that always run |
| `occlusion_patch_size` | int | `64` | Patch size for occlusion sensitivity |
| `occlusion_stride` | int | `32` | Stride for occlusion sensitivity |
| `temporal_frames` | int | `20` | Frames for temporal trajectory analysis |
| `llm_provider` | str | — | Override LLM provider (optional, prefers env var) |
| `llm_model` | str | — | Override LLM model (optional) |

### 16.2 Quick Config Profile (`configs/diagnostic_quick.yaml`)

A reduced-cost profile for fast iteration:

| Key | Quick Value | Default Value |
|-----|------------|---------------|
| `diversity_episodes` | 5 | 10 |
| `diversity_frames_per_episode` | 2 | 3 |
| `max_expensive_signals` | 3 | 5 |
| `num_frames` | 2 | 4 |

Pass via `config_path="configs/diagnostic_quick.yaml"` on `run_diagnostic` or `run_diagnostic_on_model`.

### 16.3 Inspection Pipeline Config (`configs/defaults.yaml`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `method` | str | `"last-layer"` | Attention aggregation: last-layer, rollout, all-layers |
| `cross_attention` | bool | `false` | Capture action expert cross-attention |
| `show_heads` | bool | `false` | Per-head attention grid for first frame |
| `raw_attention` | bool | `false` | Skip positional baseline subtraction |
| `attn_threshold` | float | `0.5` | Percentile threshold for zeroing low attention values |
| `gradient` | str\|null | `null` | saliency, gradcam, both, or null |
| `smooth_grad` | int | `1` | SmoothGrad sample count (1 = no smoothing) |
| `smooth_grad_sigma` | float | `0.15` | Gaussian noise std for SmoothGrad |
| `gradient_seed` | int | `42` | Fixed noise seed for reproducibility |
| `gradcam_connector` | bool | `false` | GradCAM on pixel-shuffle connector |
| `gradcam_vlm_layers` | str\|null | `null` | Comma-separated layer indices (e.g. "4,8,12,16") |
| `per_step_cross_attention` | bool | `false` | Per-denoising-step visualization |
| `vision_vs_state` | bool | `false` | Vision vs state gradient ratio |
| `per_action_dim` | bool | `false` | Per-action-dimension GradCAM |
| `language_diff` | bool | `false` | Language conditional GradCAM |
| `skip_attention` | bool | `false` | Skip hook-based attention capture; gradient-only mode |
| `with_internals` | bool | `false` | Run model health analysis after inspection |
| `internals_frames` | int | `5` | Frames for entropy/redundancy analysis |
| `export_data` | bool | `true` | Export structured npz + JSON data |

### 16.4 Model Health Thresholds

| Key | Default | Description |
|-----|---------|-------------|
| `entropy_warn` | `0.8` | Entropy ratio above this = possible over-smoothing |
| `entropy_critical` | `0.95` | Near-uniform attention (not learning) |
| `entropy_low` | `0.1` | Suspiciously focused (possible collapse) |
| `redundancy_warn` | `0.7` | Head cosine similarity above this = redundant |
| `redundancy_critical` | `0.9` | Nearly identical heads |

### 16.5 Environment Variables

| Variable | Description |
|----------|-------------|
| `SMOLVLA_LLM_PROVIDER` | LLM provider: "anthropic" or "openai" |
| `SMOLVLA_LLM_MODEL` | Model ID (e.g. "claude-sonnet-4-20250514") |
| `SMOLVLA_LLM_API_KEY` | API key (overrides provider-specific vars) |
| `SMOLVLA_LLM_BASE_URL` | Custom endpoint for OpenAI-compatible servers |
| `ANTHROPIC_API_KEY` | Fallback if provider=anthropic and no SMOLVLA_LLM_API_KEY |
| `OPENAI_API_KEY` | Fallback if provider=openai and no SMOLVLA_LLM_API_KEY |

---

## 17. Data Model Reference

### 17.1 Run Manifest Schema (`run_manifest.json`)

```json
{
  "version": 2,
  "created_at": "2025-03-15T10:30:00Z",
  "cli_args": {
    "model": "lerobot/smolvla_base",
    "dataset": "lerobot/svla_so101_pickplace",
    "episode": 0,
    "num_frames": 8,
    "method": "last-layer",
    "cross_attention": true,
    "gradient": "both",
    "gradient_seed": 42
  },
  "model_info": {
    "model_id": "lerobot/smolvla_base",
    "vision_encoder": {
      "image_size": 512, "patch_size": 16,
      "num_heads": 12, "num_layers": 12,
      "grid_size": 32, "num_patches": 1024
    },
    "vlm": { "num_heads": 15, "num_layers": 32 },
    "expert": { "num_heads": 15, "num_layers": 16 },
    "action_dim": 7,
    "total_params_m": 500.0
  },
  "dataset_info": {
    "dataset_id": "lerobot/svla_so101_pickplace",
    "episode_idx": 0,
    "num_frames": 8,
    "task_string": "pick the lego and place it in the bowl",
    "total_frames": 1200,
    "action_dim_names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos", "gripper.vel"],
    "image_keys": ["observation.images.top"]
  },
  "available_visualizations": {
    "self_attention": true,
    "cross_attention": true,
    "per_head": true,
    "per_step_cross_attention": true,
    "saliency": true,
    "gradcam_siglip": true,
    "gradcam_connector": false,
    "gradcam_vlm_layers": false,
    "per_action_dim": true,
    "language_diff": false,
    "vision_vs_state": true,
    "model_internals": true
  },
  "tags": ["baseline", "v1"],
  "images": ["images/episode_dashboard_ep000.png", "..."]
}
```

### 17.2 Sidecar Files

**`run_notes.json`** — User annotations:
```json
{
  "notes": "Baseline run before fine-tuning. Background reliance is high.",
  "updated_at": "2025-03-15T14:22:00Z"
}
```

**`llm_analyses.json`** — Cached LLM analysis results:
```json
{
  "version": 1,
  "analyses": {
    "run_insights": {
      "response": "The model shows strong spatial priors...",
      "model": "claude-sonnet-4-20250514",
      "provider": "anthropic",
      "custom_prompt": null,
      "created_at": "2025-03-15T15:00:00Z",
      "updated_at": "2025-03-15T15:00:00Z"
    }
  }
}
```

Both use atomic writes (temp file + rename) to prevent corruption.

### 17.3 Diagnostic Report Files

| File | Content |
|------|---------|
| `diagnostic/report.json` | Full `DiagnosticReport` serialized as JSON |
| `diagnostic/report.md` | Human-readable markdown report |
| `diagnostic/matrix.json` | `DiagnosticMatrix` standalone (signal × region attribution) |
| `diagnostic/evidence_chain.json` | `EvidenceEntry[]` — ordered audit trail |
| `diagnostic/scene/detections.json` | Object detection results |
| `diagnostic/scene/segmentation.npz` | Per-object binary masks |
| `diagnostic/scene/annotated_frame.png` | Visualization with bounding boxes |
| `diagnostic/counterfactuals/{test_name}/result.json` | `CounterfactualResult` per test |
| `diagnostic/counterfactuals/{test_name}/comparison.png` | Side-by-side original vs perturbed |

### 17.4 Model Health Report Components

The `get_model_internals` tool returns three sub-reports:

| Component | What it measures | Key metrics |
|-----------|------------------|-------------|
| **WeightWatcher alpha** | Power-law exponent of weight matrix singular values per layer | `mean_alpha` per component. Healthy: 2-4. <2: overcorrelated. >6: severely undertrained |
| **Attention entropy** | Per-head entropy ratio (actual/max possible) | Ratio near 1.0 = near-uniform (not discriminating). <0.1 = collapsed to single token |
| **Head redundancy** | Pairwise cosine similarity between attention heads | >0.7 = heads learning similar patterns. >0.9 = nearly identical (wasted capacity) |

---

## 18. Streaming & Event Protocol

### 18.1 Diagnostic Streaming (SSE)

When `run_diagnostic` or `run_diagnostic_on_model` is triggered via the web backend, progress is reported as Server-Sent Events:

```
event: started
data: {"run_id": "abc123", "phases": ["scene", "diversity", "triage", ...]}

event: progress
data: {"phase": "scene_understanding", "detail": "Detecting objects with OWL-ViT..."}

event: progress
data: {"phase": "counterfactuals", "detail": "Running background_substitution (3/7)..."}

event: complete
data: {"run_id": "abc123", "report_path": "diagnostic/report.json"}

event: error
data: {"message": "OOM during GradCAM computation", "phase": "triage"}
```

### 18.2 LLM Analysis Streaming (SSE)

Token-by-token streaming for `analyze_with_llm`:

```
data: {"token": "The model"}
data: {"token": " shows strong"}
data: {"token": " spatial priors..."}
data: [DONE]
```

### 18.3 MCP Transport Mapping

- **stdio transport**: Long-running tools return `job_id` immediately; client polls via `get_job_status`. No streaming.
- **SSE transport**: Progress notifications sent as MCP protocol notifications. LLM streaming maps to chunked tool responses where the SDK supports it.
- **Progress callback**: The diagnostic agent accepts `progress_callback: Callable[[str, str], None]` (phase, detail) — the MCP server converts these to the appropriate transport mechanism.

---

## 19. Extensibility

### 19.1 Registry-Based Plugin System

The diagnostic agent's capabilities are registered via decorators in `smolvla_inspect/diagnostic/registry.py`. The MCP server exposes these registries read-only via introspection tools (`list_primitives`, `list_signals`, etc.).

**Adding a new diagnostic primitive:**
```python
# In smolvla_inspect/diagnostic/my_custom_primitive.py
from smolvla_inspect.diagnostic.registry import register_primitive

@register_primitive(
    name="custom.my_analysis",
    category="model",
    cost="moderate",
    requires_gpu=True,
    requires_model=True,
    description="My custom analysis",
    prompt_description="Runs custom analysis on vision features",
    param_schema='{"threshold": "float, default 0.5"}',
)
def my_analysis(policy, frames, scene, **kwargs):
    ...
    return result
```

**Adding a new expensive signal:**
```python
from smolvla_inspect.diagnostic.registry import register_signal

@register_signal(
    name="my_signal",
    cost_description="~45s/frame",
    prompt_description="Computes custom attribution via my method",
    requires_model=True,
)
def run_my_signal(policy, frames, device, **kwargs):
    ...
    return heatmaps
```

**Adding a new symptom detector:**
```python
from smolvla_inspect.diagnostic.matrix import register_symptom_detector

@register_symptom_detector("my_custom_symptom")
def detect_my_symptom(matrix, scene, config):
    if matrix.scalars.get("my_metric", 0) > 0.8:
        return Symptom(type="my_custom_symptom", severity="warning", ...)
    return None
```

New registrations are automatically discovered by `list_primitives`, `list_signals`, and `list_symptom_detectors` — no MCP server changes needed.

### 19.2 Custom LLM Providers

The `LLM_PROVIDERS` registry supports adding custom backends:
```python
from smolvla_inspect.diagnostic.registry import register_llm_provider

@register_llm_provider("my_provider")
async def my_llm_call(prompt, settings):
    # Call custom LLM endpoint
    return response_text
```

---

## 20. Testing Strategy

### 20.1 Existing Test Infrastructure

- `run_single_counterfactual.py` — Standalone runner for individual counterfactual tests on existing runs
- `regenerate_report.py` — Report regeneration without GPU (useful for testing report logic)
- Pre-computed runs in `outputs/` — Can serve as test fixtures

### 20.2 MCP Server Test Plan

```
tests/
├── test_mcp_discovery.py       # list_runs, search_runs, tag/untag, index operations
├── test_mcp_inspection.py      # All read-only viz tools against fixture runs
├── test_mcp_diagnostic.py      # get_diagnostic_report, get_symptoms, get_findings
├── test_mcp_comparison.py      # compare_diagnostic_runs, compare_matrix
├── test_mcp_probes.py          # get_semantic_probe, get_qk_probe
├── test_mcp_jobs.py            # Job queue, GPU serialization, status polling
├── test_mcp_context_limits.py  # Summary mode, pagination, selective inclusion
├── test_mcp_errors.py          # Graceful degradation: no GPU, no LLM, missing data
├── fixtures/
│   ├── sample_run_with_diagnostic/   # Pre-computed run with full diagnostic data
│   ├── sample_run_minimal/           # Run with only self-attention (no gradient)
│   └── sample_legacy_run/            # Pre-manifest run for legacy handling
└── conftest.py                 # Shared fixtures, mock GPU operations
```

**Testing principles:**
- All GPU-bound tools are mocked in unit tests — GPU tests are integration-only
- Pre-computed fixture runs provide deterministic test data
- Every tool is tested for both success and expected error cases
- Schema validation ensures tool responses match documented data models
- Run `pytest tests/ -m "not gpu"` for fast CI; `pytest tests/` for full suite
