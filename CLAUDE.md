# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

`smolvla-inspect` is an interpretability tool for SmolVLA — a vision-language-action robot policy. It extracts and visualizes attention maps (SigLIP self-attention and action expert cross-attention), gradient-based attribution (saliency, GradCAM), and runs model health diagnostics. It also includes an interactive web viewer for exploring results.

## Commands

### Run attention visualization

```bash
source .venv/bin/activate
./run.sh                          # Default: rollout + cross-attn + per-head grid
./run.sh --config configs/gpu.yaml  # GPU config (all features)
./run.sh --num-frames 2           # Quick sanity check
```

`run.sh` sets `DYLD_LIBRARY_PATH` to find FFmpeg 6 libs on macOS. Always use it instead of calling `python inspect_attention.py` directly unless the env var is already set.

### Web viewer

```bash
# Development (backend + frontend together):
./start_servers.sh
# Backend at http://localhost:8080, frontend at http://localhost:5173

# Production (built frontend):
cd web/frontend && npm install && npm run build && cd ../..
python inspect_attention.py serve --port 8080 --base-dir ./outputs
```

### Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS: FFmpeg 6 required (brew install ffmpeg@6)
# Ubuntu GPU: chmod +x setup-gpu.sh && ./setup-gpu.sh
```

## Architecture

### Python package: `smolvla_inspect/`

`inspect_attention.py` is a thin entry point that delegates to the package. The main modules:

- **`cli.py`** — CLI argument parsing, top-level orchestration. Calls into all other modules. Contains `extract_attention_maps()` and `main()`.
- **`capture.py`** — PyTorch forward hooks: `SigLIPAttentionCapture` (hooks SigLIP ViT attention layers) and `ActionVisionAttentionCapture` (monkey-patches `eager_attention_forward()` to intercept expert→vision cross-attention).
- **`heatmap.py`** — Converts raw attention weights to spatial heatmaps: patch score aggregation, attention rollout, positional baseline computation and subtraction, bilinear upsampling.
- **`gradient.py`** — Gradient attribution: `compute_gradient_maps()` (saliency + GradCAM), `compute_gradcam_connector_maps()`, `compute_gradcam_vlm_layers_maps()`, `compute_vision_vs_state_maps()`, `compute_per_action_dim_maps()`, `compute_language_conditional_maps()`.
- **`data.py`** — Dataset loading helpers, image key detection, batch construction for the policy forward pass.
- **`viz.py`** — Grid image assembly, heatmap overlays, per-head grids, per-step cross-attention grids, per-action-dim grids.
- **`health.py`** — Model health diagnostics: spectral analysis (WeightWatcher), attention entropy, head redundancy.
- **`export.py`** — Structured output: creates run directories, saves frames and attention data, builds `manifest.json` for the web viewer.
- **`serve.py`** — Launches the `serve` subcommand.

### Web viewer: `web/`

- **`web/backend/`** — FastAPI app (`main.py` → `create_app()`). Routers in `routers/` handle: runs, visualizations, health, compare, LLM analysis, notes, LLM cache. Services in `services/` handle run scanning and image loading. Settings in `config.py` (pydantic-settings).
- **`web/frontend/`** — React + Vite + TypeScript. State managed with Zustand stores (`src/stores/`). API calls in `src/services/`. Key components: `RunSelector`, `HeatmapCanvas`, `LLMPanel`.

### Key data flow

1. LeRobot dataset → `data.py` builds policy batches → `cli.py` runs forward pass with hooks attached
2. `capture.py` hooks collect raw attention tensors during forward pass
3. `heatmap.py` converts attention tensors to normalized 2D heatmaps (patch scores → rollout/last-layer → positional baseline subtraction → bilinear upsample)
4. `gradient.py` runs `.backward()` from predicted action to input pixels for gradient attribution
5. `viz.py` assembles multi-row grid PNGs
6. `export.py` saves structured output + `manifest.json` to `outputs/<run>/`
7. Web viewer reads `outputs/` via FastAPI; frontend renders with React

### Configs

`configs/defaults.yaml` — CPU-friendly, no gradients. `configs/gpu.yaml` — all features enabled (CUDA, SmoothGrad N=20, all extended attribution). CLI flags always override config values.

### Output structure

Results go to `outputs/<run>/`. The web viewer discovers runs by scanning for `manifest.json` files in subdirectories of `--base-dir`.

## Notes on key implementation details

- **Cross-attention capture**: The expert's cross-attention to vision tokens is captured by monkey-patching `eager_attention_forward()` on the expert layers, not via standard hooks. Only vision token columns are retained from the full KV cache.
- **Positional baseline**: A forward pass with a solid gray image captures position-only attention. This baseline is subtracted before thresholding to remove SigLIP's learned positional bias.
- **Split device execution**: Attention can run on MPS while gradients run on CPU (`--device mps --gradient-device cpu`) because MPS has limited `.backward()` support.
- **FFmpeg requirement**: LeRobot uses TorchCodec for video decoding, which requires FFmpeg 4–7. FFmpeg 8 (Homebrew default) does not work — install `ffmpeg@6`.
