# smolvla-inspect

Inspect where a SmolVLA policy looks, what pixels actually drive its actions, and how its internal attention/weight structure behaves.

![Example attention grid](assets/example_grid.png)
*Example inspection grid for a pick-and-place episode. It combines raw attention, overlays, and gradient attribution in one view.*

## Table of Contents

- [Quick Start](#quick-start)
- [What This Tool Does](#what-this-tool-does)
- [Setup](#setup)
- [Run](#run)
- [Web Viewer](#web-viewer)
- [How It Works](#how-it-works)
- [Interpreting Results](#interpreting-results)
- [CLI Reference](#cli-reference)
- [Project Layout](#project-layout)
- [Roadmap](#roadmap)

## Quick Start

### 1. Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, install FFmpeg 6 first:

```bash
brew install ffmpeg@6
```

### 2. Run a standard inspection

```bash
source .venv/bin/activate
./run.sh
```

### 3. Run gradients and the web-viewer-friendly export

```bash
./run.sh --config configs/gpu.yaml
```

### 4. Run the model internals report

```bash
./run.sh --internals-only
```

Or append it to a normal run:

```bash
./run.sh --with-internals
```

### 5. Launch the web viewer

```bash
source .venv/bin/activate
./start_servers.sh
```

Results are written to `outputs/` by default.

## What This Tool Does

SmolVLA is a vision-language-action policy: it takes camera images and a language instruction, then predicts robot actions. This repository gives you four practical ways to inspect that behavior:

| Capability | Primary flags | What it answers | Main outputs |
|------------|---------------|-----------------|--------------|
| Attention visualization | default, `--cross-attention`, `--show-heads` | Where does the encoder or action decoder focus? | `episode_dashboard_ep*.png`, `per_head_ep*.png` |
| Gradient attribution | `--gradient` | Which pixels causally affect the predicted action? | extra rows in the dashboard |
| Extended attribution | `--gradcam-connector`, `--gradcam-vlm-layers`, `--vision-vs-state`, `--per-action-dim`, `--language-diff` | How information moves through the connector, VLM, and action heads | feature-specific PNGs / reports |
| Model internals report | `--internals-only`, `--with-internals` | Are weights and attention heads well-behaved internally? | `model_internals_report.md`, `model_internals_report.png` |

### Core outputs

- Main grid per episode: attention, overlays, and optional gradient rows.
- Optional per-head grid for SigLIP attention heads on the first frame.
- Structured run directory for the web viewer when `--export-data` is enabled.
- Model internals report covering spectral alpha, attention entropy, and head redundancy.

### Model internals at a glance

The internals report runs three checks across the SigLIP vision encoder, VLM text model, action expert, connector, and projection heads:

1. Weight spectral analysis with WeightWatcher.
2. Attention entropy across key attention operations.
3. Head redundancy within each layer.

Use `--internals-only` when you want just that report. Use `--with-internals` when you want it in addition to the normal attention / gradient run.

![Example model internals report](assets/example_model_internals_report.png)
*Example 3-panel internals report. The full markdown version lives at [assets/example_model_internals_report.md](assets/example_model_internals_report.md).*

For a visual walkthrough of the architecture behind these views, see [assets/architecture.md](assets/architecture.md).

## Setup

### Requirements

- Python 3.10+
- FFmpeg 4-7 for video decoding through TorchCodec
- Node.js 20.19+ for the web viewer

### macOS

Install FFmpeg 6:

```bash
brew install ffmpeg@6
```

Then install Python dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Node.js if you want the web viewer:

```bash
brew install node
```

### Ubuntu + CUDA

For an NVIDIA GPU machine, use the setup helper:

```bash
chmod +x clone-and-setup.sh && ./clone-and-setup.sh
```

Or if the repo is already cloned:

```bash
chmod +x setup-gpu.sh && ./setup-gpu.sh
```

`setup-gpu.sh` installs CUDA-compatible PyTorch, creates a virtualenv, installs dependencies, and checks GPU access.

### Node.js note

On Ubuntu, the default `apt install nodejs` is often too old. Prefer [NodeSource](https://github.com/nodesource/distributions) or `nvm`.

## Run

Use `run.sh` on macOS so TorchCodec can find FFmpeg 6:

```bash
source .venv/bin/activate
./run.sh
```

If you prefer calling Python directly:

```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@6/lib:$DYLD_LIBRARY_PATH"
python inspect_attention.py
```

### Config files

Defaults come from `configs/defaults.yaml`. Use `--config` to load another config; direct CLI flags still override config values.

| Config | Purpose |
|--------|---------|
| `configs/defaults.yaml` | Conservative CPU-friendly defaults |
| `configs/gpu.yaml` | CUDA-oriented config with gradients and extended attribution enabled |

Example:

```bash
./run.sh --config configs/gpu.yaml
./run.sh --config configs/gpu.yaml --episode 3
```

### Common commands

#### Basic attention inspection

```bash
./run.sh
./run.sh --model path/to/finetuned_checkpoint --dataset path/to/dataset
./run.sh --episode 3 --num-frames 12
./run.sh --task "pick up the red cube"
./run.sh --method last-layer
./run.sh --raw-attention
./run.sh --attn-threshold 0.7
./run.sh --attn-threshold 0
```

#### Gradient attribution

```bash
./run.sh --gradient
./run.sh --gradient saliency
./run.sh --gradient saliency --smooth-grad 20
./run.sh --device mps --gradient both --gradient-device cpu
```

#### Extended attribution

```bash
./run.sh --cross-attention --per-step-cross-attention
./run.sh --gradient gradcam --gradcam-connector --gradcam-vlm-layers 4,8,12,16
./run.sh --gradient gradcam --vision-vs-state
./run.sh --gradient gradcam --per-action-dim
./run.sh --language-diff "pick up the blue cube"
```

#### Model internals

```bash
./run.sh --internals-only
./run.sh --with-internals
./run.sh --internals-only --internals-frames 10
./run.sh --internals-only --entropy-warn 0.85 --redundancy-warn 0.75
```

Backward-compatible aliases `--model-health` and `--health-frames` are still accepted, but `--internals-only` and `--internals-frames` are the primary names now.

### Output layout

With `--export-data` enabled, each run gets a structured folder under `outputs/`:

```text
run_YYYY-MM-DD_HH-MM-SS/
  images/
  data/
  run_manifest.json
```

That structure is what the web viewer reads.

## Web Viewer

The web viewer lets you:

- browse generated runs and available visualizations,
- inspect frames interactively,
- compare runs side by side,
- view model internals when exported,
- attach LLM-generated analysis to runs and visualizations.

![Main visualization view](assets/web_viewer_main.png)
*Browsing per-frame visualizations in the main viewer.*

![Run Insights with LLM analysis](assets/web_viewer_insights.png)
*Run Insights summarizes statistics across a run and supports LLM analysis.*

![Compare Runs](assets/web_viewer_compare.png)
*Compare multiple runs side by side.*

### Development launch

```bash
source .venv/bin/activate
./start_servers.sh
./start_servers.sh --base-dir ./my_outputs
```

This starts:

- backend on `http://localhost:8080`
- frontend on `http://localhost:5173`

### Production-style launch

Build the frontend once, then serve from FastAPI:

```bash
cd web/frontend
npm install
npm run build
cd ../..
python inspect_attention.py serve --port 8080 --base-dir ./outputs
```

### LLM setup

Set one of these before launching if you want LLM analysis:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

### `serve` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8080` | Server port |
| `--host` | `0.0.0.0` | Server host |
| `--base-dir` | `./outputs` | Root directory scanned for runs |
| `--no-open` | off | Do not auto-open the browser |

## How It Works

![Architecture and attention-to-heatmap pipeline](assets/how_it_works_architecture.png)
*Left: where attention is captured. Right: how patch attention becomes a spatial heatmap.*

### Pipeline

1. Load a SmolVLA policy and a LeRobot dataset.
2. Capture self-attention from the SigLIP vision encoder.
3. Optionally capture action-expert cross-attention into the VLM prefix.
4. Convert patch-level scores into spatial heatmaps.
5. Optionally compute gradients, GradCAM, or extended attribution views.
6. Save images plus structured data for the viewer.

### Main dashboard rows

| Row | Content | When shown |
|-----|---------|------------|
| 1 | Original frame | always |
| 2 | SigLIP self-attention heatmap | always |
| 3 | Action cross-attention heatmap | `--cross-attention` |
| 4 | Saliency / SmoothGrad overlay | `--gradient saliency` or `both` |
| 5 | Self-attention overlay | always |
| 6 | Co-attention overlay | `--cross-attention` |
| 7 | GradCAM overlay (SigLIP) | `--gradient gradcam` or `both` |
| 8 | GradCAM overlay (Connector) | `--gradcam-connector` |
| 9 | Language-conditional diff | `--language-diff` |

### Per-head grid

With `--show-heads`, the first frame gets a separate 12-head SigLIP grid:

![Per-head attention grid](assets/example_per_head.png)
*Look for specialization: some heads should track objects, gripper geometry, or broader scene structure.*

## Interpreting Results

### Self-attention

| Pattern | Interpretation |
|---------|----------------|
| Bright on gripper, object, and goal | good task-relevant visual focus |
| Bright on shelves, cables, or table texture | possible background shortcut |
| Uniform / diffuse everywhere | weak or unfocused visual features |
| Focus shifts sensibly over time | model is tracking task progression |

### Cross-attention

| Pattern | Interpretation |
|---------|----------------|
| Tight focus on gripper tip and target object | decoder is reading useful vision tokens |
| Diffuse over all vision tokens | decoder has not specialized well |
| Self-attn diffuse but cross-attn focused | decoder is filtering noisy encoder features |
| Self-attn focused but cross-attn diffuse | encoder is better than the decoder's use of it |

### Gradient attribution

| Pattern | Interpretation |
|---------|----------------|
| Saliency highlights object / gripper edges | action depends on relevant pixels |
| GradCAM agrees with attention | representation and causal signal align |
| Attention focused but saliency diffuse | model may look there without using it |
| Saliency spikes on irrelevant structure | likely shortcut or bias |

### Extended attribution checks

| Feature | What to look for |
|---------|-----------------|
| Per-step cross-attention | focus should sharpen over denoising steps |
| Connector GradCAM | should broadly agree with SigLIP GradCAM at coarser resolution |
| VLM layer GradCAM | later layers should become more task-specific |
| Vision vs. state | extreme imbalance can indicate one modality is ignored |
| Per-action-dim | different joints should not all attend to identical regions |
| Language diff | changing the instruction should move visual emphasis |

### Attention vs. gradient

| Case | Meaning |
|------|---------|
| High attention, low gradient | model represents the region but may not rely on it |
| Low attention, high gradient | subtle but causally important region |
| High attention, high gradient | strongest evidence of behavior-driving focus |

### Model internals report

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| Spectral alpha | 2-4 | 4-6 | >6 or <2 |
| Attention entropy | 0.10-0.80 | >0.80 | >0.95 or <0.10 |
| Head redundancy | <0.70 | >0.70 | >0.90 |

The report covers three attention components:

| Report component | Architecture operation |
|-----------------|------------------------|
| SigLIP Vision (12L, 12H) | self-attention inside the vision encoder |
| VLM+Expert Joint Self-Attn (16L, 15H) | joint prefill self-attention |
| Expert-to-VLM Cross-Attn (16L, 8H) | action decoding cross-attention |

### Split-device tip

If MPS backward is unstable or slow, run attention on MPS and gradients on CPU:

```bash
./run.sh --device mps --gradient both --gradient-device cpu
```

## CLI Reference

### General

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/defaults.yaml` | Load defaults from a YAML config |
| `--model` | `lerobot/smolvla_base` | HuggingFace model ID or local path |
| `--dataset` | `lerobot/svla_so101_pickplace` | LeRobot dataset ID or local path |
| `--episode` | `0` | Episode index |
| `--num-frames` | `8` | Number of sampled frames |
| `--image-key` | auto | Dataset image key override |
| `--image-map` | off | Explicit dataset-to-policy image key mapping |
| `--task` | dataset value | Override language instruction |
| `--output-dir` | `./outputs` | Output directory |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--save-individual` | `true` | Save per-frame overlays as separate files |
| `--export-data` | `true` | Save structured run data for the web viewer |
| `--no-export-data` | off | Disable structured run export |
| `--run-name` | timestamped | Override the generated run folder name |

### Attention

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `rollout` | `last-layer`, `rollout`, or `all-layers` |
| `--cross-attention` | `true` | Capture action-expert cross-attention |
| `--show-heads` | `true` | Save a per-head grid for frame 0 |
| `--raw-attention` | `false` | Skip positional baseline subtraction |
| `--attn-threshold` | `0.5` | Zero out low attention values after normalization |
| `--skip-attention` | `false` | Skip hook-based attention extraction and only run gradient features |

### Gradients and extended attribution

| Flag | Default | Description |
|------|---------|-------------|
| `--gradient` | off | `saliency`, `gradcam`, or `both` |
| `--gradient-device` | same as `--device` | Device for gradient computation |
| `--gradient-seed` | `42` | Fixed seed for reproducibility |
| `--smooth-grad` | `1` | SmoothGrad sample count |
| `--smooth-grad-sigma` | `0.15` | SmoothGrad noise std |
| `--per-step-cross-attention` | `false` | Save cross-attention per denoising step |
| `--gradcam-connector` | `false` | GradCAM on connector output |
| `--gradcam-vlm-layers` | off | GradCAM on selected VLM layers |
| `--vision-vs-state` | `false` | Compare vision vs state attribution |
| `--per-action-dim` | `false` | Per-action-dimension GradCAM |
| `--language-diff` | off | Compare attribution between two task prompts |

### Model internals

| Flag | Default | Description |
|------|---------|-------------|
| `--internals-only` | `false` | Run only the model internals report |
| `--with-internals` | `false` | Add the model internals report to a standard run |
| `--internals-frames` | `5` | Sampled frames for entropy / redundancy |
| `--entropy-warn` | `0.8` | Unfocused-head threshold |
| `--entropy-critical` | `0.95` | Dead-head threshold |
| `--entropy-low` | `0.1` | Collapsed-head threshold |
| `--redundancy-warn` | `0.7` | High-redundancy threshold |
| `--redundancy-critical` | `0.9` | Collapsed-redundancy threshold |

## Project Layout

```text
smolvla-inspect/
├── inspect_attention.py
├── smolvla_inspect/
│   ├── cli.py
│   ├── capture.py
│   ├── data.py
│   ├── export.py
│   ├── gradient.py
│   ├── heatmap.py
│   ├── internals.py
│   ├── serve.py
│   ├── viz.py
│   └── _compat.py
├── web/
│   ├── backend/
│   └── frontend/
├── assets/
├── configs/
├── docs/
├── clone-and-setup.sh
├── setup-gpu.sh
├── start_servers.sh
├── run.sh
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Gradient-based attribution
- [x] SmoothGrad
- [x] Extended attribution features
- [x] Config file support
- [x] Interactive web viewer
- [ ] Occlusion / perturbation sensitivity
- [ ] Representation probing
- [ ] Causal tracing / activation patching
- [ ] Temporal consistency analysis

## Note on FFmpeg

If you linked `ffmpeg@6` and want to switch back later:

```bash
brew unlink ffmpeg@6 && brew link ffmpeg
```
