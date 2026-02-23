# Testing Guide

## 1. Smoke test — defaults (simplest)

```bash
source .venv/bin/activate
./run.sh
```

**What to expect:**
- Downloads `lerobot/smolvla_base` model and `lerobot/svla_so101_pickplace` dataset from HuggingFace (first run only, ~few GB)
- Auto-detects device (MPS on Apple Silicon, CUDA on Linux/Windows, CPU otherwise)
- Processes 8 frames from episode 0
- Saves to `outputs/`:
  - `attention_grid_ep000.png` — 3-row grid (original / self-attention heatmap / overlay) x 8 columns
  - Individual frame PNGs (since `save_individual: true` in defaults)
- Heatmaps should show colored blobs over the image — bright areas = high attention

## 2. Attention rollout — deeper aggregation

```bash
./run.sh --method rollout
```

**What to expect:**
- Same output structure, but heatmaps use **rollout** (multiplies attention across all SigLIP layers accounting for residual connections)
- Heatmaps should look more refined/focused compared to `last-layer`, since they reflect cumulative information flow

## 3. Cross-attention — action decoder focus

```bash
./run.sh --cross-attention
```

**What to expect:**
- **Slower** — runs the full policy forward pass (not just vision encoder)
- Grid gains 2 extra rows per frame (5 rows total): Row 4 = cross-attention heatmap (hot colormap), Row 5 = dual-color overlay (blue = self-attn, red = cross-attn)
- Cross-attention heatmap shows what the **action expert** reads from the image — should be tighter/more focused than self-attention if the model is well-trained

## 4. Per-head grid — head specialization

```bash
./run.sh --show-heads
```

**What to expect:**
- An additional grid PNG showing attention from **each head separately** for the first frame
- Look for head specialization: different heads attending to gripper, object, background, etc.

## 5. Full combo — everything at once

```bash
./run.sh --method rollout --cross-attention --show-heads
```

**What to expect:**
- All of the above combined
- Slowest run, most comprehensive output
- This is the "everything works" confidence check

## 6. Quick sanity check with fewer frames

```bash
./run.sh --num-frames 2 --episode 0
```

Takes much less time, good for verifying the pipeline runs end-to-end.

## What "working" looks like

- **No errors/tracebacks** — script runs to completion
- **Output files appear** in `outputs/`
- **Heatmaps are non-uniform** — if every heatmap is perfectly flat/solid, something went wrong with attention capture
- **Grid image opens and shows** 3 (or 5 with `--cross-attention`) rows per frame
- **Overlay images** show colored attention blobs on top of recognizable robot scene images

## Red flags

| Symptom | Likely cause |
|---------|-------------|
| `FFmpeg`/`TorchCodec` error on dataset load | FFmpeg 6 not installed or `DYLD_LIBRARY_PATH` not set — use `./run.sh` |
| All heatmaps are uniform gray | Attention hooks didn't capture weights — check for SDPA/Flash fallback warnings |
| `CUDA out of memory` | Use `--device cpu` or `--device mps` (device is auto-detected by default) |
| `KeyError` on image key | Dataset doesn't match expected camera key — check dataset schema |
