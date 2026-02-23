# smolvla-inspect

See what SmolVLA's vision encoder and action expert are looking at when the model predicts robot actions.

![Example attention grid](assets/example_grid.png)
*5-row attention grid for a pick-and-place episode. Row 1: original frames. Row 2: SigLIP self-attention heatmap. Row 3: self-attention overlay. Row 4: action cross-attention heatmap. Row 5: co-attention (self x cross) overlay in cyan.*

---

## What this does

SmolVLA is a **vision-language-action** policy: it takes camera images and a language instruction, then outputs robot actions. This tool **visualizes where the model looks** by extracting attention maps from two places:

1. **SigLIP vision encoder** (self-attention) -- which image patches the encoder considers important during feature extraction
2. **Action expert** (cross-attention) -- which image regions the action decoder actually reads when predicting actions

That lets you check whether the model attends to task-relevant regions (gripper, object, goal) or background (walls, table texture) -- useful for debugging overfitting or distribution shift.

**Input:** A pretrained or fine-tuned SmolVLA policy + a LeRobot dataset (e.g. episodes of pick-and-place).
**Output:** A multi-row grid PNG per episode, optional per-frame PNGs, and an optional per-head attention grid.

---

## How it works

![Architecture and attention-to-heatmap pipeline](assets/how_it_works_architecture.png)
*Left: Where the vision encoder lives and where we hook to capture attention. Right: How attention weights become a spatial heatmap.*

### Pipeline

1. **Load model and dataset** -- loads a SmolVLA policy (e.g. `lerobot/smolvla_base`) and a LeRobot dataset. The dataset provides image sequences; the script uses images as input and runs inference.

2. **Capture self-attention from SigLIP** -- the vision encoder (SigLIP ViT, 12 layers, 12 heads) splits each image into patches (32x32 grid for 512px images with 16px patches) and runs self-attention. Forward hooks on the attention layers capture the weight matrices.

3. **Aggregate across layers** (`--method`):
   - `last-layer` -- uses only the final encoder layer
   - `rollout` (default) -- multiplies attention across all layers with residual connections, giving a more complete picture of information flow
   - `all-layers` -- keeps each layer separately

4. **Capture cross-attention** (`--cross-attention`, on by default) -- SmolVLA's VLM builds a KV cache from the prefix (vision + language + state tokens). The action expert queries that cache. The script monkey-patches `eager_attention_forward()` on the expert layers to intercept the softmax attention when expert Q attends to prefix K. Only columns corresponding to vision tokens are kept, giving a heatmap of which image regions the action decoder reads.

5. **Turn attention into spatial heatmaps** -- patch-level importance scores are reshaped into a 2D grid, upsampled with bilinear interpolation to image size, and normalized to [0, 1].

6. **Visualize** -- the output grid has up to 5 rows per frame:

| Row | Content | Colormap |
|-----|---------|----------|
| 1 | Original frame | -- |
| 2 | SigLIP self-attention heatmap | jet (blue-to-red) |
| 3 | Self-attention overlay on frame | jet |
| 4 | Action cross-attention heatmap | Greens |
| 5 | Co-attention overlay (self x cross) | cyan (black-cyan-white) |

Rows 4-5 only appear when cross-attention is enabled. The co-attention overlay multiplies self-attention and cross-attention element-wise, highlighting regions that are **both** visually salient and action-relevant.

### Per-head grid

With `--show-heads`, a separate grid shows each of the 12 SigLIP attention heads individually for the first frame:

![Per-head attention grid](assets/example_per_head.png)
*Each subplot is one attention head. Look for specialization -- e.g. one head tracking the gripper, another tracking the object.*

---

## Setup

**Requirements:** Python 3.10+, and FFmpeg 4-7 for video decoding (LeRobot uses TorchCodec). On macOS, Homebrew's default `ffmpeg` is often v8; install FFmpeg 6:

```bash
brew install ffmpeg@6
```

Then:

```bash
cd smolvla-inspect
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run

Use `run.sh` so TorchCodec finds FFmpeg 6's libs:

```bash
source .venv/bin/activate
./run.sh
```

Or set the library path yourself:

```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@6/lib:$DYLD_LIBRARY_PATH"
python inspect_attention.py
```

### Examples

```bash
# Default config: rollout method, cross-attention enabled, per-head grid enabled
./run.sh

# Your fine-tuned model
./run.sh --model path/to/finetuned_checkpoint --dataset path/to/dataset

# More frames, specific episode
./run.sh --episode 3 --num-frames 12

# Last-layer only (faster, no rollout)
./run.sh --method last-layer

# Skip cross-attention (faster, 3-row grid only) — edit configs/defaults.yaml:
#   cross_attention: false

# Raw attention without positional baseline subtraction
./run.sh --raw-attention

# Explicit device override (auto-detected by default: mps > cuda > cpu)
./run.sh --device cuda
```

Results land in `outputs/`.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `lerobot/smolvla_base` | HuggingFace model ID or local path |
| `--dataset` | `lerobot/svla_so101_pickplace` | LeRobot dataset ID or local path |
| `--episode` | `0` | Episode index to visualize |
| `--num-frames` | `8` | Number of frames to sample |
| `--image-key` | auto-detected | Dataset image key override |
| `--output-dir` | `./outputs` | Output directory |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--save-individual` | `true` | Save each frame as a separate PNG |
| `--method` | `rollout` | `last-layer`, `rollout`, or `all-layers` |
| `--cross-attention` | `true` | Capture action-expert cross-attention |
| `--show-heads` | `true` | Save per-head attention grid for first frame |
| `--raw-attention` | `false` | Skip positional baseline subtraction |

Defaults can be changed in `configs/defaults.yaml`.

---

## What to look for

### Self-attention (SigLIP vision encoder -- rows 2-3)

| Attention pattern | Interpretation |
|-------------------|----------------|
| Bright on gripper + object + goal | **Healthy** -- model attends to task-relevant regions |
| Bright on shelves, cables, table grain | **Background overfitting** -- model may be using scene cues |
| Uniform / diffuse everywhere | Model may not have learned focused visual features yet |
| Shifts from background to object across frames | Model is tracking the task over time (good sign) |

### Cross-attention (action expert -- rows 4-5)

| Attention pattern | Interpretation |
|-------------------|----------------|
| Tight focus on gripper tip + target object | **Healthy** -- action decoder reads exactly what it needs |
| Diffuse across all vision tokens | Decoder hasn't specialized; may predict generic actions |
| Self-attn diffuse but cross-attn focused | Decoder learned to select useful tokens despite a noisy encoder |
| Self-attn focused but cross-attn diffuse | Encoder features are good but the decoder doesn't exploit them |

### Co-attention (row 5)

The cyan overlay highlights regions where **both** the vision encoder and the action expert agree something is important. Bright cyan = high self-attention AND high cross-attention. This is the strongest signal for task-relevant regions.

### Per-head patterns

Look for heads that specialize: one head tracking the gripper, another tracking the object, another attending globally. Specialization is a sign of a well-trained encoder. Heads that all look identical suggest the model hasn't learned diverse attention strategies.

---

## Project layout

```
smolvla-inspect/
├── inspect_attention.py      # All logic: model loading, hooks, heatmaps, visualization
├── assets/
│   ├── how_it_works_architecture.png
│   ├── example_grid.png
│   └── example_per_head.png
├── configs/
│   └── defaults.yaml         # Default CLI values (model, dataset, method, flags)
├── docs/
│   ├── ELI5.md               # Plain-language explanation of the interpretability approach
│   └── TESTING.md            # CLI test commands and expected output
├── outputs/                  # Generated images (gitignored)
├── run.sh                    # Wrapper that sets FFmpeg lib path
├── requirements.txt
└── README.md
```

---

## Note on FFmpeg

If you installed `ffmpeg@6` and linked it (`brew link --overwrite ffmpeg@6`), your default `ffmpeg` is now 6.x. To switch back later: `brew unlink ffmpeg@6 && brew link ffmpeg`.
