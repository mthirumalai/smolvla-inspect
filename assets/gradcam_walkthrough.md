# GradCAM Walkthrough — Worked Example

A step-by-step numerical example of how GradCAM works inside `smolvla-inspect`,
using a tiny 4-patch, 3-channel toy case instead of the real 1024-patch, 768-channel tensors.

---

## Setup

- **Target layer**: last SigLIP encoder layer (layer 12 in the real model)
- **Two hooks** registered on that layer:
  - **Forward hook** → captures activations **A** when the forward pass flows through
  - **Backward hook** → captures gradients **dA** when `.backward()` flows back through
- **Backprop target**: `actions[:, 0, :].sum()` — the sum of the predicted action vector (a scalar)

---

## Where the Two Tensors Come From

There is only **one** forward pass and **one** backward pass. The two hooks fire at different times on the same layer:

```
                    ONE FORWARD PASS
                    ════════════════

  Image (512×512×3)
        │
        ▼
  ┌─────────────┐
  │ SigLIP L1   │
  ├─────────────┤
  │ SigLIP L2   │
  ├─────────────┤
  │   ...       │
  ├─────────────┤
  │ SigLIP L12  │──── fwd_hook fires! ──── saves A (1, 1024, 768)
  └──────┬──────┘
         │
         ▼
  Connector → VLM → Expert → actions[:, 0, :].sum() = scalar
```

```
                    ONE BACKWARD PASS
                    ═════════════════

  scalar.backward()
         │
         ▼
  Expert ← VLM ← Connector
         │
         ▼
  ┌─────────────┐
  │ SigLIP L12  │──── bwd_hook fires! ──── saves dA (1, 1024, 768)
  ├─────────────┤
  │   ...       │
  └─────────────┘
```

Same layer, two moments in time:
- Forward pass captures **what was there** (A)
- Backward pass captures **how much the output cares** (dA)

---

## The Two Tensors (Toy Example)

4 patches, 3 channels (real: 1024 patches, 768 channels):

### A — Activations (captured by forward hook)

What the encoder computed. Each channel is a learned feature detector.

```
              ch0     ch1     ch2
            ┌───────┬───────┬───────┐
  patch 0   │  0.9  │  0.1  │  0.2  │   ← "gripper" patch
  patch 1   │  0.1  │  0.8  │  0.3  │   ← "red object" patch
  patch 2   │  0.0  │  0.1  │  0.7  │   ← "table edge" patch
  patch 3   │  0.1  │  0.0  │  0.1  │   ← "background" patch
            └───────┴───────┴───────┘
```

### dA — Gradients (captured by backward hook)

How much the predicted action changes if each activation value increases by a tiny amount.

```
              ch0     ch1     ch2
            ┌───────┬───────┬───────┐
  patch 0   │  0.5  │  0.0  │  0.1  │
  patch 1   │  0.3  │  0.6  │  0.0  │
  patch 2   │  0.1  │  0.2  │  0.0  │
  patch 3   │  0.0  │  0.0  │  0.0  │
            └───────┴───────┴───────┘
```

---

## Step 1: Which Channels Matter Most?

Average dA across all patches (down the rows) to get one importance weight per channel:

```python
alpha = dA.mean(dim=1)  # → (1, 768) in real code
```

```
              ch0     ch1     ch2
            ┌───────┬───────┬───────┐
  patch 0   │  0.5  │  0.0  │  0.1  │
  patch 1   │  0.3  │  0.6  │  0.0  │
  patch 2   │  0.1  │  0.2  │  0.0  │
  patch 3   │  0.0  │  0.0  │  0.0  │
            └───────┴───────┴───────┘
                │       │       │
               mean    mean    mean
                │       │       │
                ▼       ▼       ▼

  alpha =  [ 0.225,  0.200,  0.025 ]
```

**Interpretation:**
- ch0 (0.225) — most important to the action
- ch1 (0.200) — also important
- ch2 (0.025) — barely matters

---

## Step 2: Weight Activations by Channel Importance

Multiply each row of A by alpha, then sum across channels:

```python
cam = (alpha * A).sum(dim=-1)  # → (1, 1024) in real code
```

```
  alpha * A    (element-wise multiply each row by alpha)

              ch0           ch1           ch2
            ┌─────────────┬─────────────┬─────────────┐
  patch 0   │ 0.9 × 0.225 │ 0.1 × 0.200│ 0.2 × 0.025│
            │   = 0.203   │   = 0.020   │   = 0.005   │
  patch 1   │ 0.1 × 0.225 │ 0.8 × 0.200│ 0.3 × 0.025│
            │   = 0.023   │   = 0.160   │   = 0.008   │
  patch 2   │ 0.0 × 0.225 │ 0.1 × 0.200│ 0.7 × 0.025│
            │   = 0.000   │   = 0.020   │   = 0.018   │
  patch 3   │ 0.1 × 0.225 │ 0.0 × 0.200│ 0.1 × 0.025│
            │   = 0.023   │   = 0.000   │   = 0.003   │
            └─────────────┴─────────────┴─────────────┘

  .sum(dim=-1)   (sum across channels per patch)

  patch 0:  0.203 + 0.020 + 0.005  =  0.228   ← HIGH (gripper)
  patch 1:  0.023 + 0.160 + 0.008  =  0.191   ← HIGH (red object)
  patch 2:  0.000 + 0.020 + 0.018  =  0.038   ← low  (table edge)
  patch 3:  0.023 + 0.000 + 0.003  =  0.026   ← low  (background)
```

---

## Step 3: ReLU and Reshape

```python
cam = torch.relu(cam)                          # discard negative values
cam = cam.reshape(grid_h, grid_w)              # back to spatial grid
cam = cam / (cam.max() + 1e-8)                 # normalize to [0, 1]
```

```
  After ReLU (no change here, all positive):

  cam = [0.228, 0.191, 0.038, 0.026]

  Reshape to 2×2 grid:

  ┌───────┬───────┐
  │ 0.228 │ 0.191 │
  │gripper│  obj  │
  ├───────┼───────┤
  │ 0.038 │ 0.026 │
  │ table │  bg   │
  └───────┴───────┘

  Normalize to [0, 1]:

  ┌───────┬───────┐
  │ 1.000 │ 0.838 │    ██████   █████
  ├───────┼───────┤
  │ 0.167 │ 0.114 │    ░░       ░
  └───────┴───────┘
```

The gripper and object patches are bright — they are the regions that most
influence the predicted action. The table and background are dim — the network
doesn't care about them for deciding what to do next.

In the real code this is a 32×32 grid (SigLIP: image_size=512, patch_size=16)
that gets upsampled to 512×512 and blended onto the original image with the
`magma` colormap.

---

## Code Reference

The implementation lives in `smolvla_inspect/gradient.py:175-269` (`compute_gradcam_map`).

Key lines:
- Hook registration: lines 211-212
- Forward pass with grad: line 222 (`_run_forward_with_grad`)
- Backward: line 223 (`action_scalar.backward()`)
- GradCAM formula: lines 233-237
- Reshape to grid: lines 242-253
- Hook cleanup: lines 268-269
