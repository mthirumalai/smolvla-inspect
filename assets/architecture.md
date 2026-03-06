# SmolVLA Architecture Diagrams

Reference diagrams for understanding the SmolVLA model structure and what
the `--internals-only` / `--with-internals` report measures.

---

## 1. High-level data flow

```
  Camera Images (512x512)       Task String            Robot State     Noisy Actions
  +--------+ +--------+         "pick up the           (joint          (from flow
  |  side  | |   up   |          red block"             positions)       matching)
  +---+----+ +---+----+              |                     |               |
      |          |                   |                     |               |
      v          v                   v                     v               v
  +------------------+     +---------------+     +---------------+  +---------------+
  | SigLIP Vision    |     | Tokenizer     |     | state_proj    |  | action_in_proj|
  | Encoder          |     |               |     | (Linear)      |  | (Linear)      |
  | (frozen)         |     |               |     |               |  | + time MLP    |
  | 12L, 12H         |     |               |     | trainable     |  | trainable     |
  +--------+---------+     +-------+-------+     +-------+-------+  +-------+-------+
           |                       |                     |                  |
     1024 patches            token embeds            1 token          10 tokens
           |                       |                     |                  |
           v                       |                     |                  |
  +-------------------+            |                     |                  |
  | Connector         |            |                     |                  |
  | (pixel shuffle)   |            |                     |                  |
  | trainable         |            |                     |                  |
  +--------+----------+            |                     |                  |
           |                       |                     |                  |
     64 tokens                     |                     |                  |
           |                       |                     |                  |
           +-----------+-----------+                     |                  |
                       |                                 |                  |
                       v                                 |                  |
           +-----------------------+                     |                  |
           | PREFIX SEQUENCE       |                     |                  |
           | [vision] [language]   +<--------------------+                  |
           | [state]               |     VLM dim (960)                     |
           +-----------+-----------+                                       |
                       |                                                   |
                       |                                 +-----------------+
                       |                                 |
                       v                                 v
           +---------------------------------------------------+
           |         SmolVLMWithExpert (16 layers)               |
           |                                                     |
           |   PREFIX  ------------------->  VLM Text Model      |
           |   (vision+lang+state)            (frozen)           |
           |                                  960-dim, 15 heads  |
           |                                                     |
           |   SUFFIX  ------------------->  Action Expert       |
           |   (noisy actions+time)           (trainable)        |
           |                                  480-dim, 8 heads   |
           |                                                     |
           |         (see attention diagrams below)              |
           +--------------------------+--------------------------|
                                      |
                                      |  Expert output (last 10 tokens)
                                      v
                          +-------------------+
                          | action_out_proj   |
                          | (Linear)          |
                          | trainable         |
                          +---------+---------+
                                    |
                                    v
                          Predicted Actions
                          (10 steps x 6 DOF)
```

---

## 2. The three attention components

These are the three distinct attention operations that the model internals report
measures for entropy and head redundancy.

### A. SigLIP Vision (12 layers, 12 heads)

Standard self-attention inside the vision encoder. Happens before anything
enters SmolVLMWithExpert. Image patches attend to other image patches.

```
  patch_1  patch_2  patch_3  ...  patch_1024
     |        |        |              |
     v        v        v              v
  +----------------------------------------------+
  | SigLIP Self-Attention                         |
  |                                               |
  | Each patch can see all patches.               |
  |                                               |
  | Q <-- patches        K <-- patches            |
  | (1024 x 1024 attention matrix)                |
  +----------------------------------------------+

  Report label: "SigLIP Vision (12L, 12H)"
```

### B. VLM+Expert Joint Self-Attention (16 layers, 15 heads)

Runs during prefill (initial encoding). VLM and Expert tokens are
concatenated into one sequence and attend to each other in a single
attention call.

```
  VLM tokens (prefix)           Expert tokens (suffix)
  [vision][lang][state]          [action+time]
        |                              |
        v                              v
  +-------------+               +-------------+
  | VLM layer   |               | Expert layer|
  | q/k/v proj  |               | q/k/v proj  |
  +------+------+               +------+------+
         |                             |
         +-------------+---------------+
                       |
                       v
                torch.cat(dim=seq)
                       |
                       v
  +----------------------------------------------+
  | JOINT Self-Attention                          |
  |                                               |
  | All tokens see all tokens:                    |
  |   vision  <-->  vision                        |
  |   vision  <-->  language                      |
  |   vision  <-->  action                        |
  |   action  <-->  language                      |
  |   action  <-->  action                        |
  |   ... etc                                     |
  |                                               |
  | Q <-- [VLM+Expert]     K <-- [VLM+Expert]     |
  | Q_len == K_len  (self-attention)              |
  +----------------------------------------------+

  Report label: "VLM+Expert Joint Self-Attn (16L, 15H)"
```

### C. Expert-to-VLM Cross-Attention (16 layers, 8 heads)

Runs during generation (autoregressive action decoding). The Expert
generates queries from its own hidden states, but keys and values come
from the VLM's cached representations, re-projected through the Expert's
k_proj and v_proj. Repeats 10 times (one per action token).

```
  VLM KV Cache                       Expert hidden states
  (built during prefill)             (current action token)
        |                                    |
        |                                    v
        |                            +---------------+
        |                            | Expert q_proj |
        |                            +-------+-------+
        |                                    |
        |     re-projected through           |
        |     Expert k_proj / v_proj         |
        v                                    |
  +-------------+                            |
  | Expert K, V |<-- VLM cache               |
  | (from VLM)  |   re-projected             |
  +------+------+                            |
         |                                   |
         +---------------+-------------------+
                         |
                         v
  +----------------------------------------------+
  | CROSS-Attention                               |
  |                                               |
  | Expert reads VLM:                             |
  |   action --> vision   (what to grab?)         |
  |   action --> language  (what was asked?)      |
  |   action --> state     (current pose?)        |
  |                                               |
  | VLM does NOT read Expert here.                |
  |                                               |
  | Q <-- Expert tokens (few)                     |
  | K <-- VLM prefix (many)                       |
  | Q_len != K_len  (cross-attention)             |
  +----------------------------------------------+

  Report label: "Expert-to-VLM Cross-Attn (16L, 8H)"
```

---

## 3. Execution timeline

```
  IMAGE ENCODING            PREFILL                   GENERATION
  +--------------+    +------------------+    +------+------+     +------+
  | SigLIP       |    | Joint Self-Attn  |    | XA   | XA   |    | XA   |
  | 12 layers    +--->| 16 layers        +--->| #1   | #2   |...>| #10  |
  | (A)          |    | (B)              |    |      |      |    |      |
  +--------------+    +------------------+    +------+------+    +------+
                                              <--- 10 action steps --->

  Report:              Report:                 Report:
  "SigLIP Vision"      "VLM+Expert             "Expert-to-VLM
                        Joint Self-Attn"        Cross-Attn"
  12L x 12H            16L x 15H               16L x 8H
                                                (x10 steps, averaged)
```

---

## 4. What the model internals report measures

### Section 1: Weight Spectral Analysis (alpha)

Analyzes weight matrices directly (no data needed). Each transformer layer
contains multiple weight matrices (q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj). A power-law is fit to each matrix's
singular values; the exponent (alpha) indicates training quality.

```
  +----------------------------------+-------------+---------------------+
  | Component                        | Wt Matrices | What it is          |
  +----------------------------------+-------------+---------------------+
  | Expert (trainable)               | 112 = 16x7  | Action decoder      |
  | VLM Text Model (frozen)          | 113 = 16x7+ | Language model      |
  | Vision Encoder (frozen)          |  74 = 12x~6 | SigLIP              |
  | Connector (trainable)            |   1          | Pixel shuffle       |
  | Projections (trainable)          | too small    | state/action linear |
  +----------------------------------+-------------+---------------------+

  Healthy: alpha 2-4     Undertrained: 4-6     Severe: >6
```

### Section 2: Attention Entropy

Requires a forward pass with real data. Measures how spread out each
attention head's focus is (normalized by log of sequence length).

```
  Collapsed: <0.10     Healthy: 0.10-0.80     Unfocused: >0.80     Dead: >0.95
```

### Section 3: Head Redundancy

Same forward pass data. Measures pairwise cosine similarity between
flattened head attention patterns within each layer. Heads should learn
different patterns.

```
  Diverse: <0.70       High redundancy: >0.70       Collapsed: >0.90
```

---

## 5. Trainable vs frozen components

```
  FROZEN (pretrained, not updated during fine-tuning)
  +----------------------------------------------------+
  | SigLIP Vision Encoder                               |
  | 12 layers, 12 heads, 74 weight matrices             |
  +----------------------------------------------------+
  | VLM Text Model (SmolLM2)                            |
  | 16 layers, 15 heads, 113 weight matrices            |
  +----------------------------------------------------+

  TRAINABLE (learned during fine-tuning)
  +----------------------------------------------------+
  | Action Expert                                       |
  | 16 layers, 8 heads, 112 weight matrices             |
  +----------------------------------------------------+
  | Connector (pixel shuffle)                           |
  | 1 weight matrix                                     |
  +----------------------------------------------------+
  | state_proj, action_in_proj, action_out_proj         |
  | (too small for spectral analysis)                   |
  +----------------------------------------------------+
```
