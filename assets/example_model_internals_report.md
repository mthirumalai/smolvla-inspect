# Model Health Report

## 1. Weight Spectral Analysis (alpha)

Fits a power-law to each weight matrix's singular values. Alpha measures how well-trained a layer is — values of 2-4 indicate strong correlation structure learned during training. High alpha means the layer hasn't learned enough structure; low alpha means overcorrelation.

> Healthy: 2-4 | Undertrained: 4-6 | Overcorrelated: <2 | Severe: >6

| Component | Wt Matrices | Mean alpha | Min alpha | Max alpha | Status |
|-----------|------------:|-----------:|----------:|----------:|--------|
| Expert (trainable) | 112 | 10.16 | 3.62 | 33.49 | CRITICAL severely undertrained |
| Connector (trainable) | 1 | 6.77 | 6.77 | 6.77 | CRITICAL severely undertrained |
| Projection/state_proj (trainable) | -- | N/A | N/A | N/A | too small |
| Projection/action_in_proj (trainable) | -- | N/A | N/A | N/A | too small |
| Projection/action_out_proj (trainable) | -- | N/A | N/A | N/A | too small |
| Vision Encoder (frozen) | 74 | 5.31 | 1.54 | 22.76 | WARN undertrained |
| VLM Text Model (frozen) | 113 | 6.39 | 1.38 | 25.30 | CRITICAL severely undertrained |

## 2. Attention Entropy (fraction of max)

Measures how spread out each attention head's focus is. Low entropy means the head attends to very few tokens (collapsed/dead). High entropy means the head spreads attention nearly uniformly (unfocused). Healthy heads are selective but not degenerate — attending to a meaningful subset.

> Collapsed: <0.10 | Healthy: 0.10-0.80 | Unfocused: >0.80 | Dead: >0.95


### SigLIP Vision (12L, 12H)

| Layer | Mean Ent | Min Ent | Max Ent | Status |
|------:|---------:|--------:|--------:|--------|
| 0 | 0.7926 | 0.6119 | 0.8551 | OK healthy |
| 1 | 0.6408 | 0.0514 | 0.9715 | OK healthy |
| 2 | 0.6626 | 0.1127 | 0.9057 | OK healthy |
| 3 | 0.6226 | 0.3508 | 0.9186 | OK healthy |
| 4 | 0.7288 | 0.5579 | 0.8916 | OK healthy |
| 5 | 0.7208 | 0.5479 | 0.8824 | OK healthy |
| 6 | 0.6469 | 0.5850 | 0.8121 | OK healthy |
| 7 | 0.6925 | 0.6098 | 0.8261 | OK healthy |
| 8 | 0.6698 | 0.5974 | 0.7539 | OK healthy |
| 9 | 0.6402 | 0.5778 | 0.7571 | OK healthy |
| 10 | 0.6675 | 0.5774 | 0.7822 | OK healthy |
| 11 | 0.7118 | 0.6013 | 0.8185 | OK healthy |

### VLM+Expert Joint Self-Attn (16L, 15H)

| Layer | Mean Ent | Min Ent | Max Ent | Status |
|------:|---------:|--------:|--------:|--------|
| 0 | 0.8687 | 0.3922 | 0.9836 | WARN unfocused |
| 1 | 0.6089 | 0.2493 | 0.9195 | OK healthy |
| 2 | 0.6568 | 0.2870 | 0.8795 | OK healthy |
| 3 | 0.6717 | 0.4935 | 0.8078 | OK healthy |
| 4 | 0.5567 | 0.2679 | 0.8070 | OK healthy |
| 5 | 0.5222 | 0.2390 | 0.8364 | OK healthy |
| 6 | 0.6408 | 0.3294 | 0.8395 | OK healthy |
| 7 | 0.7283 | 0.3072 | 0.9092 | OK healthy |
| 8 | 0.7180 | 0.5422 | 0.8647 | OK healthy |
| 9 | 0.7230 | 0.4175 | 0.8630 | OK healthy |
| 10 | 0.6639 | 0.4483 | 0.8663 | OK healthy |
| 11 | 0.8324 | 0.7138 | 0.9104 | WARN unfocused |
| 12 | 0.7782 | 0.6291 | 0.9170 | OK healthy |
| 13 | 0.7333 | 0.5639 | 0.8677 | OK healthy |
| 14 | 0.7876 | 0.5989 | 0.9295 | OK healthy |
| 15 | 0.8046 | 0.6538 | 0.8607 | WARN unfocused |

### Expert-to-VLM Cross-Attn (16L, 8H)

| Layer | Mean Ent | Min Ent | Max Ent | Status |
|------:|---------:|--------:|--------:|--------|
| 0 | 0.8044 | 0.4733 | 0.9298 | WARN unfocused |
| 1 | 0.5660 | 0.2890 | 0.7517 | OK healthy |
| 2 | 0.8031 | 0.5559 | 0.9331 | WARN unfocused |
| 3 | 0.4855 | 0.3421 | 0.6717 | OK healthy |
| 4 | 0.6554 | 0.2899 | 0.9049 | OK healthy |
| 5 | 0.6032 | 0.3654 | 0.7872 | OK healthy |
| 6 | 0.7729 | 0.4828 | 0.9374 | OK healthy |
| 7 | 0.5886 | 0.3391 | 0.7896 | OK healthy |
| 8 | 0.7946 | 0.6051 | 0.9377 | OK healthy |
| 9 | 0.5766 | 0.3948 | 0.7962 | OK healthy |
| 10 | 0.8457 | 0.6762 | 0.9274 | WARN unfocused |
| 11 | 0.6577 | 0.4307 | 0.8673 | OK healthy |
| 12 | 0.7770 | 0.5937 | 0.8953 | OK healthy |
| 13 | 0.7252 | 0.5465 | 0.8845 | OK healthy |
| 14 | 0.7860 | 0.5844 | 0.9007 | OK healthy |
| 15 | 0.7010 | 0.3601 | 0.8994 | OK healthy |

## 3. Head Redundancy (cosine similarity)

Measures how similar the attention heads are to each other within each layer. Each layer has multiple heads that should learn different patterns (e.g., one head for spatial relations, another for color). High similarity means heads are redundant — wasted capacity. Collapsed means nearly identical heads.

> Diverse: <0.70 | High: >0.70 | Collapsed: >0.90


### SigLIP Vision (12L, 12H)

| Layer | Mean Sim | Max Sim | Status |
|------:|---------:|--------:|--------|
| 0 | 0.4325 | 0.8165 | OK diverse |
| 1 | 0.2093 | 0.6849 | OK diverse |
| 2 | 0.2812 | 0.6402 | OK diverse |
| 3 | 0.2243 | 0.6496 | OK diverse |
| 4 | 0.3254 | 0.5524 | OK diverse |
| 5 | 0.3247 | 0.5485 | OK diverse |
| 6 | 0.3435 | 0.5408 | OK diverse |
| 7 | 0.4119 | 0.6366 | OK diverse |
| 8 | 0.5707 | 0.8143 | OK diverse |
| 9 | 0.6851 | 0.8987 | OK diverse |
| 10 | 0.6697 | 0.9205 | OK diverse |
| 11 | 0.4863 | 0.9259 | OK diverse |

### VLM+Expert Joint Self-Attn (16L, 15H)

| Layer | Mean Sim | Max Sim | Status |
|------:|---------:|--------:|--------|
| 0 | 0.5098 | 0.9331 | OK diverse |
| 1 | 0.2914 | 0.8093 | OK diverse |
| 2 | 0.3013 | 0.7712 | OK diverse |
| 3 | 0.1861 | 0.5591 | OK diverse |
| 4 | 0.4979 | 0.8321 | OK diverse |
| 5 | 0.5328 | 0.8697 | OK diverse |
| 6 | 0.6398 | 0.9422 | OK diverse |
| 7 | 0.4894 | 0.9447 | OK diverse |
| 8 | 0.5196 | 0.8909 | OK diverse |
| 9 | 0.5996 | 0.9147 | OK diverse |
| 10 | 0.4213 | 0.7618 | OK diverse |
| 11 | 0.6982 | 0.8499 | OK diverse |
| 12 | 0.5257 | 0.8012 | OK diverse |
| 13 | 0.5006 | 0.8399 | OK diverse |
| 14 | 0.4711 | 0.8028 | OK diverse |
| 15 | 0.4259 | 0.7048 | OK diverse |

### Expert-to-VLM Cross-Attn (16L, 8H)

| Layer | Mean Sim | Max Sim | Status |
|------:|---------:|--------:|--------|
| 0 | 0.4843 | 0.9573 | OK diverse |
| 1 | 0.1499 | 0.8986 | OK diverse |
| 2 | 0.4202 | 0.8331 | OK diverse |
| 3 | 0.1925 | 0.9334 | OK diverse |
| 4 | 0.4150 | 0.9433 | OK diverse |
| 5 | 0.3017 | 0.7776 | OK diverse |
| 6 | 0.4690 | 0.8690 | OK diverse |
| 7 | 0.3699 | 0.9330 | OK diverse |
| 8 | 0.6153 | 0.9186 | OK diverse |
| 9 | 0.2292 | 0.8986 | OK diverse |
| 10 | 0.5643 | 0.8731 | OK diverse |
| 11 | 0.4494 | 0.8642 | OK diverse |
| 12 | 0.6087 | 0.8594 | OK diverse |
| 13 | 0.3594 | 0.8417 | OK diverse |
| 14 | 0.6707 | 0.8954 | OK diverse |
| 15 | 0.5495 | 0.8975 | OK diverse |