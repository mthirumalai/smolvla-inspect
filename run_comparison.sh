#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

DATASET="mthirumalai/so101.pnp.1"
BASE_MODEL="lerobot/smolvla_base"
FINETUNED_MODEL="mthirumalai/so101.pnp.1.policy.s1"
OUTPUT_DIR="./outputs/mthirumalai"

BASE_RUN="base_model"
FINETUNED_RUN="finetuned_model"

# ── Step 1/3: Full inspection + diagnostics — base model ────
echo "=== Step 1/3: Inspect + diagnose — base model ==="
./run.sh diagnose \
  --model "$BASE_MODEL" \
  --dataset "$DATASET" \
  --run-name "$BASE_RUN" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda

# ── Step 2/3: Full inspection + diagnostics — fine-tuned model
echo "=== Step 2/3: Inspect + diagnose — fine-tuned model ==="
./run.sh diagnose \
  --model "$FINETUNED_MODEL" \
  --dataset "$DATASET" \
  --run-name "$FINETUNED_RUN" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda

# ── Step 3/3: Compare both runs ─────────────────────────────
echo "=== Step 3/3: Compare runs ==="
./run.sh compare \
  --runs "$OUTPUT_DIR/$BASE_RUN" "$OUTPUT_DIR/$FINETUNED_RUN" \
  --labels "Base SmolVLA" "Fine-tuned"

echo "=== Done ==="
