#!/usr/bin/env bash
# Fast test runner — re-run individual counterfactual tests or regenerate
# visualizations on an existing run directory WITHOUT the full pipeline.
#
# Usage:
#   ./run_test.sh <run_dir> <test_name> [--param key=value ...] [--viz-only]
#
# Examples:
#   # Re-run background_substitution on the finetuned model
#   ./run_test.sh outputs/mthirumalai/finetuned_model background_substitution
#
#   # Re-run distractor with custom params
#   ./run_test.sh outputs/mthirumalai/finetuned_model distractor_insertion \
#       --param position='[200,200]' --param distractor_size=60
#
#   # Re-run task_string_swap with a different replacement task
#   ./run_test.sh outputs/mthirumalai/finetuned_model task_string_swap \
#       --param replacement_task='pick up the red cube'
#
#   # Just regenerate the visualization from existing result (no model needed)
#   ./run_test.sh outputs/mthirumalai/finetuned_model task_string_swap --viz-only
#
#   # List available counterfactual tests
#   ./run_test.sh --list

set -e
cd "$(dirname "$0")"

# macOS: Homebrew ffmpeg@6 for TorchCodec compatibility
FFMPEG6_LIB="/opt/homebrew/opt/ffmpeg@6/lib"
if [[ -d "$FFMPEG6_LIB" && -f "$FFMPEG6_LIB/libavutil.58.dylib" ]]; then
  export DYLD_LIBRARY_PATH="${FFMPEG6_LIB}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

# Linux: Ensure system FFmpeg 4.x libs load first
SYS_FFMPEG="/lib/x86_64-linux-gnu"
if [[ -f "$SYS_FFMPEG/libavutil.so.56" ]]; then
  export LD_LIBRARY_PATH="${SYS_FFMPEG}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec python3 run_single_counterfactual.py "$@"
