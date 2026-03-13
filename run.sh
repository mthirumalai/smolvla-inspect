#!/usr/bin/env bash
# Run the inspector with FFmpeg libs visible to TorchCodec.
# TorchCodec supports FFmpeg 4–7; macOS Homebrew often has FFmpeg 8 (libavutil.60).
# If you have ffmpeg@6 installed, we point the loader at it so video decoding works.

set -e
cd "$(dirname "$0")"

# macOS: Homebrew ffmpeg@6 for TorchCodec compatibility
FFMPEG6_LIB="/opt/homebrew/opt/ffmpeg@6/lib"
if [[ -d "$FFMPEG6_LIB" && -f "$FFMPEG6_LIB/libavutil.58.dylib" ]]; then
  export DYLD_LIBRARY_PATH="${FFMPEG6_LIB}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

# Linux: Ensure system FFmpeg 4.x libs load before any older copies
# (e.g. ArenaSDK ships an older libavutil.56.14 that is missing symbols
# required by the system libavfilter.so.7).
SYS_FFMPEG="/lib/x86_64-linux-gnu"
if [[ -f "$SYS_FFMPEG/libavutil.so.56" ]]; then
  export LD_LIBRARY_PATH="${SYS_FFMPEG}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec python3 inspect_attention.py "$@"
