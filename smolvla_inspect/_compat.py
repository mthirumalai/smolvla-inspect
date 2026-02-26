"""
Runtime compatibility shims — FFmpeg preload, matplotlib backend, resize_with_pad.

This module has NO internal dependencies and must be imported before any
module that touches ``matplotlib.pyplot``.
"""

import os

# Preload Homebrew FFmpeg 6 libavdevice so PyAV/av doesn't load its bundled copy;
# avoids "Class AVFFrameReceiver is implemented in both..." duplicate symbol warning.
_ffmpeg6_lib = "/opt/homebrew/opt/ffmpeg@6/lib"
if os.path.isdir(_ffmpeg6_lib):
    _libavdevice = os.path.join(_ffmpeg6_lib, "libavdevice.60.dylib")
    if os.path.isfile(_libavdevice):
        try:
            import ctypes
            ctypes.CDLL(_libavdevice)
        except OSError:
            pass

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files

import torch.nn.functional as F

try:
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
except ImportError:
    def resize_with_pad(img, width, height, pad_value=-1):
        """Aspect-ratio-preserving resize with top/left padding."""
        if img.ndim != 4:
            raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
        cur_height, cur_width = img.shape[2:]
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = F.interpolate(
            img, size=(resized_height, resized_width), mode="bilinear", align_corners=False,
        )
        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
        return padded_img
