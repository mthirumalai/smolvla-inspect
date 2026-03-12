"""Occlusion sensitivity analysis.

Slide a gray patch across the image and measure how much the predicted
actions change, producing a spatial sensitivity map.
"""

from __future__ import annotations

import numpy as np
import torch

from .counterfactual import _clone_sample, _get_actions
from .models import OcclusionMap
from .registry import register_primitive


@register_primitive(
    "occlusion.compute_sensitivity",
    category="model",
    cost="expensive",
    requires_model=True,
    requires_gpu=True,
    description="Slide an occluding patch across the image and measure action sensitivity.",
)
def compute_occlusion_sensitivity(
    policy,
    sample: dict,
    dataset,
    image_key: str,
    device: str | torch.device,
    patch_size: int = 64,
    stride: int = 32,
    image_map=None,
) -> OcclusionMap:
    """Compute a spatial sensitivity map via systematic occlusion.

    For each grid position, a gray (0.5) patch is placed over the image and
    the model is re-run.  The L2 distance between the perturbed and baseline
    action predictions is recorded, producing a ``(grid_h, grid_w)`` map.

    Parameters
    ----------
    patch_size : int
        Side length (pixels) of the occluding square.
    stride : int
        Step size (pixels) between successive patch positions.

    Returns
    -------
    OcclusionMap
        Sensitivity map and summary statistics.
    """
    img_tensor = sample[image_key]  # (C, H, W) float [0, 1]
    _, H, W = img_tensor.shape

    grid_h = (H - patch_size) // stride + 1
    grid_w = (W - patch_size) // stride + 1
    total_patches = grid_h * grid_w

    # Baseline forward pass
    policy.reset()
    baseline_actions = _get_actions(policy, sample, dataset, image_key, device, image_map)

    sensitivity_map = np.zeros((grid_h, grid_w), dtype=np.float32)

    for idx in range(total_patches):
        gy = idx // grid_w
        gx = idx % grid_w

        y0 = gy * stride
        x0 = gx * stride

        # Clone sample and occlude the patch region with gray
        occluded = _clone_sample(sample, image_key)
        occluded[image_key][:, y0 : y0 + patch_size, x0 : x0 + patch_size] = 0.5

        policy.reset()
        perturbed_actions = _get_actions(
            policy, occluded, dataset, image_key, device, image_map,
        )

        delta_l2 = float(np.linalg.norm(perturbed_actions - baseline_actions))
        sensitivity_map[gy, gx] = delta_l2

        if (idx + 1) % 10 == 0 or (idx + 1) == total_patches:
            print(f"  Occlusion: {idx + 1}/{total_patches} patches")

    return OcclusionMap(
        sensitivity_map=sensitivity_map,
        patch_size=patch_size,
        stride=stride,
        max_delta=float(sensitivity_map.max()),
        mean_delta=float(sensitivity_map.mean()),
    )
