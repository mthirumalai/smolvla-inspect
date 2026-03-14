"""Pluggable registries for all diagnostic components.

Registries
----------
REGISTRY               — diagnostic primitives (scene / model / counterfactual / …)
SIGNAL_REGISTRY        — expensive signal runner functions
HYPOTHESIS_TEMPLATES   — rule-based anomaly → hypothesis mapping
LLM_PROVIDERS          — LLM provider callables
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# 1. Primitive registry
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveSpec:
    """Describes a registered diagnostic primitive."""
    name: str
    fn: Callable
    category: str          # "scene" | "model" | "composite" | "counterfactual" | "dataset"
    requires_gpu: bool
    requires_model: bool
    requires_scene_models: bool
    cost: str              # "cheap" | "moderate" | "expensive"
    description: str
    prompt_description: str = ""
    param_schema: str = ""


REGISTRY: dict[str, PrimitiveSpec] = {}


def register_primitive(name: str, *, category: str, cost: str = "cheap",
                       requires_gpu: bool = False, requires_model: bool = False,
                       requires_scene_models: bool = False, description: str = "",
                       prompt_description: str = "", param_schema: str = ""):
    """Decorator to register a function as a diagnostic primitive."""
    def decorator(fn: Callable) -> Callable:
        REGISTRY[name] = PrimitiveSpec(
            name=name, fn=fn, category=category,
            requires_gpu=requires_gpu, requires_model=requires_model,
            requires_scene_models=requires_scene_models,
            cost=cost, description=description,
            prompt_description=prompt_description,
            param_schema=param_schema,
        )
        return fn
    return decorator


def get_primitive(name: str) -> PrimitiveSpec:
    """Look up a primitive by name. Raises KeyError if not found."""
    return REGISTRY[name]


def list_primitives(category: str | None = None) -> list[PrimitiveSpec]:
    """List all registered primitives, optionally filtered by category."""
    specs = list(REGISTRY.values())
    if category is not None:
        specs = [s for s in specs if s.category == category]
    return sorted(specs, key=lambda s: s.name)


# ---------------------------------------------------------------------------
# 2. Signal registry — expensive signal runner functions
# ---------------------------------------------------------------------------

@dataclass
class SignalSpec:
    """Describes a registered expensive signal runner."""
    name: str
    cost_description: str       # e.g. "~30s/frame"
    prompt_description: str     # description for LLM triage prompt
    requires_model: bool = True
    fn: Callable | None = None


SIGNAL_REGISTRY: dict[str, SignalSpec] = {}


def register_signal(name: str, *, cost_description: str = "",
                    prompt_description: str = "", requires_model: bool = True):
    """Decorator to register an expensive signal runner function."""
    def decorator(fn: Callable) -> Callable:
        spec = SignalSpec(
            name=name,
            cost_description=cost_description,
            prompt_description=prompt_description,
            requires_model=requires_model,
            fn=fn,
        )
        SIGNAL_REGISTRY[name] = spec
        return fn
    return decorator


# ---------------------------------------------------------------------------
# 3. Hypothesis template registry — rule-based anomaly → hypothesis mapping
# ---------------------------------------------------------------------------

@dataclass
class HypothesisTemplate:
    """Maps an anomaly type to a canned hypothesis and counterfactual test."""
    anomaly_type: str
    description_template: str      # May contain {target}
    confidence: float
    test_type: str
    test_params_template: dict     # May contain "{target}" string values
    expected_if_true: str
    expected_if_false: str
    confirms_on_change: bool = True


HYPOTHESIS_TEMPLATES: dict[str, HypothesisTemplate] = {}


def register_hypothesis_template(anomaly_type: str, **kwargs) -> None:
    """Register a rule-based hypothesis template for a given anomaly type."""
    HYPOTHESIS_TEMPLATES[anomaly_type] = HypothesisTemplate(
        anomaly_type=anomaly_type, **kwargs,
    )


# ---------------------------------------------------------------------------
# 4. LLM provider registry
# ---------------------------------------------------------------------------

LLM_PROVIDERS: dict[str, Callable] = {}


def register_llm_provider(name: str):
    """Decorator to register an LLM provider callable."""
    def decorator(fn: Callable) -> Callable:
        LLM_PROVIDERS[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# 5. Prompt auto-generation helpers
# ---------------------------------------------------------------------------

def format_counterfactual_prompt_section() -> str:
    """Auto-generate the available counterfactual tests section from the registry."""
    lines = []
    for spec in list_primitives(category="counterfactual"):
        short_name = spec.name.removeprefix("counterfactual.")
        desc = spec.prompt_description or spec.description
        line = f"- {short_name}: {desc}"
        if spec.param_schema:
            line += f"\n  Params: {{{spec.param_schema}}}"
        lines.append(line)
    lines.append(
        "- none: No test needed — hypothesis is already well-supported "
        "by the matrix data alone."
    )
    return "\n".join(lines)


def format_signals_prompt_section() -> str:
    """Auto-generate the expensive signals section from the registry."""
    lines = []
    for name, spec in sorted(SIGNAL_REGISTRY.items()):
        lines.append(f"- {name}: {spec.prompt_description} Cost: {spec.cost_description}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Built-in hypothesis templates
#    (translated from the if/elif chain in agent.py _rule_based_hypotheses)
# ---------------------------------------------------------------------------

register_hypothesis_template(
    "high_background_attribution",
    description_template="Model relies on background features rather than task-relevant objects",
    confidence=0.7,
    test_type="background_substitution",
    test_params_template={"replacement": "gray"},
    expected_if_true="Action prediction changes significantly when background is replaced",
    expected_if_false="Action prediction remains stable",
    confirms_on_change=True,
)

register_hypothesis_template(
    "spatial_shortcut",
    description_template="Model uses spatial shortcuts — memorized {target} position rather than recognizing it",
    confidence=0.8,
    test_type="object_relocation",
    test_params_template={"target_object": "{target}", "shift_pixels": [100, -80]},
    expected_if_true="Action prediction MSE increases >2x when object is relocated",
    expected_if_false="Model correctly adjusts actions to new object position",
    confirms_on_change=True,
)

register_hypothesis_template(
    "low_object_attribution",
    description_template="Model has weak visual grounding for {target}",
    confidence=0.6,
    test_type="object_recolor",
    test_params_template={"target_object": "{target}", "hue_shift": 0.5},
    expected_if_true="Model is insensitive to object appearance changes",
    expected_if_false="Action changes, showing some object-appearance sensitivity",
    confirms_on_change=False,
)

register_hypothesis_template(
    "dead_state_pathway",
    description_template="Proprioceptive state input is not contributing to action decisions",
    confidence=0.6,
    test_type="none",
    test_params_template={},
    expected_if_true="N/A — confirmed by vision vs state analysis",
    expected_if_false="N/A",
    confirms_on_change=True,
)

register_hypothesis_template(
    "low_dataset_diversity",
    description_template="Model memorized fixed {target} positions due to low dataset diversity",
    confidence=0.7,
    test_type="object_relocation",
    test_params_template={"target_object": "{target}", "shift_pixels": [80, -60]},
    expected_if_true="Action breaks when object is moved from memorized position",
    expected_if_false="Model generalizes to new positions despite low training diversity",
    confirms_on_change=True,
)

register_hypothesis_template(
    "gripper_fixation",
    description_template="Model fixates on robot gripper instead of manipulation target",
    confidence=0.7,
    test_type="occlusion_targeted",
    test_params_template={"target_object": "{target}", "fill": "gray"},
    expected_if_true="Occluding target object has minimal effect (model relies on gripper)",
    expected_if_false="Occluding target changes actions significantly",
    confirms_on_change=False,
)

register_hypothesis_template(
    "cross_attention_diffuse",
    description_template="Action expert cross-attention is near-uniform — not selectively querying",
    confidence=0.5,
    test_type="distractor_insertion",
    test_params_template={"position": [100, 100], "distractor_size": 80},
    expected_if_true="Model is equally distracted by novel objects",
    expected_if_false="Model ignores distractors despite diffuse attention",
    confirms_on_change=True,
)

register_hypothesis_template(
    "language_insensitivity",
    description_template="Model ignores language conditioning — acts the same regardless of instruction",
    confidence=0.7,
    test_type="task_string_swap",
    test_params_template={"replacement_task": "do nothing"},
    expected_if_true="Changing instruction to 'do nothing' has no effect on actions",
    expected_if_false="Actions change, suggesting some language sensitivity",
    confirms_on_change=False,
)

register_hypothesis_template(
    "temporal_attention_instability",
    description_template="Attention is temporally unstable — jumps between frames",
    confidence=0.6,
    test_type="temporal_consistency",
    test_params_template={"perturbation_type": "background_substitution", "num_frames": 5},
    expected_if_true="Model responds inconsistently to same perturbation across frames",
    expected_if_false="Model responds consistently despite attention instability",
    confirms_on_change=True,
)
