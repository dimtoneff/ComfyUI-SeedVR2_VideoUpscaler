"""
NVFP4 checkpoint loading via ComfyUI mixed_precision_ops.

ComfyUI NVFP4 weights store packed uint8 tensors with half the in_features
dimension plus weight_scale / weight_scale_2 metadata. Plain nn.Linear cannot
load them; MixedPrecisionOps.Linear handles the packed layout via comfy_quant.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    import comfy.model_management
    import comfy.ops
    import comfy.utils
    COMFY_QUANT_AVAILABLE = True
except ImportError:
    COMFY_QUANT_AVAILABLE = False

try:
    from safetensors import safe_open
    SAFETENSORS_METADATA_AVAILABLE = True
except ImportError:
    SAFETENSORS_METADATA_AVAILABLE = False


def read_safetensors_metadata(checkpoint_path: str) -> Dict[str, str]:
    if not checkpoint_path.endswith(".safetensors") or not SAFETENSORS_METADATA_AVAILABLE:
        return {}
    try:
        with safe_open(checkpoint_path, framework="pt") as handle:
            return handle.metadata() or {}
    except Exception:
        return {}


def is_nvfp4_checkpoint(checkpoint_path: str, state: Optional[Dict[str, torch.Tensor]] = None) -> bool:
    filename = os.path.basename(checkpoint_path).lower()
    if "nvfp4" in filename or "_fp4" in filename:
        return True

    metadata = read_safetensors_metadata(checkpoint_path)
    if metadata:
        metadata_text = str(metadata).lower()
        if "nvfp4" in metadata_text or metadata.get("quantization") == "nvfp4":
            return True
        if "_quantization_metadata" in metadata:
            return True

    if state is not None and is_nvfp4_state_dict(state):
        return True

    return False


def is_nvfp4_state_dict(state: Dict[str, torch.Tensor]) -> bool:
    if any(key.endswith(".comfy_quant") for key in state):
        return True
    if any(key.endswith(".weight_scale_2") for key in state):
        return True
    return False


def _inject_comfy_quant_from_scale_keys(state: Dict[str, torch.Tensor]) -> None:
    for key in list(state.keys()):
        if not key.endswith(".weight_scale_2"):
            continue
        layer = key[: -len(".weight_scale_2")]
        weight_key = f"{layer}.weight"
        quant_key = f"{layer}.comfy_quant"
        if weight_key not in state or quant_key in state:
            continue
        conf = json.dumps({"format": "nvfp4"})
        state[quant_key] = torch.tensor(list(conf.encode("utf-8")), dtype=torch.uint8)


def prepare_nvfp4_state_dict(
    state: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    if not COMFY_QUANT_AVAILABLE:
        return state

    metadata = metadata or {}
    state, _metadata = comfy.utils.convert_old_quants(state, model_prefix="", metadata=metadata)
    if not is_nvfp4_state_dict(state):
        _inject_comfy_quant_from_scale_keys(state)
    return state


def model_has_nvfp4_layers(module: nn.Module) -> bool:
    for child in module.modules():
        if getattr(child, "quant_format", None) == "nvfp4":
            return True
        if getattr(child, "layout_type", None) == "TensorCoreNVFP4Layout":
            return True
    return False


def _patch_nvfp4_linear_forward(linear: nn.Module, compute_dtype: torch.dtype) -> None:
    if getattr(linear, "_nvfp4_input_dtype_patched", False):
        return

    original_forward = linear.forward

    def forward(input, *args, **kwargs):
        if torch.is_tensor(input) and input.dtype == torch.float32:
            input = input.to(compute_dtype)
        return original_forward(input, *args, **kwargs)

    linear.forward = forward
    linear._nvfp4_input_dtype_patched = True


def replace_linear_with_mixed_precision(
    module: nn.Module,
    compute_dtype: torch.dtype,
    layer_device: torch.device,
) -> int:
    if not COMFY_QUANT_AVAILABLE:
        raise RuntimeError(
            "NVFP4 checkpoint requires ComfyUI quantization support (comfy.ops / comfy.utils)."
        )

    disabled = []
    if layer_device.type == "cuda" and not comfy.model_management.supports_nvfp4_compute(layer_device):
        disabled.append("nvfp4")

    ops = comfy.ops.mixed_precision_ops(compute_dtype=compute_dtype, disabled=disabled)
    replacements = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new_linear = ops.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=layer_device,
                dtype=compute_dtype,
            )
            _patch_nvfp4_linear_forward(new_linear, compute_dtype)
            setattr(module, name, new_linear)
            replacements += 1
        else:
            replacements += replace_linear_with_mixed_precision(child, compute_dtype, layer_device)

    return replacements


def _resolve_layer_device(model: nn.Module, load_device: torch.device) -> torch.device:
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        return load_device
    if model_device.type == "meta":
        return load_device
    return model_device


def load_nvfp4_model_weights(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
    compute_dtype: torch.dtype,
    load_device: torch.device,
    debug: Optional[Any] = None,
) -> Tuple[nn.Module, int]:
    layer_device = _resolve_layer_device(model, load_device)
    replacements = replace_linear_with_mixed_precision(model, compute_dtype, layer_device)
    if debug is not None:
        debug.log(
            f"Replaced {replacements} Linear layers for NVFP4 loading on {layer_device}",
            category="dit",
            force=True,
        )
    model.load_state_dict(state, strict=False, assign=True)
    model._seedvr2_nvfp4 = True
    return model, replacements
