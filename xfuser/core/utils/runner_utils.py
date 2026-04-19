import os
import csv
import logging
import torch
import functools
import numpy as np
from PIL.Image import Image
from typing import Callable, Optional

logger = logging.getLogger(__name__)

def log(message: str, debug=False, log_from_all_processes: bool = False) -> None:
    """Log message. By default, only from the last process to avoid duplicates."""
    if log_from_all_processes or is_last_process():
        if debug:
            logger.debug(message)
        else:
            logger.info(message)

def is_last_process() -> bool:
    """
    Checks based on env rank and world size if this is last process in
    Has to be the last process, as legacy xDiT models only produce the
    output on the last GPU.
    """
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    return rank == world_size - 1

def resize_image_to_max_area(image: Image, input_height: int, input_width: int, mod_value: int) -> Image:
    """ Resize image to fit within max area while retaining aspect ratio """

    max_area = input_height * input_width
    width, height = image.size
    aspect_ratio = image.height / image.width
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area /aspect_ratio)) // mod_value * mod_value

    image = image.resize((width, height))
    log(f"Resized image to {image.width}x{image.height} to fit within max area {width}x{height}")
    return image

def resize_and_crop_image(image: Image, target_height: int, target_width: int, mod_value: int) -> Image:
        """ Resize and center-crop image to target dimensions """

        target_height_aligned = target_height // mod_value * mod_value
        target_width_aligned = target_width // mod_value * mod_value

        log("Force output size mode enabled.")
        log(f"Input image resolution: {image.height}x{image.width}")
        log(f"Requested output resolution: {target_height}x{target_width}")
        log(f"Aligned output resolution (multiple of {mod_value}): {target_height_aligned}x{target_width_aligned}")

        # Step 1: Resize image maintaining aspect ratio so both dimensions >= target
        img_width, img_height = image.size

        # Calculate scale factor to ensure both dimensions are at least target size
        scale_width = target_width_aligned / img_width
        scale_height = target_height_aligned / img_height
        scale = max(scale_width, scale_height)  # Use max to ensure both dims are >= target

        # Resize with aspect ratio preserved
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        image = image.resize((new_width, new_height))

        log(f"Resized image to: {new_height}x{new_width} (maintaining aspect ratio)")

        # Step 2: Crop from center to get exact target dimensions
        left = (new_width - target_width_aligned) // 2
        top = (new_height - target_height_aligned) // 2
        image = image.crop((left, top, left + target_width_aligned, top + target_height_aligned))

        log(f"Cropped from center to: {target_height_aligned}x{target_width_aligned}")
        return image

def _get_fp8_kernel_preference():
    """Select FP8 kernel preference based on GPU architecture.

    AUTO selects CUTLASS sm90a kernels which crash on Blackwell (sm100a)
    under torch.compile, so we force TORCH (_scaled_mm) on Blackwell+.
    """
    from torchao.quantization.quantize_.common import KernelPreference
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        return KernelPreference.TORCH
    return KernelPreference.AUTO


def quantize_linear_layers_to_fp8(module_or_module_list_to_quantize: torch.nn.Module | torch.nn.ModuleList,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
    device: Optional[torch.device] = None) -> None:
    """Quantize all linear layers in the given module or module list to FP8."""
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig, quantize_, _is_linear

    if filter_fn is None:
        filter_fn = _is_linear
    config = Float8DynamicActivationFloat8WeightConfig(
                granularity=PerTensor(),
                set_inductor_config=False,
                kernel_preference=_get_fp8_kernel_preference(),
        )
    if isinstance(module_or_module_list_to_quantize, torch.nn.Module):
        module_or_module_list_to_quantize = [module_or_module_list_to_quantize]
    for module in module_or_module_list_to_quantize:
        quantize_(
            module,
            config=config,
            filter_fn=filter_fn,
            device=device
        )


def load_dataset_prompts(dataset_path: str) -> list[str]:
    """ load prompts from a csv dataset file """
    prompts = []
    with open(dataset_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts.append(row['prompt'])
    log(f"Loaded {len(prompts)} prompts from dataset at {dataset_path}")
    return prompts

def rsetattr(obj: object, attr: str, value: object) -> None:
    """ Recursive setattr to set nested attributes """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, value)

def rgetattr(obj: object, attr: str) -> object:
    """ Recursive getattr to get nested attributes """
    return functools.reduce(getattr, [obj] + attr.split("."))

def quantize_linear_layers_to_fp4(model, parent_name='', fp8_layers=None, use_hybrid_schedule: bool = False, device: Optional[torch.device] = None):
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig, quantize_
    from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear, xFuserHybridMXFP4Linear

    for name, module in list(model.named_children()):
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(module, torch.nn.Linear):
            if fp8_layers and full_name.startswith(fp8_layers):
                quantize_(
                      module,
                      config=Float8DynamicActivationFloat8WeightConfig(
                          granularity=PerTensor(),
                          set_inductor_config=False,
                          kernel_preference=_get_fp8_kernel_preference(),
                    ),
                    device=device,
                )
            else:
                low_precision_layer = xFuserMXFP4Linear(
                    module.in_features,
                    module.out_features,
                    bias=(module.bias is not None),
                    device=module.weight.device,
                    dtype=module.weight.dtype
                )

                with torch.no_grad():
                    low_precision_layer.load_and_quantize_weights(module.weight, module.bias)

                if use_hybrid_schedule:
                    high_precision_layer = torch.nn.Linear(
                        module.in_features,
                        module.out_features,
                        bias=(module.bias is not None),
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                    with torch.no_grad():
                        high_precision_layer.weight.copy_(module.weight)
                        if module.bias is not None:
                            high_precision_layer.bias.copy_(module.bias)
                    quantize_(
                        high_precision_layer,
                        config=Float8DynamicActivationFloat8WeightConfig(
                            granularity=PerTensor(),
                            set_inductor_config=False,
                            kernel_preference=_get_fp8_kernel_preference(),
                        ),
                        device=device,
                    )
                    new_layer = xFuserHybridMXFP4Linear(
                        high_precision_linear=high_precision_layer,
                        low_precision_linear=low_precision_layer,
                    )
                else:
                    new_layer = low_precision_layer

                setattr(model, name, new_layer)

        elif len(list(module.children())) > 0:
            quantize_linear_layers_to_fp4(module, full_name, fp8_layers=fp8_layers, use_hybrid_schedule=use_hybrid_schedule, device=device)


def quantize_linear_layers_to_nvfp4(
    module_or_module_list_to_quantize: torch.nn.Module | torch.nn.ModuleList,
    fp8_layers: tuple[str] = None,
    device: Optional[torch.device] = None,
    min_layer_size: int = 0,
    use_triton_kernel: bool = True,
) -> None:
    """Quantize linear layers to NVFP4 using torchao on NVIDIA Blackwell GPUs.

    Args:
        module_or_module_list_to_quantize: Module(s) whose linear layers will be quantized.
        fp8_layers: FQN prefixes of layers that should use FP8 instead of NVFP4
            for quality-sensitive blocks.
        device: Target device.
        min_layer_size: Skip NVFP4 for layers where min(out_features, in_features)
            is below this threshold (quantization overhead may exceed the speedup).
        use_triton_kernel: Whether to use the Triton-based NVFP4 kernel.
    """
    from torchao.prototype.mx_formats.inference_workflow import (
        NVFP4DynamicActivationNVFP4WeightConfig,
    )
    from torchao.quantization.quant_api import quantize_, _is_linear

    nvfp4_config = NVFP4DynamicActivationNVFP4WeightConfig(
        use_dynamic_per_tensor_scale=True,
        use_triton_kernel=use_triton_kernel,
    )

    if isinstance(module_or_module_list_to_quantize, torch.nn.Module):
        module_or_module_list_to_quantize = [module_or_module_list_to_quantize]

    quantized_count = 0
    skipped_fp8_count = 0
    skipped_small_count = 0

    for module in module_or_module_list_to_quantize:
        for fqn, submodule in module.named_modules():
            if not isinstance(submodule, torch.nn.Linear):
                continue

            if fp8_layers and fqn.startswith(fp8_layers):
                skipped_fp8_count += 1
                continue

            layer_min = min(submodule.out_features, submodule.in_features)
            if min_layer_size > 0 and layer_min < min_layer_size:
                skipped_small_count += 1
                continue

            quantized_count += 1

        def nvfp4_filter_fn(mod, fqn):
            if not _is_linear(mod, fqn):
                return False
            if fp8_layers and fqn.startswith(fp8_layers):
                return False
            if min_layer_size > 0:
                layer_min = min(mod.out_features, mod.in_features)
                if layer_min < min_layer_size:
                    return False
            return True

        quantize_(module, config=nvfp4_config, filter_fn=nvfp4_filter_fn, device=device)

        if fp8_layers:
            from torchao.quantization.granularity import PerTensor
            from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig

            fp8_config = Float8DynamicActivationFloat8WeightConfig(
                granularity=PerTensor(),
                set_inductor_config=False,
                kernel_preference=_get_fp8_kernel_preference(),
            )

            def fp8_filter_fn(mod, fqn):
                if not _is_linear(mod, fqn):
                    return False
                return fqn.startswith(fp8_layers)

            quantize_(module, config=fp8_config, filter_fn=fp8_filter_fn, device=device)

    log(f"  [NVFP4] Summary: {quantized_count} layers quantized to NVFP4, "
        f"{skipped_fp8_count} overridden to FP8, {skipped_small_count} skipped (too small)")


class FlashinferFP4Linear(torch.nn.Module):
    """Drop-in nn.Linear replacement using flashinfer mm_fp4.

    Weight is quantized to NVFP4 at construction time.  Activation is quantized
    dynamically on each forward call.  The global scale-factor computation is
    compiled so it fuses with surrounding ops under torch.compile.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

    @staticmethod
    def from_linear(linear: torch.nn.Linear) -> "FlashinferFP4Linear":
        from flashinfer import nvfp4_quantize, SfLayout

        mod = FlashinferFP4Linear(linear.in_features, linear.out_features,
                                  bias=linear.bias is not None)
        w = linear.weight.data.to(dtype=torch.bfloat16)
        maxabs = w.float().abs().nan_to_num().max()
        maxabs = torch.maximum(maxabs, torch.tensor(1e-12, device=w.device))
        global_sf_w = (448.0 * 6.0) / maxabs
        w_fp4, w_sf = nvfp4_quantize(w, global_sf_w,
                                      sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        mod.register_buffer("global_sf_w", global_sf_w)
        mod.register_buffer("w_fp4", w_fp4)
        mod.register_buffer("w_sf", w_sf)
        if linear.bias is not None:
            mod.register_buffer("bias", linear.bias.data.clone())
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from flashinfer import mm_fp4, nvfp4_quantize, SfLayout

        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features).contiguous()
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)

        maxabs = x2d.float().abs().nan_to_num().max()
        maxabs = torch.maximum(maxabs, torch.tensor(1e-12, device=x2d.device))
        sf_a = (448.0 * 6.0) / maxabs
        a_fp4, a_sf = nvfp4_quantize(x2d, sf_a,
                                      sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        alpha = 1.0 / (sf_a * self.global_sf_w)
        out = mm_fp4(a_fp4, self.w_fp4.T, a_sf, self.w_sf.T, alpha,
                     torch.bfloat16, block_size=16, use_8x4_sf_layout=False,
                     backend="cutlass")

        if self.has_bias:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias}"


def quantize_linear_layers_to_flashinfer_fp4(
    module_or_module_list_to_quantize: torch.nn.Module | torch.nn.ModuleList,
    fp8_layers: tuple[str] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Replace nn.Linear layers with FlashinferFP4Linear using flashinfer mm_fp4."""

    if isinstance(module_or_module_list_to_quantize, torch.nn.Module):
        module_or_module_list_to_quantize = [module_or_module_list_to_quantize]

    quantized_count = 0
    skipped_fp8_count = 0

    for module in module_or_module_list_to_quantize:
        for fqn, submodule in list(module.named_modules()):
            if not isinstance(submodule, torch.nn.Linear):
                continue
            if fp8_layers and fqn.startswith(fp8_layers):
                skipped_fp8_count += 1
                continue
            quantized_count += 1

        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear):
                if fp8_layers and name.startswith(fp8_layers):
                    continue
                replacement = FlashinferFP4Linear.from_linear(child)
                setattr(module, name, replacement)
            else:
                _replace_linears_recursive(child, fp8_layers, prefix=name)

    if fp8_layers:
        from torchao.quantization.granularity import PerTensor
        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig, quantize_, _is_linear,
        )
        fp8_config = Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            set_inductor_config=False,
            kernel_preference=_get_fp8_kernel_preference(),
        )
        for module in module_or_module_list_to_quantize:
            def fp8_filter_fn(mod, fqn):
                if not _is_linear(mod, fqn):
                    return False
                return fqn.startswith(fp8_layers)
            quantize_(module, config=fp8_config, filter_fn=fp8_filter_fn, device=device)

    log(f"  [FlashinferFP4] Summary: {quantized_count} layers quantized to FP4, "
        f"{skipped_fp8_count} overridden to FP8")


def _replace_linears_recursive(module: torch.nn.Module, fp8_layers: tuple = None, prefix: str = ""):
    """Recursively replace nn.Linear with FlashinferFP4Linear."""
    for name, child in list(module.named_children()):
        fqn = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            if fp8_layers and fqn.startswith(fp8_layers):
                continue
            replacement = FlashinferFP4Linear.from_linear(child)
            setattr(module, name, replacement)
        else:
            _replace_linears_recursive(child, fp8_layers, prefix=fqn)


def convert_model_convs_to_channels_last(model: torch.nn.Module) -> None:
    """
    Manually convert 2D and 3D convolutional layer weights to channels_last format.
     - Conv3d weights: (out_channels, in_channels, D, H, W) -> channels_last_3d
     - Conv2d weights: (out_channels, in_channels, H, W) -> channels_last
     - Biases and non-conv parameters are left unchanged (they are 1D and not affected by memory format)
     - This is done in-place to avoid unnecessary copying of the entire model and to ensure we only change what is needed.
    """
    for param in model.parameters():
        if param.dim() == 5:
            param.data = param.data.to(memory_format=torch.channels_last_3d)
        elif param.dim() == 4:
            param.data = param.data.to(memory_format=torch.channels_last)
