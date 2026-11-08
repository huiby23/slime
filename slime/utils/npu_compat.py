"""Ascend integration for Ray scheduling and the pinned MindSpeed backend.

NPU workers use Ray's NPU resources, while existing CUDA workers keep the
reserved num_gpus option. Importing this module does not initialize MindSpeed.
"""

import ray

from slime.utils import accelerator


def is_npu_backend():
    backend = accelerator.initialize_accelerator()
    return backend is not None and backend.device_type == "npu"


def ray_resource_name():
    return "NPU" if is_npu_backend() else "GPU"


def ray_accelerator_options(amount):
    if is_npu_backend():
        return {"resources": {"NPU": amount}}
    return {"num_gpus": amount}


def ray_device_ids():
    if is_npu_backend():
        return ray.get_runtime_context().get_accelerator_ids().get("NPU", [])
    return ray.get_gpu_ids()


def bootstrap_megatron_npu():
    if is_npu_backend():
        # Must run before importing Megatron layers so MindSpeed can replace
        # their CUDA-specific implementations with Ascend implementations.
        import mindspeed.megatron_adaptor  # noqa: F401


def repatch_megatron_npu(args):
    if is_npu_backend():
        from mindspeed.megatron_adaptor import repatch

        repatch(args)
