"""Synchronous MCore writer with bounded CPU staging.

The planner/metadata format stays unchanged. Only one tensor per rank is
copied to CPU at a time, in the training process, without forking NPU tensors.
"""

import inspect
import os
import queue

from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
from torch.distributed.checkpoint.filesystem import DEFAULT_SUFFIX, _write_item
from torch.distributed.checkpoint.planner import WriteItemType


class StreamingFileSystemWriter(FileSystemWriterAsync):
    instances_created = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).instances_created += 1

    def prepare_write_data(self, plan, planner):
        # The synchronous path holds live tensor references until writing.
        # Do not clone CPU views or create a multiprocessing Manager here.
        file_name = f"{plan.storage_data.prefix}0{DEFAULT_SUFFIX}"
        bytes_data, tensor_data = [], []
        for item in plan.items:
            data = planner.resolve_data(item)
            if item.type == WriteItemType.BYTE_IO:
                bytes_data.append((item, data))
            else:
                tensor_data.append((item, data.detach()))
        self.write_buckets = []
        if bytes_data or tensor_data:
            self.write_buckets.append((
                os.path.join(self.checkpoint_dir, file_name), file_name,
                (bytes_data, tensor_data),
            ))
        self.results_queue = queue.Queue() if self.write_buckets else None

    def get_save_function_and_args(self):
        if not self.write_buckets:
            return None, None, []
        return self._write_streaming, None, []

    def _write_streaming(self):
        results = {}
        try:
            if self.use_msc:
                raise NotImplementedError("Streaming checkpoint writer requires a local filesystem")
            transforms = [self.transforms] if hasattr(self, "transforms") else []
            options = {}
            if "serialization_format" in inspect.signature(_write_item).parameters:
                from torch.distributed.checkpoint.filesystem import SerializationFormat

                options["serialization_format"] = SerializationFormat.TORCH_SAVE
            for index, bucket in enumerate(self.write_buckets):
                file_name, storage_key, (bytes_data, tensor_data) = bucket
                written = []
                with open(file_name, "wb") as stream:
                    for item, data in bytes_data:
                        written.append(_write_item(*transforms, stream, data, item, storage_key, **options))
                    for item, tensor in tensor_data:
                        cpu_tensor = tensor.detach().to("cpu", non_blocking=False)
                        if cpu_tensor.untyped_storage().nbytes() != cpu_tensor.numel() * cpu_tensor.element_size():
                            cpu_tensor = cpu_tensor.clone()
                        written.append(_write_item(*transforms, stream, cpu_tensor, item, storage_key, **options))
                        del cpu_tensor
                    stream.flush()
                    os.fsync(stream.fileno())
                results[index] = written
        except Exception as error:
            results = error
        self.results_queue.put(results)


def install_streaming_checkpoint_writer():
    import megatron.core.dist_checkpointing.strategies.torch as strategy

    strategy.FileSystemWriterAsync = StreamingFileSystemWriter
