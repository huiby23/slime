"""Keep a small CUDA allocation visible while an owned container initializes.

This is a cooperative occupancy marker, not an exclusive device lock. It never
stops another process or changes model arithmetic. The parent container owns
its lifetime. Uses the CUDA driver directly to avoid slow framework imports.
"""

import argparse
import ctypes as c
import json
import os
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    driver = c.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": ([c.c_uint], c.c_int),
        "cuDeviceGetCount": ([c.POINTER(c.c_int)], c.c_int),
        "cuDeviceGet": ([c.POINTER(c.c_int), c.c_int], c.c_int),
        "cuDeviceGetName": ([c.c_char_p, c.c_int, c.c_int], c.c_int),
        "cuDevicePrimaryCtxRetain": ([c.POINTER(c.c_void_p), c.c_int], c.c_int),
        "cuCtxSetCurrent": ([c.c_void_p], c.c_int),
        "cuMemAlloc_v2": ([c.POINTER(c.c_uint64), c.c_size_t], c.c_int),
        "cuMemFree_v2": ([c.c_uint64], c.c_int),
        "cuDevicePrimaryCtxRelease_v2": ([c.c_int], c.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = argtypes, restype

    def call(name, *values):
        result = getattr(driver, name)(*values)
        if result:
            raise RuntimeError(f"{name} returned CUDA driver error {result}")

    class CudaUuid(c.Structure):
        _fields_ = [("bytes", c.c_ubyte * 16)]

    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", driver.cuDeviceGetUuid)
    get_uuid.argtypes, get_uuid.restype = [c.POINTER(CudaUuid), c.c_int], c.c_int
    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    allocations = []
    report = dict(
        pid=os.getpid(),
        parent_pid=args.parent_pid,
        active=False,
        devices=[],
        purpose="Small occupancy marker for an already launched training container; not an exclusive lock",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def publish():
        report["at"] = datetime.now(timezone.utc).isoformat()
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    try:
        call("cuInit", 0)
        count = c.c_int()
        call("cuDeviceGetCount", c.byref(count))
        if not count.value:
            raise RuntimeError("No visible CUDA devices for the launched job")
        for ordinal in range(count.value):
            device, context, pointer = c.c_int(), c.c_void_p(), c.c_uint64()
            call("cuDeviceGet", c.byref(device), ordinal)
            call("cuDevicePrimaryCtxRetain", c.byref(context), device)
            allocations.append((device, context, pointer))
            call("cuCtxSetCurrent", context)
            call("cuMemAlloc_v2", c.byref(pointer), 64 * 1024**2)
            ident, name = CudaUuid(), c.create_string_buffer(128)
            if get_uuid(c.byref(ident), device):
                raise RuntimeError("Cannot identify claimed GPU UUID")
            call("cuDeviceGetName", name, len(name), device)
            report["devices"].append(
                dict(
                    ordinal=ordinal,
                    uuid="GPU-" + str(uuid.UUID(bytes=bytes(ident.bytes))),
                    name=name.value.decode(),
                    allocated_bytes=64 * 1024**2,
                )
            )
        report["active"] = True
        publish()
        while not stop:
            try:
                os.kill(args.parent_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(1)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for device, context, pointer in reversed(allocations):
            driver.cuCtxSetCurrent(context)
            if pointer.value:
                driver.cuMemFree_v2(pointer)
            driver.cuDevicePrimaryCtxRelease_v2(device)
        report["active"] = False
        publish()


if __name__ == "__main__":
    main()
