"""Defer finite-flag readback only inside one SGLang model forward."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch


_active = ContextVar("hif4_forward_flags", default=None)


def check_invalid(flag, label):
    pending = _active.get()
    if pending is None:
        if bool(flag):
            raise FloatingPointError(label)
    else:
        pending.append((label, flag))


@contextmanager
def check_before_return():
    if _active.get() is not None:
        yield
        return
    pending = []
    token = _active.set(pending)
    try:
        yield
        if pending:
            flags = torch.stack([flag for _, flag in pending])
            if bool(flags.any()):
                indices = flags.nonzero().flatten().tolist()
                raise FloatingPointError(f"HiF4 forward failure at check {indices[0]}: {pending[indices[0]][0]}")
    finally:
        _active.reset(token)
