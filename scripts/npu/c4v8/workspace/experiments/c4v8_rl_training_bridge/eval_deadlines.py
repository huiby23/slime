"""Process-local evaluation limits; no accelerator imports or background jobs."""
from contextlib import contextmanager
import math
import signal
import time


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Timeout must be finite and positive')
    return value


class EvaluationBudget:
    """A wall-clock budget, including I/O, with bounded owned cleanup afterwards.

    The optional timer defaults to disabled for old callers. SIGTERM still takes
    the exception/finally path. Deferral does not block inherited child signals:
    a signal during Popen is remembered until the owner has its child handle.
    """
    def __init__(self, seconds=None):
        self.seconds = None if seconds is None else positive_seconds(seconds)
        self.depth = 0
        self.pending = None

    def _interrupt(self, signum, _frame):
        error = (TimeoutError('Total 21-point evaluation time budget exceeded')
                 if signum == signal.SIGALRM else InterruptedError('Evaluation termination requested'))
        if self.depth:
            self.pending = self.pending or error
        else:
            raise error

    def __enter__(self):
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError('Refusing to replace an existing process timer')
        self.started = time.monotonic()
        self.old = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        for sig in self.old: signal.signal(sig, self._interrupt)
        if self.seconds is not None: signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in self.old.items(): signal.signal(sig, handler)

    @contextmanager
    def defer_termination(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if self.depth == 0 and self.pending is not None:
                error, self.pending = self.pending, None
                raise error

    @contextmanager
    def owned_cleanup(self):
        # Cleanup has its own bounded waits. A deadline or repeated TERM must
        # not abandon the already verified process group halfway through it.
        signal.setitimer(signal.ITIMER_REAL, 0)
        old = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGTERM, old)
