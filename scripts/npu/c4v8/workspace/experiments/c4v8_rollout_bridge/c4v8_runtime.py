"""Slime's JSON-stdin server protocol with an explicit C4V8 scheduler target."""
from __future__ import annotations

import json
import os
import sys


def main():
    # Slime launch_server_subprocess supplies exactly this schema. Do not dump
    # payload/server_args because API credentials may be present in that dict.
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict) or set(payload) != {'server_args', 'parent_pid'}:
        raise ValueError('Expected Slime server_args/parent_pid stdin payload')
    if type(payload['parent_pid']) is not int or payload['parent_pid'] <= 1:
        raise ValueError('Invalid owning parent PID')
    if os.getppid() != payload['parent_pid']:
        raise RuntimeError('Owning Slime process exited before server startup')
    requested = payload['server_args']
    if not isinstance(requested, dict):
        raise TypeError('server_args must be an object')
    # Preserve existing allocator sanitization and Mooncake defaults exactly.
    from slime.backends.sglang_utils.runtime import prepare_server_environment
    prepare_server_environment(os.environ)
    from c4v8_bootstrap import (install_current_process, audit_effective_server_args,
                               run_scheduler_process)
    from sglang_bridge import BACKEND_NAME
    if requested.get('attention_backend') != BACKEND_NAME:
        raise ValueError('Dedicated C4V8 executable requires explicit C4V8 backend selection')
    install_current_process('http_parent')
    from sglang.srt.server_args import ServerArgs
    server_args = ServerArgs(**requested)
    server_args.host = server_args.host.strip('[]')
    audit_effective_server_args(server_args, 'http_parent')
    # Preserve Slime's stdin workaround and parent-death ownership lifecycle.
    from slime.backends.sglang_utils.sglang_engine import _prepare_server_stdin
    from sglang.srt.utils import kill_itself_when_parent_died
    kill_itself_when_parent_died()
    if os.getppid() != payload['parent_pid']:
        raise RuntimeError('Owning Slime process exited during server preparation')
    _prepare_server_stdin()
    from sglang.srt.entrypoints.http_server import launch_server
    launch_server(server_args, run_scheduler_process_func=run_scheduler_process)


if __name__ == '__main__':
    main()
