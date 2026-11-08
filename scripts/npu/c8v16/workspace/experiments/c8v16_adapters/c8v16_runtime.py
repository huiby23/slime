"""Slime's JSON stdin protocol with C8V16 registration in all SGLang roles."""

import json
import os
import sys

from c8v16_profile import BACKEND_NAME


def main():
    from slime.backends.sglang_utils.runtime import prepare_server_environment
    from c8v16_bootstrap import audit_effective_args, install_current_process, run_scheduler_process

    payload = json.load(sys.stdin)
    if not isinstance(payload, dict) or set(payload) != {"server_args", "parent_pid"}:
        raise ValueError("Expected Slime server_args/parent_pid payload")
    parent = payload["parent_pid"]
    if type(parent) is not int or parent <= 1 or parent != os.getppid():
        raise RuntimeError("C8V16 server parent identity mismatch")
    requested = payload["server_args"]
    if not isinstance(requested, dict) or requested.get("attention_backend") != BACKEND_NAME:
        raise ValueError("Dedicated shim requires the C8V16 backend")
    prepare_server_environment(os.environ)
    profile = install_current_process("http_parent")
    from sglang.srt.server_args import ServerArgs
    args = ServerArgs(**requested)
    args.host = args.host.strip("[]")
    audit_effective_args(profile, args, "http_parent")
    from slime.backends.sglang_utils.sglang_engine import _prepare_server_stdin
    from sglang.srt.utils import kill_itself_when_parent_died
    kill_itself_when_parent_died()
    if os.getppid() != parent:
        raise RuntimeError("Slime parent exited during server initialization")
    _prepare_server_stdin()
    from sglang.srt.entrypoints.http_server import launch_server
    launch_server(args, run_scheduler_process_func=run_scheduler_process)


if __name__ == "__main__":
    main()
