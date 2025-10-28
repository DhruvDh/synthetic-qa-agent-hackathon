#!/usr/bin/python3
import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import signal
import shutil


@dataclass
class VLLMConfig:
    model: str = "openai/gpt-oss-120b"
    host: str = "127.0.0.1"
    port: int = 4200
    scheme: str = "http"
    launch_command: Optional[str] = None
    sentinel_file: Path = field(default_factory=lambda: Path("runtime/vllm_server.json"))
    startup_timeout: Optional[float] = None
    poll_interval: float = 1.0

    def __post_init__(self) -> None:
        env_cmd = os.getenv("VLLM_LAUNCH_COMMAND")
        if env_cmd and not self.launch_command:
            self.launch_command = env_cmd
        env_model = os.getenv("VLLM_MODEL")
        if env_model:
            self.model = env_model
        env_host = os.getenv("VLLM_HOST")
        if env_host:
            self.host = env_host
        env_port = os.getenv("VLLM_PORT")
        if env_port:
            try:
                self.port = int(env_port)
            except ValueError:
                pass

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    def command(self) -> List[str]:
        if self.launch_command:
            return shlex.split(self.launch_command)
        return [
            "vllm",
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]


def _ensure_package(module_name: str, package_name: Optional[str] = None):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pkg = package_name or module_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(module_name)


requests = _ensure_package("requests")
_ensure_package("openai", "openai>=1.0.0")
from openai import OpenAI


def _pid_active(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_sentinel(config: VLLMConfig) -> Optional[Dict[str, Any]]:
    path = config.sentinel_file
    if not path.exists():
        return None
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _write_sentinel(config: VLLMConfig, data: Dict[str, Any]) -> None:
    path = config.sentinel_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def _remove_sentinel(config: VLLMConfig) -> None:
    try:
        config.sentinel_file.unlink()
    except FileNotFoundError:
        pass


def server_healthy(config: VLLMConfig) -> bool:
    url = f"{config.base_url}/v1/models"
    try:
        response = requests.get(url, timeout=1.5)
        return response.ok
    except requests.RequestException:
        return False


def launch_server(config: VLLMConfig) -> subprocess.Popen:
    cmd = config.command()
    env = os.environ.copy()
    executable = cmd[0]
    if "/" not in executable and not shutil.which(executable):
        raise FileNotFoundError(
            f"Unable to locate executable '{executable}'. "
            "Install vLLM or set VLLM_LAUNCH_COMMAND to a valid startup command."
        )
    process = subprocess.Popen(
        cmd,
        text=True,
        env=env,
    )
    return process


def ensure_vllm_server_running(config: Optional[VLLMConfig] = None) -> VLLMConfig:
    cfg = config or VLLMConfig()

    sentinel = _read_sentinel(cfg)
    if server_healthy(cfg):
        if sentinel and not _pid_active(sentinel.get("pid", -1)):
            _write_sentinel(
                cfg,
                {
                    "pid": sentinel.get("pid", -1),
                    "checked_at": time.time(),
                    "status": "external",
                },
            )
        return cfg

    if sentinel:
        pid = sentinel.get("pid")
        if pid and _pid_active(pid):
            deadline = (
                time.time() + cfg.startup_timeout
                if cfg.startup_timeout is not None
                else None
            )
            while True:
                if server_healthy(cfg):
                    return cfg
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for existing vLLM server to respond."
                    )
                time.sleep(cfg.poll_interval)
        _remove_sentinel(cfg)

    process = launch_server(cfg)
    start = time.time()
    deadline = start + cfg.startup_timeout if cfg.startup_timeout is not None else None
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited early with code {process.returncode}."
            )
        if server_healthy(cfg):
            _write_sentinel(
                cfg,
                {
                    "pid": process.pid,
                    "launched_at": start,
                    "command": process.args,
                },
            )
            return cfg
        if deadline is not None and time.time() >= deadline:
            process.terminate()
            raise TimeoutError(
                f"vLLM server did not become ready within {cfg.startup_timeout} seconds."
            )
        time.sleep(cfg.poll_interval)


def stop_vllm_server(config: Optional[VLLMConfig] = None, force: bool = False) -> bool:
    cfg = config or VLLMConfig()
    sentinel = _read_sentinel(cfg)
    if not sentinel:
        return False
    pid = sentinel.get("pid", -1)
    if pid <= 0 or not _pid_active(pid):
        _remove_sentinel(cfg)
        return False

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except OSError:
        pass
    _remove_sentinel(cfg)
    return True


def chat_completion(
    messages: List[Dict[str, str]],
    config: Optional[VLLMConfig] = None,
    timeout: float = 60.0,
    **params: Any,
) -> Dict[str, Any]:
    cfg = config or VLLMConfig()
    request_kwargs: Dict[str, Any] = {"model": cfg.model, "messages": messages}
    request_kwargs.update(params)

    try:
        request_payload = dict(request_kwargs)
        unsupported_keys = {"do_sample"}
        for key in unsupported_keys:
            request_payload.pop(key, None)
        if request_payload.get("repetition_penalty") is not None:
            if request_payload["repetition_penalty"] <= 0:
                request_payload.pop("repetition_penalty", None)
        base_url = cfg.base_url.rstrip("/")
        client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        )

        extra_body: Dict[str, Any] = dict(request_payload.pop("extra_body", {}))
        for key in unsupported_keys:
            extra_body.pop(key, None)
        if extra_body.get("repetition_penalty") is not None:
            if extra_body["repetition_penalty"] <= 0:
                extra_body.pop("repetition_penalty", None)
        transport_keys = {"extra_headers", "extra_query", "timeout"}
        allowed_keys = {
            "model",
            "messages",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
            "n",
            "stream",
            "stream_options",
            "user",
            "seed",
            "logit_bias",
            "response_format",
            "tools",
            "tool_choice",
            "stop",
            "logprobs",
            "top_logprobs",
            "parallel_tool_calls",
            "functions",
            "function_call",
            "extra_headers",
            "extra_query",
        }

        migrated: Dict[str, Any] = {}
        for key in list(request_payload.keys()):
            if key in allowed_keys or key in transport_keys:
                continue
            migrated[key] = request_payload.pop(key)

        if migrated or extra_body:
            merged = {**migrated, **extra_body}
            request_payload["extra_body"] = merged

        if timeout is not None:
            request_payload["timeout"] = timeout

        response = client.chat.completions.create(**request_payload)
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "to_dict"):
            return response.to_dict()  # type: ignore[return-value]
        return json.loads(response.model_dump_json())  # type: ignore[attr-defined]
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to reach vLLM server: {exc}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM server utilities")
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Terminate the managed vLLM server if it is running.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Use SIGKILL instead of SIGTERM when stopping the server.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report whether the managed vLLM server appears healthy.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Ensure the vLLM server is running (launch if needed).",
    )
    return parser.parse_args()


def _status_message(cfg: VLLMConfig) -> str:
    sentinel = _read_sentinel(cfg)
    pid = sentinel.get("pid") if sentinel else None
    healthy = server_healthy(cfg)
    if healthy:
        return f"vLLM server running on {cfg.base_url} (pid={pid})"
    if pid:
        return f"vLLM server not responding, sentinel pid={pid}"
    return "vLLM server not running."


if __name__ == "__main__":
    args = _parse_args()
    config = VLLMConfig()

    if args.stop:
        stopped = stop_vllm_server(config, force=args.force)
        if stopped:
            print("Stopped managed vLLM server.")
        else:
            print("No managed vLLM server found to stop.")

    if args.launch:
        ensure_vllm_server_running(config)
        print("vLLM server is running.")

    if args.status or (not args.stop and not args.launch):
        print(_status_message(config))
