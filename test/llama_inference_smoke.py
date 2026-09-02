#!/usr/bin/env python3
"""Drive stock llama-cli and llama-server through penzai.inference.v1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import time
from typing import Any
from urllib import error, request


GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGML_TYPE_F32 = 0
ALIGNMENT = 32


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def pack_value(value_type: int, value: Any) -> bytes:
    if value_type == GGUF_TYPE_UINT32:
        return struct.pack("<I", value)
    if value_type == GGUF_TYPE_INT32:
        return struct.pack("<i", value)
    if value_type == GGUF_TYPE_FLOAT32:
        return struct.pack("<f", value)
    if value_type == GGUF_TYPE_BOOL:
        return struct.pack("<?", value)
    if value_type == GGUF_TYPE_STRING:
        return pack_string(value)
    if value_type == GGUF_TYPE_ARRAY:
        item_type, items = value
        payload = struct.pack("<IQ", item_type, len(items))
        return payload + b"".join(pack_value(item_type, item) for item in items)
    raise ValueError(f"unsupported GGUF value type: {value_type}")


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def write_tiny_qwen3(path: Path) -> None:
    """Write a valid, tiny Qwen3 GGUF using only the Python standard library."""
    tokens = ["<unk>", "<s>", "</s>", "\u2581", "\u2581a", "\u2581OK", "!"]
    token_types = [2, 3, 3, 1, 1, 1, 1]
    for byte in range(256):
        tokens.append(f"<0x{byte:02X}>")
        token_types.append(6)

    metadata = [
        ("general.architecture", GGUF_TYPE_STRING, "qwen3"),
        ("general.name", GGUF_TYPE_STRING, "penzai-inference-smoke"),
        ("general.file_type", GGUF_TYPE_UINT32, 0),
        ("general.quantization_version", GGUF_TYPE_UINT32, 2),
        ("qwen3.context_length", GGUF_TYPE_UINT32, 512),
        ("qwen3.embedding_length", GGUF_TYPE_UINT32, 8),
        ("qwen3.feed_forward_length", GGUF_TYPE_UINT32, 16),
        ("qwen3.block_count", GGUF_TYPE_UINT32, 1),
        ("qwen3.attention.head_count", GGUF_TYPE_UINT32, 1),
        ("qwen3.attention.head_count_kv", GGUF_TYPE_UINT32, 1),
        ("qwen3.attention.key_length", GGUF_TYPE_UINT32, 8),
        ("qwen3.attention.value_length", GGUF_TYPE_UINT32, 8),
        ("qwen3.rope.dimension_count", GGUF_TYPE_UINT32, 8),
        ("qwen3.rope.freq_base", GGUF_TYPE_FLOAT32, 1_000_000.0),
        ("qwen3.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, 1.0e-6),
        ("tokenizer.ggml.model", GGUF_TYPE_STRING, "llama"),
        ("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, tokens)),
        ("tokenizer.ggml.scores", GGUF_TYPE_ARRAY, (GGUF_TYPE_FLOAT32, [0.0] * len(tokens))),
        ("tokenizer.ggml.token_type", GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, token_types)),
        ("tokenizer.ggml.bos_token_id", GGUF_TYPE_UINT32, 1),
        ("tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32, 2),
        ("tokenizer.ggml.unknown_token_id", GGUF_TYPE_UINT32, 0),
        ("tokenizer.ggml.add_bos_token", GGUF_TYPE_BOOL, True),
        ("tokenizer.ggml.add_eos_token", GGUF_TYPE_BOOL, False),
        (
            "tokenizer.chat_template",
            GGUF_TYPE_STRING,
            "{% for message in messages %}{{ message['content'] }}{% endfor %}",
        ),
    ]

    vocab = len(tokens)
    tensors = [
        ("token_embd.weight", (8, vocab)),
        ("output_norm.weight", (8,)),
        ("blk.0.attn_norm.weight", (8,)),
        ("blk.0.attn_q.weight", (8, 8)),
        ("blk.0.attn_k.weight", (8, 8)),
        ("blk.0.attn_v.weight", (8, 8)),
        ("blk.0.attn_output.weight", (8, 8)),
        ("blk.0.attn_q_norm.weight", (8,)),
        ("blk.0.attn_k_norm.weight", (8,)),
        ("blk.0.ffn_norm.weight", (8,)),
        ("blk.0.ffn_gate.weight", (8, 16)),
        ("blk.0.ffn_down.weight", (16, 8)),
        ("blk.0.ffn_up.weight", (8, 16)),
    ]

    kv_data = b"".join(
        pack_string(key) + struct.pack("<I", value_type) + pack_value(value_type, value)
        for key, value_type, value in metadata
    )

    tensor_info = bytearray()
    tensor_data: list[tuple[int, int]] = []
    offset = 0
    for name, dimensions in tensors:
        element_count = 1
        for dimension in dimensions:
            element_count *= dimension
        size = element_count * 4
        tensor_info += pack_string(name)
        tensor_info += struct.pack("<I", len(dimensions))
        tensor_info += b"".join(struct.pack("<Q", dimension) for dimension in dimensions)
        tensor_info += struct.pack("<IQ", GGML_TYPE_F32, offset)
        tensor_data.append((offset, size))
        offset = align(offset + size)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    with path.open("wb") as output:
        output.write(header)
        output.write(kv_data)
        output.write(tensor_info)
        output.write(b"\0" * (align(output.tell()) - output.tell()))
        data_start = output.tell()
        for tensor_offset, size in tensor_data:
            output.write(b"\0" * (data_start + tensor_offset - output.tell()))
            output.write(b"\0" * size)


def read_audit(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assert_audit(path: Path, expected_winners: list[int], expected_capacity: int) -> None:
    events = read_audit(path)
    names = [event.get("event") for event in events]
    for required in ("model.load", "session.open", "execute", "session.close", "model.unload"):
        if required not in names:
            raise AssertionError(f"missing {required!r} in audit {path}: {events}")

    if names.index("model.load") > names.index("session.open"):
        raise AssertionError(f"session opened before model load: {events}")
    if names.index("session.close") > names.index("model.unload"):
        raise AssertionError(f"model unloaded before session close: {events}")

    session_open = next(event for event in events if event.get("event") == "session.open")
    if session_open.get("capacity_tokens") != expected_capacity:
        raise AssertionError(
            f"unexpected executor context capacity: expected={expected_capacity}, event={session_open}"
        )

    executes = [event for event in events if event.get("event") == "execute"]
    if not executes or not any(event.get("token_count") == 4 for event in executes):
        raise AssertionError(f"prompt did not traverse a four-token tile: {executes}")
    for event in executes:
        expected_commit = event.get("first_position") + event.get("token_count")
        if event.get("committed_tokens") != expected_commit:
            raise AssertionError(f"non-contiguous executor commit: {event}")

    winners = [event.get("token_id") for event in executes if event.get("emit")]
    if winners != expected_winners:
        raise AssertionError(f"unexpected scripted winners: {winners}")


def executor_env(backend: Path, audit: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GGML_BACKEND_PATH": str(backend),
            "PENZAI_EXECUTOR": "mock",
            "PENZAI_EXECUTOR_TOKENS": "5,6,2",
            "PENZAI_EXECUTOR_AUDIT": str(audit),
        }
    )
    return env


def run_logits_probe(
    binary: Path, cpu_backend: Path, backend: Path, model: Path, audit: Path
) -> None:
    completed = subprocess.run(
        [str(binary.resolve()), str(model), str(cpu_backend), str(backend), "5"],
        env=executor_env(backend, audit),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"llama logits probe failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if "llama logits probe: ok" not in completed.stdout:
        raise AssertionError(f"missing logits probe success marker:\n{completed.stdout}")


def run_cli(binary: Path, backend: Path, model: Path, audit: Path, prompt: str) -> None:
    command = [
        str(binary),
        "--device",
        "penzai",
        "--model",
        str(model),
        "--prompt",
        prompt,
        "--predict",
        "2",
        "--ctx-size",
        "256",
        "--batch-size",
        "32",
        "--ubatch-size",
        "32",
        "--temp",
        "0",
        "--no-warmup",
        "--no-conversation",
        "--single-turn",
        "--no-display-prompt",
        "--simple-io",
        "--verbose",
        "--color",
        "off",
        "--log-colors",
        "off",
    ]
    completed = subprocess.run(
        command,
        env=executor_env(backend, audit),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"llama-cli failed ({completed.returncode})\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for marker in (
        "using whole-token executor penzai.inference.v1",
        "model provisioned by penzai.inference.v1",
        "direct whole-token context ready",
    ):
        if marker not in completed.stderr:
            raise AssertionError(f"missing llama-cli marker {marker!r}:\n{completed.stderr}")
    if " OK!" not in completed.stdout:
        raise AssertionError(f"llama-cli did not emit mock winners:\n{completed.stdout}")
    assert_audit(audit, [5, 6], 256)


def unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post_json(url: str, payload: dict[str, Any], timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    encoded = json.dumps(payload).encode("utf-8")
    http_request = request.Request(url, data=encoded, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except error.HTTPError as http_error:
        body = json.loads(http_error.read())
        return http_error.code, body


def wait_for_server(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"llama-server exited during startup ({process.returncode})\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except (error.URLError, TimeoutError):
            pass
        time.sleep(0.05)
    raise AssertionError("llama-server did not become healthy")


def run_server(
    binary: Path,
    backend: Path,
    model: Path,
    audit: Path,
    prompt: str,
    launcher: Path | None = None,
) -> None:
    port = unused_tcp_port()
    if launcher is None:
        command = [
            str(binary),
            "--device",
            "penzai",
            "--model",
            str(model),
            "--parallel",
            "1",
            "--ctx-size",
            "256",
            "--batch-size",
            "32",
            "--ubatch-size",
            "32",
            "--temp",
            "0",
            "--no-warmup",
            "--verbose",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-colors",
            "off",
        ]
    else:
        command = [
            str(launcher),
            "serve",
            "--model",
            str(model),
            "--context",
            "256",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    environment = executor_env(backend, audit)
    if launcher is not None:
        environment["PENZAI_LLAMA_SERVER"] = str(binary)
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_server(port, process)

        status, body = post_json(
            f"http://127.0.0.1:{port}/completion",
            {
                "prompt": prompt,
                "n_predict": 2,
                "temperature": 0.8,
                "cache_prompt": False,
            },
        )
        if status != 400:
            raise AssertionError(f"non-greedy request was not rejected: status={status}, body={body}")
        if any(event.get("event") == "execute" for event in read_audit(audit)):
            raise AssertionError("non-greedy server request reached the executor")

        status, body = post_json(
            f"http://127.0.0.1:{port}/completion",
            {
                "prompt": prompt,
                "n_predict": 2,
                "temperature": 0.0,
                "cache_prompt": False,
            },
        )
        if status != 200:
            raise AssertionError(f"greedy completion failed: status={status}, body={body}")
        if body.get("content") != " OK!":
            raise AssertionError(f"llama-server did not return mock winners: {body}")
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise AssertionError(f"llama-server did not stop after SIGTERM\nstdout:\n{stdout}\nstderr:\n{stderr}")

    if process.returncode != 0:
        raise AssertionError(
            f"llama-server failed ({process.returncode})\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    if launcher is None:
        for marker in (
            "using whole-token executor penzai.inference.v1",
            "model provisioned by penzai.inference.v1",
            "direct whole-token context ready",
        ):
            if marker not in stderr:
                raise AssertionError(f"missing llama-server marker {marker!r}:\n{stderr}")
    assert_audit(audit, [5, 6], 256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--penzai", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--cpu-backend", type=Path, required=True)
    parser.add_argument("--logits-probe", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.llama_cli,
        args.llama_server,
        args.penzai,
        args.backend,
        args.cpu_backend,
        args.logits_probe,
    ):
        if not path.is_file():
            parser.error(f"not a file: {path}")

    prompt = " ".join(["a"] * 10)
    with tempfile.TemporaryDirectory(prefix="penzai-llama-smoke-") as temporary:
        work = Path(temporary)
        model = work / "tiny-qwen3.gguf"
        write_tiny_qwen3(model)
        run_logits_probe(
            args.logits_probe,
            args.cpu_backend,
            args.backend,
            model,
            work / "logits-probe.jsonl",
        )
        run_cli(args.llama_cli, args.backend, model, work / "cli.jsonl", prompt)
        run_server(args.llama_server, args.backend, model, work / "server.jsonl", prompt)
        run_server(
            args.llama_server,
            args.backend,
            model,
            work / "penzai-server.jsonl",
            prompt,
            launcher=args.penzai,
        )

    print("llama inference smoke: ok")


if __name__ == "__main__":
    main()
