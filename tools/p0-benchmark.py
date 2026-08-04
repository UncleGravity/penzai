#!/usr/bin/env python3
"""Run an immutable P0 architecture-measurement suite."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from typing import Any


ARTIFACT_SCHEMA_VERSION = 1
SUITE_VERSION = 1
RESULT_SCHEMA_VERSION = 1
DEFAULT_BATCH = 32
DEFAULT_UBATCH = 16
DEFAULT_MODELS = {
    "q1": "models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf",
    "q2": "models/Bonsai-Ternary-1.7B/Ternary-Bonsai-1.7B-Q2_0_g64.gguf",
}
INTEGER_FIELDS = {
    "schema",
    "prompt_tokens",
    "generated_tokens",
    "prefill_wall_ns",
    "first_decode_step_ns",
    "compute_ttft_ns",
    "output_ttft_ns",
    "steady_decode_ns",
    "steady_decode_count",
    "decode_wall_ns",
    "decode_device_ns",
    "decode_transport_ns",
    "decode_residual_ns",
}
LATENCY_METRICS = (
    "prefill_wall_ms",
    "compute_ttft_ms",
    "output_ttft_ms",
    "first_decode_step_ms",
    "steady_decode_ms",
    "decode_wall_ms_per_token",
    "decode_device_ms_per_token",
    "decode_transport_ms_per_token",
    "decode_residual_ms_per_token",
)


CASE_CATALOG: dict[str, dict[str, Any]] = {
    "prefill-p128": {
        "name": "prefill-p128",
        "axis": "prefill",
        "target": 128,
        "prompt_tokens": 128,
        "max_tokens": 1,
    },
    "prefill-p512": {
        "name": "prefill-p512",
        "axis": "prefill",
        "target": 512,
        "prompt_tokens": 512,
        "max_tokens": 1,
    },
    "decode-c0": {
        "name": "decode-c0",
        "axis": "decode",
        "target": 0,
        "prompt_tokens": 1,
        "max_tokens": 64,
    },
    "decode-c512": {
        "name": "decode-c512",
        "axis": "decode",
        "target": 512,
        "prompt_tokens": 512,
        "max_tokens": 64,
    },
    "decode-c2048": {
        "name": "decode-c2048",
        "axis": "decode",
        "target": 2048,
        "prompt_tokens": 2048,
        "max_tokens": 64,
    },
    "decode-c4096": {
        "name": "decode-c4096",
        "axis": "decode",
        "target": 4096,
        "prompt_tokens": 4096,
        "max_tokens": 64,
    },
}

SUITES: dict[str, dict[str, Any]] = {
    "characterize": {
        "profile_mode": "aggregate",
        "cases": (
            ("prefill-p128", 3),
            ("prefill-p512", 3),
            ("decode-c0", 3),
            ("decode-c512", 3),
            ("decode-c2048", 1),
        ),
        "optional_cases": (("decode-c4096", 1),),
    },
    "regression": {
        "profile_mode": "off",
        "cases": (
            ("prefill-p128", 5),
            ("decode-c0", 5),
            ("decode-c512", 5),
        ),
        "optional_cases": (),
    },
}


class HarnessError(Exception):
    pass


class ResumeRejected(HarnessError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=tuple(SUITES))
    parser.add_argument("--penzai", default="result/bin/penzai")
    parser.add_argument("--device", default="tcp:kria:29092")
    parser.add_argument("--q1-model", default=DEFAULT_MODELS["q1"])
    parser.add_argument("--q2-model", default=DEFAULT_MODELS["q2"])
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--ubatch", type=int)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--resume", type=pathlib.Path)
    args = parser.parse_args()
    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.batch is not None and args.batch < 1:
        parser.error("--batch must be positive")
    if args.ubatch is not None and args.ubatch < 1:
        parser.error("--ubatch must be positive")
    if args.output and args.resume:
        parser.error("--output and --resume are mutually exclusive")
    return args


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_fingerprint(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def make_manifest(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint_sha256": identity_fingerprint(identity),
        "identity": identity,
    }


def validate_resume_manifest(manifest: dict[str, Any], identity: dict[str, Any]) -> str:
    if set(manifest) != {"fingerprint_sha256", "identity"}:
        raise ResumeRejected("manifest structure is invalid")
    stored_identity = manifest["identity"]
    stored_fingerprint = manifest["fingerprint_sha256"]
    if not isinstance(stored_identity, dict) or not isinstance(stored_fingerprint, str):
        raise ResumeRejected("manifest types are invalid")
    if identity_fingerprint(stored_identity) != stored_fingerprint:
        raise ResumeRejected("stored manifest fingerprint is invalid")
    new_fingerprint = identity_fingerprint(identity)
    if stored_fingerprint != new_fingerprint or stored_identity != identity:
        raise ResumeRejected("current benchmark identity does not match the artifact")
    return stored_fingerprint


def write_json_atomic(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_checked(command: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and " " not in key:
            values[key] = value
    return values


def validate_capabilities(capabilities: dict[str, str]) -> None:
    required = {
        "receipt_status": "loaded",
        "bitstream_hash_verified": "true",
    }
    for key, expected in required.items():
        if capabilities.get(key) != expected:
            raise HarnessError(f"capability preflight failed: {key}={capabilities.get(key)!r}, expected {expected!r}")
    try:
        engine_mask = int(capabilities.get("engine_mask", ""), 16)
        format_mask = int(capabilities.get("format_mask", ""), 16)
    except ValueError as error:
        raise HarnessError("capability preflight failed: invalid engine or format mask") from error
    if engine_mask & 0x3 != 0x3:
        raise HarnessError("capability preflight failed: matmul and flash engines are not both resident")
    if format_mask & 0x3 != 0x3:
        raise HarnessError("capability preflight failed: resident GEMM does not advertise Q1 and Q2")
    if not capabilities.get("run_id") or len(capabilities.get("bitstream_sha256", "")) != 64:
        raise HarnessError("capability preflight failed: incomplete deployed identity")


def parse_benchmark_result(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.startswith("benchmark_result ")]
    if len(lines) != 1:
        raise HarnessError(f"expected one benchmark_result line, found {len(lines)}")
    fields: dict[str, Any] = {}
    for item in shlex.split(lines[0])[1:]:
        key, separator, value = item.partition("=")
        if not separator:
            raise HarnessError(f"malformed benchmark field: {item}")
        fields[key] = int(value) if key in INTEGER_FIELDS else value
    missing = INTEGER_FIELDS.difference(fields)
    if missing:
        raise HarnessError(f"missing benchmark fields: {sorted(missing)}")
    if fields["schema"] != RESULT_SCHEMA_VERSION:
        raise HarnessError(f"unsupported result schema {fields['schema']}")
    return fields


def selected_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    suite = SUITES[args.suite]
    available = dict(suite["cases"] + suite["optional_cases"])
    if args.selected_cases:
        requested = set(args.selected_cases)
        unknown = requested.difference(available)
        if unknown:
            raise HarnessError(f"cases are not available in {args.suite}: {sorted(unknown)}")
        names = [name for name in available if name in requested]
    else:
        names = [name for name, _ in suite["cases"]]

    cases: list[dict[str, Any]] = []
    for name in names:
        case = dict(CASE_CATALOG[name])
        case["repetitions"] = args.repeats if args.repeats is not None else available[name]
        cases.append(case)
    return cases


def diagnostic_overrides(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.selected_cases:
        overrides["case"] = [case["name"] for case in cases]
    if args.repeats is not None:
        overrides["repeats"] = args.repeats
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.ubatch is not None:
        overrides["ubatch"] = args.ubatch
    return overrides


def model_paths(root: pathlib.Path, args: argparse.Namespace) -> dict[str, pathlib.Path]:
    configured = {"q1": pathlib.Path(args.q1_model), "q2": pathlib.Path(args.q2_model)}
    models = {name: path if path.is_absolute() else (root / path).resolve() for name, path in configured.items()}
    for name, path in models.items():
        if not path.is_file():
            raise HarnessError(f"{name} model not found: {path}")
    return models


def build_identity(
    args: argparse.Namespace,
    penzai: pathlib.Path,
    models: dict[str, pathlib.Path],
    capability_response: str,
    capabilities: dict[str, str],
    runner_path: pathlib.Path,
) -> dict[str, Any]:
    cases = selected_cases(args)
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "suite": args.suite,
        "cases": cases,
        "batch": args.batch if args.batch is not None else DEFAULT_BATCH,
        "ubatch": args.ubatch if args.ubatch is not None else DEFAULT_UBATCH,
        "profile_mode": SUITES[args.suite]["profile_mode"],
        "device_endpoint": args.device,
        "device_capability_response": capability_response,
        "runner_sha256": sha256(runner_path),
        "penzai_sha256": sha256(penzai),
        "models": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sorted(models.items())
        },
        "device_capabilities": capabilities,
        "workload": {
            "prompt": " benchmark",
            "raw_prompt": True,
            "exact_tokens": True,
            "backend_sampling": True,
        },
        "overrides": diagnostic_overrides(args, cases),
    }


def expected_sample_keys(identity: dict[str, Any]) -> list[tuple[str, str, int]]:
    return [
        (case["name"], model, repeat)
        for case in identity["cases"]
        for repeat in range(1, case["repetitions"] + 1)
        for model in ("q1", "q2")
    ]


def sample_key(sample: dict[str, Any]) -> tuple[str, str, int]:
    return sample["case"], sample["model"], sample["repeat"]


def display_sample_key(key: tuple[str, str, int]) -> str:
    case, model, repeat = key
    return f"{case}/{model}/r{repeat:02d}"


def load_samples(path: pathlib.Path, expected: set[tuple[str, str, int]]) -> tuple[list[dict[str, Any]], set[tuple[str, str, int]]]:
    samples: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if not path.exists():
        return samples, completed
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            key = sample_key(sample)
            if key not in expected:
                raise ResumeRejected(f"unexpected sample key at {path}:{line_number}: {key}")
            if key in completed:
                raise ResumeRejected(f"duplicate sample key at {path}:{line_number}: {key}")
            completed.add(key)
            samples.append(sample)
    return samples, completed


def latency_metrics(result: dict[str, Any]) -> dict[str, float]:
    generated = result["generated_tokens"]
    steady_count = result["steady_decode_count"]
    metrics = {
        "prefill_wall_ms": result["prefill_wall_ns"] / 1e6,
        "compute_ttft_ms": result["compute_ttft_ns"] / 1e6,
        "output_ttft_ms": result["output_ttft_ns"] / 1e6,
        "first_decode_step_ms": result["first_decode_step_ns"] / 1e6,
        "decode_wall_ms_per_token": result["decode_wall_ns"] / generated / 1e6,
        "decode_device_ms_per_token": result["decode_device_ns"] / generated / 1e6,
        "decode_transport_ms_per_token": result["decode_transport_ns"] / generated / 1e6,
        "decode_residual_ms_per_token": result["decode_residual_ns"] / generated / 1e6,
    }
    if steady_count:
        metrics["steady_decode_ms"] = result["steady_decode_ns"] / steady_count / 1e6
    return {name: metrics[name] for name in LATENCY_METRICS if name in metrics}


def summarize_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[(sample["case"], sample["model"])].append(sample)

    summaries: list[dict[str, Any]] = []
    for (case_name, model), group in sorted(groups.items()):
        metrics: dict[str, dict[str, float]] = {}
        for name in LATENCY_METRICS:
            values = [sample["metrics"][name] for sample in group if name in sample["metrics"]]
            if not values:
                continue
            metrics[name] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        summary: dict[str, Any] = {
            "case": case_name,
            "model": model,
            "samples": len(group),
            "metrics": metrics,
        }
        steady = metrics.get("steady_decode_ms")
        if steady and steady["median"] > 0:
            summary["throughput_tokens_s"] = 1000.0 / steady["median"]
        summaries.append(summary)
    return summaries


def build_summary(
    identity: dict[str, Any],
    fingerprint: str,
    samples: list[dict[str, Any]],
    run_validated: bool,
) -> dict[str, Any]:
    expected = expected_sample_keys(identity)
    completed_set = {sample_key(sample) for sample in samples}
    completed = [key for key in expected if key in completed_set]
    missing = [key for key in expected if key not in completed_set]
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "suite": identity["suite"],
        "identity_fingerprint_sha256": fingerprint,
        "complete": not missing and run_validated,
        "run_validated": run_validated,
        "completed_samples": len(completed),
        "expected_samples": len(expected),
        "completed_sample_keys": [display_sample_key(key) for key in completed],
        "expected_sample_keys": [display_sample_key(key) for key in expected],
        "missing_sample_keys": [display_sample_key(key) for key in missing],
        "groups": summarize_samples(samples),
    }


def validate_result(result: dict[str, Any], case: dict[str, Any], profile_mode: str, raw_file: pathlib.Path) -> None:
    if result["prompt_tokens"] != case["prompt_tokens"]:
        raise HarnessError(f"prompt token mismatch; see {raw_file}")
    if result["generated_tokens"] != case["max_tokens"]:
        raise HarnessError(f"generated token mismatch; see {raw_file}")
    expected_accounting = "ok" if profile_mode == "aggregate" else "unprofiled"
    if result["accounting"] != expected_accounting:
        raise HarnessError(f"accounting is {result['accounting']}, expected {expected_accounting}; see {raw_file}")


def run_sample(
    root: pathlib.Path,
    penzai: pathlib.Path,
    device: str,
    model_name: str,
    model_path: pathlib.Path,
    case: dict[str, Any],
    repeat: int,
    identity: dict[str, Any],
    raw_dir: pathlib.Path,
) -> dict[str, Any]:
    stem = f"{case['name']}-{model_name}-r{repeat:02d}"
    raw_file = raw_dir / f"{stem}.txt"
    context_size = case["prompt_tokens"] + case["max_tokens"]
    command = [
        str(penzai),
        "run",
        "--device",
        device,
        "--model",
        str(model_path),
        "--prompt",
        identity["workload"]["prompt"],
        "--raw-prompt",
        "--prompt-tokens",
        str(case["prompt_tokens"]),
        "--max-tokens",
        str(case["max_tokens"]),
        "--context",
        str(context_size),
        "--batch",
        str(identity["batch"]),
        "--ubatch",
        str(identity["ubatch"]),
        "--exact-tokens",
        "--backend-sampling",
    ]
    if identity["profile_mode"] == "aggregate":
        command.append("--prof")
    completed = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    raw_file.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise HarnessError(f"benchmark failed ({completed.returncode}); see {raw_file}")
    result = parse_benchmark_result(completed.stdout)
    validate_result(result, case, identity["profile_mode"], raw_file)
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model": model_name,
        "case": case["name"],
        "repeat": repeat,
        "raw": str(raw_file.relative_to(raw_dir.parent)),
        "result": result,
        "metrics": latency_metrics(result),
    }


def verify_file_identity(identity: dict[str, Any], penzai: pathlib.Path, models: dict[str, pathlib.Path]) -> None:
    if sha256(penzai) != identity["penzai_sha256"]:
        raise HarnessError("penzai executable changed during the run")
    for name, path in models.items():
        if sha256(path) != identity["models"][name]["sha256"]:
            raise HarnessError(f"{name} model changed during the run")


def capture_end_capabilities(
    command: list[str],
    root: pathlib.Path,
    output: pathlib.Path,
    expected_response: str,
    required: bool,
) -> None:
    try:
        raw = run_checked(command, root).stdout
    except subprocess.CalledProcessError:
        if required:
            raise HarnessError("could not read ending device capabilities")
        return
    (output / "capabilities-end.txt").write_text(raw, encoding="utf-8")
    if raw != expected_response:
        raise HarnessError("deployed capabilities changed during the benchmark run")


def main() -> int:
    args = parse_args()
    root = pathlib.Path.cwd().resolve()
    runner_path = pathlib.Path(__file__).resolve()
    penzai = pathlib.Path(args.penzai)
    if not penzai.is_absolute():
        penzai = (root / penzai).resolve()
    if not penzai.is_file():
        raise SystemExit(f"penzai executable not found: {penzai}")

    try:
        models = model_paths(root, args)
        capability_command = [str(penzai), "capabilities", "--device", args.device]
        capability_start_raw = run_checked(capability_command, root).stdout
        capabilities = parse_key_values(capability_start_raw)
        validate_capabilities(capabilities)
        identity = build_identity(args, penzai, models, capability_start_raw, capabilities, runner_path)
        manifest = make_manifest(identity)
        fingerprint = manifest["fingerprint_sha256"]

        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        resuming = args.resume is not None
        output = (
            args.resume
            if resuming
            else args.output or root / "benchmarks" / "p0" / "runs" / f"{timestamp}-{args.suite}-{fingerprint[:12]}"
        ).resolve()
        raw_dir = output / "raw"
        manifest_path = output / "manifest.json"
        sample_path = output / "samples.jsonl"
        summary_path = output / "summary.json"

        if resuming:
            if not output.is_dir() or not raw_dir.is_dir() or not manifest_path.is_file():
                raise ResumeRejected(f"resume artifact is incomplete: {output}")
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fingerprint = validate_resume_manifest(stored_manifest, identity)
        else:
            raw_dir.mkdir(parents=True, exist_ok=False)
            write_json_atomic(manifest_path, manifest)
            (output / "capabilities-start.txt").write_text(capability_start_raw, encoding="utf-8")
            sample_path.touch(exist_ok=False)

        expected = set(expected_sample_keys(identity))
        samples, completed_keys = load_samples(sample_path, expected)
        write_json_atomic(summary_path, build_summary(identity, fingerprint, samples, run_validated=False))

        interrupted = False
        failure: HarnessError | None = None
        loop_finished = False
        try:
            with sample_path.open("a", encoding="utf-8") as sample_stream:
                for case in identity["cases"]:
                    for repeat in range(1, case["repetitions"] + 1):
                        for model_name in ("q1", "q2"):
                            key = (case["name"], model_name, repeat)
                            if key in completed_keys:
                                continue
                            print(f"[{repeat}/{case['repetitions']}] {case['name']} {model_name}", flush=True)
                            sample = run_sample(
                                root,
                                penzai,
                                args.device,
                                model_name,
                                models[model_name],
                                case,
                                repeat,
                                identity,
                                raw_dir,
                            )
                            samples.append(sample)
                            completed_keys.add(key)
                            sample_stream.write(json.dumps(sample, sort_keys=True) + "\n")
                            sample_stream.flush()
                            os.fsync(sample_stream.fileno())
                            write_json_atomic(
                                summary_path,
                                build_summary(identity, fingerprint, samples, run_validated=False),
                            )
            loop_finished = True
        except KeyboardInterrupt:
            interrupted = True
        except HarnessError as error:
            failure = error
        finally:
            write_json_atomic(summary_path, build_summary(identity, fingerprint, samples, run_validated=False))
            verify_file_identity(identity, penzai, models)
            capture_end_capabilities(
                capability_command,
                root,
                output,
                capability_start_raw,
                required=loop_finished and not interrupted and failure is None,
            )
            if loop_finished and not interrupted and failure is None:
                write_json_atomic(summary_path, build_summary(identity, fingerprint, samples, run_validated=True))

        if interrupted:
            print(f"interrupted; partial artifact={output}", file=sys.stderr)
            return 130
        if failure is not None:
            raise failure
        print(f"artifact={output}")
        return 0
    except (HarnessError, ResumeRejected, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
