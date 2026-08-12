import importlib.util
import pathlib
import unittest


RUNNER = pathlib.Path(__file__).with_name("p0-benchmark.py")
SPEC = importlib.util.spec_from_file_location("p0_benchmark", RUNNER)
assert SPEC is not None and SPEC.loader is not None
p0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p0)


def identity() -> dict:
    return {
        "artifact_schema_version": p0.ARTIFACT_SCHEMA_VERSION,
        "suite_version": p0.SUITE_VERSION,
        "suite": "characterize",
        "cases": [
            {
                "name": "decode-c0",
                "axis": "decode",
                "target": 0,
                "prompt_tokens": 1,
                "max_tokens": 64,
                "repetitions": 1,
            }
        ],
        "runner_sha256": "runner-a",
    }


def attention_line(**overrides: object) -> str:
    fields: dict[str, object] = {name: 0 for name in p0.ATTENTION_INTEGER_FIELDS}
    fields.update(
        {
            "schema": p0.ATTENTION_RESULT_SCHEMA_VERSION,
            "phase": "decode",
            "backend": "pl",
            "path": "direct",
            "n_heads": 16,
            "n_head_kv": 8,
            "head_dim_q": 128,
            "head_dim_v": 128,
            "n_tokens": 1,
            "calls": 1,
            "fclk_hz": 300_000_000,
        }
    )
    fields.update(overrides)
    return "attention_result " + " ".join(f"{key}={value}" for key, value in fields.items())


def sample(model: str, repeat: int, steady_ms: float) -> dict:
    metrics = {name: 0.0 for name in p0.LATENCY_METRICS}
    metrics["steady_decode_ms"] = steady_ms
    metrics["prefill_wall_ms"] = steady_ms * 2
    return {
        "case": "decode-c0",
        "model": model,
        "repeat": repeat,
        "metrics": metrics,
    }


class ManifestTests(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_runner_change_rejects_resume(self) -> None:
        original = identity()
        reordered = {key: original[key] for key in reversed(original)}
        self.assertEqual(p0.identity_fingerprint(original), p0.identity_fingerprint(reordered))

        manifest = p0.make_manifest(original)
        self.assertEqual(manifest["fingerprint_sha256"], p0.validate_resume_manifest(manifest, original))

        changed = dict(original)
        changed["runner_sha256"] = "runner-b"
        with self.assertRaises(p0.ResumeRejected):
            p0.validate_resume_manifest(manifest, changed)


class SummaryTests(unittest.TestCase):
    def test_summary_uses_conventional_median_and_range(self) -> None:
        samples = [sample("q1", index, value) for index, value in enumerate((1.0, 2.0, 10.0, 20.0), start=1)]
        group = p0.summarize_samples(samples)[0]
        steady = group["metrics"]["steady_decode_ms"]
        self.assertEqual(6.0, steady["median"])
        self.assertEqual(1.0, steady["min"])
        self.assertEqual(20.0, steady["max"])
        self.assertAlmostEqual(1000.0 / 6.0, group["throughput_tokens_s"])
        self.assertNotIn("p95", steady)

    def test_partial_summary_lists_completed_and_expected_samples(self) -> None:
        manifest_identity = identity()
        fingerprint = p0.identity_fingerprint(manifest_identity)
        partial = p0.build_summary(
            manifest_identity,
            fingerprint,
            [sample("q1", 1, 4.0)],
            run_validated=False,
        )
        self.assertFalse(partial["complete"])
        self.assertFalse(partial["run_validated"])
        self.assertEqual(1, partial["completed_samples"])
        self.assertEqual(2, partial["expected_samples"])
        self.assertEqual(["decode-c0/q2/r01"], partial["missing_sample_keys"])

        complete = p0.build_summary(
            manifest_identity,
            fingerprint,
            [sample("q1", 1, 4.0), sample("q2", 1, 5.0)],
            run_validated=True,
        )
        self.assertTrue(complete["complete"])
        self.assertTrue(complete["run_validated"])
        self.assertEqual([], complete["missing_sample_keys"])


class AttentionResultTests(unittest.TestCase):
    def test_parser_preserves_shapes_and_metrics_use_only_pl_counters(self) -> None:
        text = "\n".join(
            (
                attention_line(
                    backend="ps",
                    path="software",
                    calls=2,
                    valid_qkv_pairs=10,
                    processed_qkv_pairs=12,
                    valid_qhkv_updates=160,
                    processed_qhkv_updates=192,
                ),
                attention_line(
                    n_tokens=2,
                    calls=3,
                    kernel_runs=3,
                    cycles=24_000,
                    valid_qkv_pairs=100,
                    processed_qkv_pairs=125,
                    valid_qhkv_updates=1_600,
                    processed_qhkv_updates=2_000,
                ),
            )
        )
        records = p0.parse_attention_results(text)
        self.assertEqual(2, len(records["decode"]))
        self.assertEqual({"ps", "pl"}, {record["backend"] for record in records["decode"]})
        self.assertEqual({1, 2}, {record["n_tokens"] for record in records["decode"]})

        metrics = p0.attention_metrics(records)
        self.assertEqual(60.0, metrics["decode_attention_pl_call_pct"])
        self.assertEqual(8_000.0, metrics["decode_attention_cycles_per_kernel_run"])
        self.assertEqual(15.0, metrics["decode_attention_cycles_per_valid_qhkv_update"])
        self.assertEqual(12.0, metrics["decode_attention_cycles_per_processed_qhkv_update"])
        self.assertEqual(20.0, metrics["decode_attention_mvalid_qhkv_per_s"])
        self.assertEqual(80.0, metrics["decode_attention_valid_density_pct"])

    def test_zero_work_does_not_invent_efficiency(self) -> None:
        records = p0.parse_attention_results(attention_line())
        metrics = p0.attention_metrics(records)
        self.assertEqual(100.0, metrics["decode_attention_pl_call_pct"])
        self.assertNotIn("decode_attention_cycles_per_valid_qhkv_update", metrics)
        self.assertNotIn("decode_attention_mvalid_qhkv_per_s", metrics)

    def test_duplicate_shape_and_missing_field_are_rejected(self) -> None:
        line = attention_line()
        with self.assertRaises(p0.HarnessError):
            p0.parse_attention_results(f"{line}\n{line}")
        missing_cycles = " ".join(item for item in line.split() if not item.startswith("cycles="))
        with self.assertRaises(p0.HarnessError):
            p0.parse_attention_results(missing_cycles)
        with self.assertRaises(p0.HarnessError):
            p0.parse_attention_results(attention_line(path="unknown"))


if __name__ == "__main__":
    unittest.main()
