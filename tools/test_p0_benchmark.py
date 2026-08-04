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
        "artifact_schema_version": 1,
        "suite_version": 1,
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


if __name__ == "__main__":
    unittest.main()
