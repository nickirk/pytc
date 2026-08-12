"""Protocol tests for the periodic batched-FFT benchmark."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pytc.pbc.fft_benchmark import (
    DEFAULT_K1_TARGET,
    FFTBenchmarkCase,
    K1ProjectionTarget,
    fft_flops,
    main,
    projected_k1_wall_seconds,
    projected_wall_seconds,
    recommended_cases,
)


class TestFFTBenchmarkProtocol(unittest.TestCase):
    def test_production_mesh_flop_count_matches_cost_card(self):
        self.assertAlmostEqual(fft_flops((57, 57, 57)) / 1e6, 16.2, places=1)

    def test_case_resolves_panel_and_batch_shapes(self):
        case = FFTBenchmarkCase(panel_size=4, channel_batch=8, repeats=2)
        self.assertEqual(case.batch_size, 32)
        self.assertEqual(case.array_shape, (4, 8, 57, 57, 57))

    def test_case_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            FFTBenchmarkCase(mesh=(57, 57, 0))
        with self.assertRaises(ValueError):
            FFTBenchmarkCase(panel_size=0)
        with self.assertRaises(ValueError):
            FFTBenchmarkCase(channel_batch=True)

    def test_projection_uses_measured_transform_rate(self):
        self.assertEqual(projected_wall_seconds(10, 2.0), 5.0)
        with self.assertRaises(ValueError):
            projected_wall_seconds(1, 0.0)

    def test_k1_projection_keeps_forward_and_inverse_rates_separate(self):
        target = K1ProjectionTarget(n_atoms=2, n_mu=10, basis="test")
        projected = projected_k1_wall_seconds(2.0, 4.0, target)
        expected = target.forward_transform_count / 2.0
        expected += target.inverse_transform_count / 4.0
        self.assertEqual(projected["total_seconds"], expected)
        self.assertEqual(
            projected["effective_transforms_per_second"],
            target.total_transform_count / expected,
        )

    def test_default_projection_target_records_fixture_and_formulas(self):
        receipt = DEFAULT_K1_TARGET.receipt()
        self.assertEqual(receipt["n_atoms"], 54)
        self.assertEqual(receipt["n_mu"], 15_660)
        self.assertEqual(receipt["basis"], "cc-pVTZ")
        self.assertIsNone(receipt["pseudo"])
        self.assertEqual(receipt["forward_channel_count"], 163)
        self.assertEqual(receipt["inverse_channel_count"], 668)
        self.assertEqual(receipt["forward_transform_count"], 2_552_580)
        self.assertEqual(receipt["inverse_transform_count"], 10_460_880)
        self.assertEqual(receipt["total_transform_count"], 13_013_460)

    def test_projection_target_rejects_ambiguous_fixture_fields(self):
        with self.assertRaises(ValueError):
            K1ProjectionTarget(n_atoms=0)
        with self.assertRaises(ValueError):
            K1ProjectionTarget(n_mu=1.5)
        with self.assertRaises(ValueError):
            K1ProjectionTarget(basis="")
        with self.assertRaises(ValueError):
            K1ProjectionTarget(pseudo="")

    def test_recommended_cases_have_unique_increasing_batches(self):
        batches = [case.batch_size for case in recommended_cases()]
        self.assertEqual(batches, sorted(set(batches)))
        self.assertEqual(batches, [1, 8, 32, 128, 512])

    def test_cli_creates_output_parent(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "matrix.json"
            args = [
                "--emit-matrix",
                "--n-atoms",
                "2",
                "--n-mu",
                "10",
                "--basis",
                "test-basis",
                "--pseudo",
                "test-pseudo",
                "--output",
                str(output),
            ]
            self.assertEqual(main(args), 0)
            self.assertTrue(output.is_file())
            matrix = json.loads(output.read_text())
            target = matrix["projection_target"]
            self.assertEqual(target["n_atoms"], 2)
            self.assertEqual(target["n_mu"], 10)
            self.assertEqual(target["basis"], "test-basis")
            self.assertEqual(target["pseudo"], "test-pseudo")


if __name__ == "__main__":
    unittest.main()
