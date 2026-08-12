"""Protocol tests for the periodic batched-FFT benchmark."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pytc.pbc.fft_benchmark import (
    FFTBenchmarkCase,
    K1_FORWARD_TRANSFORMS,
    K1_INVERSE_TRANSFORMS,
    K1_TOTAL_TRANSFORMS,
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
        self.assertEqual(projected_wall_seconds(K1_TOTAL_TRANSFORMS, 2.0), 6_506_730)
        with self.assertRaises(ValueError):
            projected_wall_seconds(1, 0.0)

    def test_k1_projection_keeps_forward_and_inverse_rates_separate(self):
        projected = projected_k1_wall_seconds(2.0, 4.0)
        expected = K1_FORWARD_TRANSFORMS / 2.0 + K1_INVERSE_TRANSFORMS / 4.0
        self.assertEqual(projected["total_seconds"], expected)
        self.assertEqual(
            projected["effective_transforms_per_second"],
            K1_TOTAL_TRANSFORMS / expected,
        )

    def test_recommended_cases_have_unique_increasing_batches(self):
        batches = [case.batch_size for case in recommended_cases()]
        self.assertEqual(batches, sorted(set(batches)))
        self.assertEqual(batches, [1, 8, 32, 128, 512])

    def test_cli_creates_output_parent(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "matrix.json"
            self.assertEqual(main(["--emit-matrix", "--output", str(output)]), 0)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
