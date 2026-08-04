"""Tests for FFT-friendly grid selection (pytc.pbc.fft_mesh)."""

import unittest

from pytc.pbc.fft_mesh import (
    DEFAULT_FFT_RADICES,
    describe_fft_mesh,
    factorize,
    good_fft_mesh,
    good_fft_size,
    is_fft_friendly,
)


class TestFactorize(unittest.TestCase):
    def test_known_factorizations(self):
        for n, expected in [
            (1, ()), (2, (2,)), (64, (2,) * 6), (72, (2, 2, 2, 3, 3)),
            (76, (2, 2, 19)), (80, (2, 2, 2, 2, 5)), (84, (2, 2, 3, 7)),
            (19, (19,)), (293, (293,)),
        ]:
            with self.subTest(n=n):
                self.assertEqual(factorize(n), expected)

    def test_product_reconstructs_input(self):
        # Independent of the table above: the factors must multiply back.
        for n in range(1, 500):
            product = 1
            for f in factorize(n):
                product *= f
            self.assertEqual(product, n, f"factorize({n}) does not reproduce n")

    def test_rejects_non_positive(self):
        for bad in (0, -1, -76):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                factorize(bad)


class TestIsFftFriendly(unittest.TestCase):
    def test_the_production_mesh_is_hostile(self):
        # 76 = 2^2 x 19. This is the case that motivated the module.
        self.assertFalse(is_fft_friendly(76))

    def test_smooth_neighbours_are_friendly(self):
        for n in (64, 72, 80, 84, 96, 100):
            with self.subTest(n=n):
                self.assertTrue(is_fft_friendly(n))

    def test_one_is_friendly(self):
        self.assertTrue(is_fft_friendly(1))

    def test_agrees_with_factorize_over_a_range(self):
        # Cross-check against an independent definition rather than a table.
        for n in range(1, 400):
            expected = all(f in DEFAULT_FFT_RADICES for f in factorize(n))
            self.assertEqual(is_fft_friendly(n), expected, f"disagreement at n={n}")

    def test_radices_are_configurable(self):
        self.assertFalse(is_fft_friendly(11))
        self.assertTrue(is_fft_friendly(11, radices=(2, 3, 5, 7, 11)))

    def test_rejects_bad_radices(self):
        with self.assertRaises(ValueError):
            is_fft_friendly(8, radices=())
        with self.assertRaises(ValueError):
            is_fft_friendly(8, radices=(1, 2))


class TestGoodFftSize(unittest.TestCase):
    def test_the_motivating_case(self):
        # 77 = 7 x 11, 78 = 2 x 3 x 13, 79 prime, so 80 is the answer.
        self.assertEqual(good_fft_size(76), 80)

    def test_friendly_sizes_are_unchanged(self):
        for n in (1, 2, 64, 72, 80, 84):
            with self.subTest(n=n):
                self.assertEqual(good_fft_size(n), n)

    def test_never_rounds_down(self):
        # The accuracy guarantee: the returned grid is never coarser.
        for n in range(1, 400):
            self.assertGreaterEqual(good_fft_size(n), n, f"rounded down at n={n}")

    def test_result_is_always_friendly(self):
        for n in range(1, 400):
            self.assertTrue(is_fft_friendly(good_fft_size(n)), f"unfriendly at n={n}")

    def test_result_is_minimal(self):
        # Nothing strictly between n and the answer may be friendly.
        for n in range(1, 400):
            got = good_fft_size(n)
            for candidate in range(n, got):
                self.assertFalse(
                    is_fft_friendly(candidate),
                    f"good_fft_size({n})={got} skipped friendly {candidate}",
                )

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            good_fft_size(0)


class TestGoodFftMesh(unittest.TestCase):
    def test_production_mesh(self):
        self.assertEqual(good_fft_mesh((76, 76, 76)), (80, 80, 80))

    def test_axes_are_independent(self):
        # A separable transform penalises only the awkward axis, so only that
        # axis should move.
        self.assertEqual(good_fft_mesh((64, 76, 72)), (64, 80, 72))

    def test_friendly_mesh_unchanged(self):
        self.assertEqual(good_fft_mesh((64, 72, 80)), (64, 72, 80))

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            good_fft_mesh(())


class TestDescribeFftMesh(unittest.TestCase):
    def test_silent_on_friendly_mesh(self):
        self.assertIsNone(describe_fft_mesh((64, 72, 80)))

    def test_reports_hostile_mesh(self):
        message = describe_fft_mesh((76, 76, 76))
        self.assertIsNotNone(message)
        self.assertIn("76 = 2 x 2 x 19", message)
        self.assertIn("(80, 80, 80)", message)

    def test_states_the_cost_of_the_suggestion(self):
        # The suggestion is not free; the message must say so rather than
        # reading as an unqualified win. 80^3/76^3 = 1.17.
        message = describe_fft_mesh((76, 76, 76))
        self.assertIn("1.17x the grid points", message)
        self.assertIn("not free", message)

    def test_partial_hostility_named_once(self):
        message = describe_fft_mesh((64, 76, 72))
        self.assertIn("76 = 2 x 2 x 19", message)
        self.assertNotIn("64 =", message)


class TestBuildPathEmitsTheWarning(unittest.TestCase):
    """The check must fire from the build path, not merely exist.

    A diagnostic that is never reached is the defect it was written to
    prevent, so this asserts the wiring rather than the helper.
    """

    @staticmethod
    def _diamond_111(mesh):
        from pyscf.pbc.gto import Cell

        cell = Cell()
        cell.atom = "C 0 0 0; C .8917 .8917 .8917"
        cell.a = "0 1.7834 1.7834\n1.7834 0 1.7834\n1.7834 1.7834 0"
        cell.unit = "A"
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pbe"
        cell.mesh = mesh
        cell.verbose = 0
        cell.build()
        return cell

    def _run_build(self, cell):
        from pytc.pbc import coulomb

        coulomb.build(
            cell, cell.make_kpts([1, 1, 1]), rank=2 * cell.nao_nr(),
            block_size=64, rtol=1e-4, selection_mode="streamed",
        )

    def test_real_build_warns_on_a_hostile_mesh(self):
        # Drives the actual build path rather than inspecting its source: a
        # diagnostic is only real if it reaches stderr from a real caller.
        import logging

        cell = self._diamond_111([19, 19, 19])
        with self.assertLogs("pytc.pbc.coulomb", level=logging.WARNING) as captured:
            self._run_build(cell)
        self.assertTrue(
            any("19 = 19" in line for line in captured.output),
            f"no FFT-mesh warning in {captured.output}",
        )
        self.assertTrue(any("(20, 20, 20)" in line for line in captured.output))

    def test_real_build_is_silent_on_a_friendly_mesh(self):
        # Negative control: without this, a warning that always fired would
        # pass the test above and be useless.
        import logging

        cell = self._diamond_111([20, 20, 20])
        logger = logging.getLogger("pytc.pbc.coulomb")
        with self.assertLogs(logger, level=logging.WARNING) as captured:
            logger.warning("sentinel so assertLogs has something to capture")
            self._run_build(cell)
        self.assertEqual(
            [line for line in captured.output if "FFT grid mesh" in line],
            [],
            f"unexpected FFT-mesh warning: {captured.output}",
        )


if __name__ == "__main__":
    unittest.main()
