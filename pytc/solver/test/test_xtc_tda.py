import unittest

import numpy as np

from pytc.solver.xtc_tda import (
    QuadraticTensor,
    StaticXTCTDAModel,
    build_reference_singles_couplings,
    build_rhf_fock,
    build_singlet_tda,
    build_tda_polynomial,
    excitation_derivative,
    optimize_form_b_scalar,
    project_hamiltonian,
    solve_biorthogonal,
)


class TestQuadraticTensor(unittest.TestCase):
    def test_symmetric_stencil_reconstructs_quadratic(self):
        constant = np.array([[1.0, -2.0], [0.5, 3.0]])
        linear = np.array([[0.2, 0.4], [-0.3, 0.1]])
        quadratic = np.array([[0.8, -0.6], [0.2, 0.4]])
        step = 0.25

        def value(q):
            return constant + q * linear + 0.5 * q**2 * quadratic

        tensor = QuadraticTensor.from_symmetric_stencil(
            value(-step),
            value(0.0),
            value(step),
            step=step,
        )
        np.testing.assert_allclose(tensor.constant, constant)
        np.testing.assert_allclose(tensor.linear, linear)
        np.testing.assert_allclose(tensor.quadratic, quadratic)
        np.testing.assert_allclose(tensor.value(0.37), value(0.37))
        np.testing.assert_allclose(
            tensor.derivative(0.37),
            linear + 0.37 * quadratic,
        )

    def test_mismatched_components_are_rejected(self):
        with self.assertRaises(ValueError):
            QuadraticTensor(np.zeros(2), np.zeros(3), np.zeros(2))


class TestNonHermitianTDA(unittest.TestCase):
    def test_tda_preserves_integral_orientation(self):
        h1 = np.zeros((3, 3))
        h1[1, 2] = 0.4
        h1[2, 1] = -0.2
        eri = np.zeros((3, 3, 3, 3))
        eri[1, 0, 0, 2] = 0.3
        eri[1, 2, 0, 0] = -0.1
        eri[2, 0, 0, 1] = -0.4
        eri[2, 1, 0, 0] = 0.2
        matrix = build_singlet_tda(h1, eri, 1)
        fock = build_rhf_fock(h1, eri, 1)
        self.assertFalse(np.allclose(matrix, matrix.conj().T))
        self.assertAlmostEqual(
            matrix[0, 1],
            fock[1, 2]
            + 2.0 * eri[1, 0, 0, 2]
            - eri[1, 2, 0, 0],
        )
        self.assertAlmostEqual(
            matrix[1, 0],
            fock[2, 1]
            + 2.0 * eri[2, 0, 0, 1]
            - eri[2, 1, 0, 0],
        )

    def test_reference_singles_sides_are_independent(self):
        h1 = np.zeros((2, 2))
        h1[0, 1] = 0.7
        h1[1, 0] = -0.2
        eri = np.zeros((2, 2, 2, 2))
        left, right = build_reference_singles_couplings(h1, eri, 1)
        self.assertAlmostEqual(left[0], np.sqrt(2.0) * 0.7)
        self.assertAlmostEqual(right[0], -np.sqrt(2.0) * 0.2)
        self.assertNotAlmostEqual(left[0], right[0])

    def test_biorthogonal_solver_residuals_and_derivatives(self):
        matrix = np.array([[1.0, 0.4], [0.1, 2.0]])
        eigensystem = solve_biorthogonal(matrix)
        self.assertLess(np.max(eigensystem.right_residual), 1.0e-12)
        self.assertLess(np.max(eigensystem.left_residual), 1.0e-12)
        self.assertLess(eigensystem.biorthogonality_error, 1.0e-12)
        derivative = np.array([[0.3, -0.2], [0.1, 0.5]])
        expected = np.array(
            [
                eigensystem.left[:, root].conj().T
                @ derivative
                @ eigensystem.right[:, root]
                for root in range(2)
            ]
        )
        np.testing.assert_allclose(
            excitation_derivative(eigensystem, derivative),
            expected,
        )

    def test_tda_polynomial_transforms_each_component(self):
        h1 = QuadraticTensor(
            np.diag([0.0, 1.0]),
            np.diag([0.0, 0.2]),
            np.diag([0.0, -0.1]),
        )
        eri = QuadraticTensor(
            np.zeros((2, 2, 2, 2)),
            np.zeros((2, 2, 2, 2)),
            np.zeros((2, 2, 2, 2)),
        )
        tda = build_tda_polynomial(h1, eri, 1)
        self.assertAlmostEqual(tda.value(0.5)[0, 0], 1.0875)


class TestProjectedHamiltonian(unittest.TestCase):
    def test_projection_does_not_hermitianize(self):
        norb = 2
        nelec = (1, 1)
        h1 = np.array([[0.1, 0.7], [-0.2, 1.0]])
        eri = np.zeros((norb, norb, norb, norb))
        eri[1, 0, 0, 1] = 0.3
        eri[0, 1, 1, 0] = -0.1
        vectors = np.eye(4)
        matrix = project_hamiltonian(
            h1,
            eri,
            0.25,
            vectors,
            norb,
            nelec,
        )
        self.assertEqual(matrix.shape, (4, 4))
        self.assertFalse(np.allclose(matrix, matrix.conj().T))

    def test_projection_requires_orthonormal_vectors(self):
        with self.assertRaises(ValueError):
            project_hamiltonian(
                np.eye(2),
                np.zeros((2, 2, 2, 2)),
                0.0,
                np.ones((4, 2)),
                2,
                (1, 1),
            )


class TestBalancedFormB(unittest.TestCase):
    @staticmethod
    def make_model():
        tda = QuadraticTensor(
            np.diag([1.0, 2.0]),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
        )
        constant = np.zeros((4, 4))
        linear = np.zeros((4, 4))
        for row, column in ((3, 1), (1, 3), (3, 0), (0, 3)):
            constant[row, column] = -0.3
            linear[row, column] = 1.0
        projected = QuadraticTensor(
            constant,
            linear,
            np.zeros((4, 4)),
        )
        return StaticXTCTDAModel(
            tda,
            projected,
            number_singles=2,
        )

    def test_form_b_has_equal_four_channel_weights(self):
        evaluation = self.make_model().evaluate(0.0)
        self.assertAlmostEqual(evaluation.objective, 0.09)
        self.assertAlmostEqual(evaluation.residual.target_term, 0.09)
        self.assertAlmostEqual(evaluation.residual.reference_term, 0.09)
        matrix, gradient = evaluation.residual.inner_linear_model()
        self.assertAlmostEqual(matrix, 1.0)
        self.assertAlmostEqual(gradient, -0.3)
        self.assertAlmostEqual(
            evaluation.residual.frozen_linearized_loss(0.3),
            0.0,
        )

    def test_safeguarded_optimizer_finds_form_b_minimum(self):
        result = optimize_form_b_scalar(
            self.make_model(),
            start_q=-0.8,
            bounds=(-1.0, 1.0),
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.final.scalar_q, 0.3, places=7)
        self.assertLess(result.final.objective, 1.0e-14)
        self.assertTrue(any(record["accepted"] for record in result.history))

    def test_model_requires_double_space(self):
        tda = QuadraticTensor(
            np.eye(2),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
        )
        projected = QuadraticTensor(
            np.eye(3),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
        )
        with self.assertRaises(ValueError):
            StaticXTCTDAModel(tda, projected, number_singles=2)


if __name__ == "__main__":
    unittest.main()
