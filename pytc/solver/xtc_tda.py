"""Static polynomial xTC-TDA utilities.

The routines in this module preserve the non-Hermitian orientation of the
transcorrelated Hamiltonian.  They provide a compact workflow for quadratic
coefficient interpolation, frozen-orbital singlet TDA, biorthogonal root
tracking, selected-space projection, and the balanced left/right residual
objective used to optimize one scalar Jastrow-mode coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import scipy.linalg
from pyscf.fci import cistring, direct_nosym
from scipy.optimize import linear_sum_assignment, minimize_scalar


Array: TypeAlias = np.ndarray


@dataclass(frozen=True)
class QuadraticTensor:
    """Tensor-valued ``X(q) = X0 + q X1 + 0.5 q**2 X2``."""

    constant: Array
    linear: Array
    quadratic: Array

    def __post_init__(self) -> None:
        constant = np.asarray(self.constant)
        linear = np.asarray(self.linear)
        quadratic = np.asarray(self.quadratic)
        if constant.shape != linear.shape or constant.shape != quadratic.shape:
            raise ValueError("quadratic tensor components must have one shape")
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadratic", quadratic)

    @classmethod
    def from_symmetric_stencil(
        cls,
        minus: Array,
        zero: Array,
        plus: Array,
        *,
        step: float = 1.0,
    ) -> "QuadraticTensor":
        """Recover exact quadratic components from ``q=-h, 0, +h``."""

        if step <= 0.0:
            raise ValueError("step must be positive")
        minus = np.asarray(minus)
        zero = np.asarray(zero)
        plus = np.asarray(plus)
        if minus.shape != zero.shape or zero.shape != plus.shape:
            raise ValueError("stencil tensors must have one shape")
        return cls(
            constant=zero,
            linear=(plus - minus) / (2.0 * step),
            quadratic=(plus - 2.0 * zero + minus) / step**2,
        )

    def value(self, scalar_q: float) -> Array:
        """Evaluate the polynomial at ``scalar_q``."""

        return (
            self.constant
            + scalar_q * self.linear
            + 0.5 * scalar_q**2 * self.quadratic
        )

    def derivative(self, scalar_q: float) -> Array:
        """Evaluate ``dX/dq`` at ``scalar_q``."""

        return self.linear + scalar_q * self.quadratic


def build_rhf_fock(h1: Array, eri: Array, nocc: int) -> Array:
    """Build a frozen-density RHF Fock matrix in chemists' notation."""

    h1 = np.asarray(h1)
    eri = np.asarray(eri)
    norb = h1.shape[0]
    if h1.shape != (norb, norb):
        raise ValueError("h1 must be square")
    if eri.shape != (norb, norb, norb, norb):
        raise ValueError("eri has an incompatible shape")
    if nocc <= 0 or nocc >= norb:
        raise ValueError("nocc must leave occupied and virtual orbitals")
    fock = h1.copy()
    fock += 2.0 * np.einsum("pqii->pq", eri[:, :, :nocc, :nocc])
    fock -= np.einsum("piiq->pq", eri[:, :nocc, :nocc, :])
    return fock


def build_singlet_tda(h1: Array, eri: Array, nocc: int) -> Array:
    """Build the frozen-orbital RHF singlet TDA matrix.

    The two-electron tensor follows the PySCF ``direct_nosym`` convention
    ``V[p,q,r,s] a_p^+ a_r^+ a_s a_q``.  No Hermitian symmetry is assumed.
    """

    h1 = np.asarray(h1)
    eri = np.asarray(eri)
    norb = h1.shape[0]
    fock = build_rhf_fock(h1, eri, nocc)
    nvir = norb - nocc
    matrix = np.empty(
        (nocc, nvir, nocc, nvir),
        dtype=np.result_type(h1, eri),
    )
    for i in range(nocc):
        for a_local, a in enumerate(range(nocc, norb)):
            for j in range(nocc):
                for b_local, b in enumerate(range(nocc, norb)):
                    value = 2.0 * eri[a, i, j, b] - eri[a, b, j, i]
                    if i == j:
                        value += fock[a, b]
                    if a == b:
                        value -= fock[j, i]
                    matrix[i, a_local, j, b_local] = value
    return matrix.reshape(nocc * nvir, nocc * nvir)


def build_reference_singles_couplings(
    h1: Array,
    eri: Array,
    nocc: int,
) -> tuple[Array, Array]:
    """Return ``(<0|H|S>, <S|H|0>)`` without equating the two sides."""

    fock = build_rhf_fock(h1, eri, nocc)
    left = np.sqrt(2.0) * fock[:nocc, nocc:].reshape(-1)
    right = np.sqrt(2.0) * fock[nocc:, :nocc].T.reshape(-1)
    return left, right


def build_tda_polynomial(
    h1: QuadraticTensor,
    eri: QuadraticTensor,
    nocc: int,
) -> QuadraticTensor:
    """Transform polynomial xTC integrals into polynomial TDA matrices."""

    return QuadraticTensor(
        constant=build_singlet_tda(h1.constant, eri.constant, nocc),
        linear=build_singlet_tda(h1.linear, eri.linear, nocc),
        quadratic=build_singlet_tda(h1.quadratic, eri.quadratic, nocc),
    )


@dataclass(frozen=True)
class BiorthogonalEigensystem:
    """Energy-sorted, biorthonormal left and right eigenvectors."""

    eigenvalues: Array
    left: Array
    right: Array
    right_residual: Array
    left_residual: Array
    biorthogonality_error: float
    condition_numbers: Array


def solve_biorthogonal(
    matrix: Array,
    *,
    maximum_basis_condition: float = 1.0e12,
) -> BiorthogonalEigensystem:
    """Diagonalize a non-Hermitian matrix and biorthonormalize its roots."""

    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    eigenvalues, right = scipy.linalg.eig(matrix)
    order = np.lexsort((eigenvalues.imag, eigenvalues.real))
    eigenvalues = eigenvalues[order]
    right = right[:, order]
    norms = np.linalg.norm(right, axis=0)
    if np.any(norms < 1.0e-14):
        raise RuntimeError("right eigenvector has near-zero norm")
    right = right / norms
    basis_condition = float(np.linalg.cond(right))
    if (
        not np.isfinite(basis_condition)
        or basis_condition > maximum_basis_condition
    ):
        raise RuntimeError(
            f"ill-conditioned right eigenvector basis: {basis_condition}"
        )
    left = np.linalg.inv(right).conj().T
    right_residual = np.asarray(
        [
            np.linalg.norm(
                matrix @ right[:, root]
                - eigenvalues[root] * right[:, root]
            )
            for root in range(right.shape[1])
        ]
    )
    left_residual = np.asarray(
        [
            np.linalg.norm(
                left[:, root].conj().T @ matrix
                - eigenvalues[root] * left[:, root].conj().T
            )
            for root in range(left.shape[1])
        ]
    )
    biorthogonality_error = float(
        np.linalg.norm(left.conj().T @ right - np.eye(matrix.shape[0]))
    )
    condition_numbers = np.linalg.norm(left, axis=0) * np.linalg.norm(
        right,
        axis=0,
    )
    return BiorthogonalEigensystem(
        eigenvalues=eigenvalues,
        left=left,
        right=right,
        right_residual=right_residual,
        left_residual=left_residual,
        biorthogonality_error=biorthogonality_error,
        condition_numbers=condition_numbers,
    )


def normalized_right_overlap(reference: Array, candidate: Array) -> Array:
    """Return the bounded Euclidean right-vector overlap matrix."""

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    reference = reference / np.linalg.norm(reference, axis=0)
    candidate = candidate / np.linalg.norm(candidate, axis=0)
    overlap = np.abs(reference.conj().T @ candidate)
    if np.max(overlap) > 1.0 + 1.0e-12:
        raise AssertionError("normalized right-vector overlap exceeds one")
    return np.clip(overlap, 0.0, 1.0)


def excitation_derivative(
    eigensystem: BiorthogonalEigensystem,
    matrix_derivative: Array,
) -> Array:
    """Evaluate ``L_k^dagger A_,q R_k`` for every TDA root."""

    matrix_derivative = np.asarray(matrix_derivative)
    return np.asarray(
        [
            eigensystem.left[:, root].conj().T
            @ matrix_derivative
            @ eigensystem.right[:, root]
            for root in range(eigensystem.right.shape[1])
        ]
    )


def project_hamiltonian(
    h1: Array,
    eri: Array,
    core: complex,
    basis_vectors: Array,
    norb: int,
    nelec: tuple[int, int],
) -> Array:
    """Project an unsymmetrized Hamiltonian into supplied FCI-space vectors.

    ``basis_vectors`` stores one flattened PySCF FCI vector per column.  The
    bra and ket bases are identical, but the returned matrix is not
    symmetrized.
    """

    h1 = np.asarray(h1)
    eri = np.asarray(eri)
    basis_vectors = np.asarray(basis_vectors)
    if h1.shape != (norb, norb):
        raise ValueError("h1 has an incompatible shape")
    if eri.shape != (norb, norb, norb, norb):
        raise ValueError("eri has an incompatible shape")
    nalpha = cistring.num_strings(norb, nelec[0])
    nbeta = cistring.num_strings(norb, nelec[1])
    if basis_vectors.ndim != 2 or basis_vectors.shape[0] != nalpha * nbeta:
        raise ValueError("basis_vectors has an incompatible FCI dimension")
    gram = basis_vectors.conj().T @ basis_vectors
    if not np.allclose(
        gram,
        np.eye(basis_vectors.shape[1]),
        atol=1.0e-10,
    ):
        raise ValueError("basis_vectors must have orthonormal columns")
    absorbed = direct_nosym.absorb_h1e(
        h1,
        eri,
        norb,
        nelec,
        fac=0.5,
    )
    applied = np.column_stack(
        [
            np.asarray(
                direct_nosym.contract_2e(
                    absorbed,
                    basis_vectors[:, column].reshape(nalpha, nbeta),
                    norb,
                    nelec,
                )
            ).ravel()
            + core * basis_vectors[:, column]
            for column in range(basis_vectors.shape[1])
        ]
    )
    return basis_vectors.conj().T @ applied


def project_hamiltonian_polynomial(
    h1: QuadraticTensor,
    eri: QuadraticTensor,
    core: QuadraticTensor,
    basis_vectors: Array,
    norb: int,
    nelec: tuple[int, int],
) -> QuadraticTensor:
    """Project all three exact polynomial Hamiltonian components."""

    if core.constant.shape != ():
        raise ValueError("core polynomial must be scalar")
    projected = [
        project_hamiltonian(
            h1_component,
            eri_component,
            complex(core_component),
            basis_vectors,
            norb,
            nelec,
        )
        for h1_component, eri_component, core_component in zip(
            (h1.constant, h1.linear, h1.quadratic),
            (eri.constant, eri.linear, eri.quadratic),
            (core.constant, core.linear, core.quadratic),
            strict=True,
        )
    ]
    return QuadraticTensor(*projected)


@dataclass(frozen=True)
class BalancedResidual:
    """Canonical unweighted form-B residual and frozen-vector derivatives."""

    target_right: Array
    target_left: Array
    reference_right: Array
    reference_left: Array
    target_right_derivative: Array
    target_left_derivative: Array
    reference_right_derivative: Array
    reference_left_derivative: Array

    @property
    def target_term(self) -> float:
        return 0.5 * float(
            np.vdot(self.target_right, self.target_right).real
            + np.vdot(self.target_left, self.target_left).real
        )

    @property
    def reference_term(self) -> float:
        return 0.5 * float(
            np.vdot(self.reference_right, self.reference_right).real
            + np.vdot(self.reference_left, self.reference_left).real
        )

    @property
    def total(self) -> float:
        """Canonical form B: equal target/reference and left/right weight."""

        return 0.5 * (self.target_term + self.reference_term)

    def inner_linear_model(self) -> tuple[float, float]:
        """Return scalar ``M`` and ``g`` with left/right vectors frozen."""

        couplings = (
            self.target_right,
            self.target_left,
            self.reference_right,
            self.reference_left,
        )
        derivatives = (
            self.target_right_derivative,
            self.target_left_derivative,
            self.reference_right_derivative,
            self.reference_left_derivative,
        )
        matrix = 0.25 * sum(
            float(np.vdot(derivative, derivative).real)
            for derivative in derivatives
        )
        gradient = 0.25 * sum(
            float(np.vdot(derivative, coupling).real)
            for derivative, coupling in zip(
                derivatives,
                couplings,
                strict=True,
            )
        )
        return matrix, gradient

    def frozen_linearized_loss(self, delta_q: float) -> float:
        """Evaluate the fixed-vector linearized form-B objective."""

        couplings = (
            self.target_right,
            self.target_left,
            self.reference_right,
            self.reference_left,
        )
        derivatives = (
            self.target_right_derivative,
            self.target_left_derivative,
            self.reference_right_derivative,
            self.reference_left_derivative,
        )
        return 0.25 * sum(
            float(
                np.vdot(
                    coupling + delta_q * derivative,
                    coupling + delta_q * derivative,
                ).real
            )
            for derivative, coupling in zip(
                derivatives,
                couplings,
                strict=True,
            )
        )


@dataclass(frozen=True)
class StaticXTCTDAEvaluation:
    """One fully rediagonalized form-B evaluation."""

    scalar_q: float
    eigensystem: BiorthogonalEigensystem
    target_index: int
    target_eigenvalue: complex
    residual: BalancedResidual
    normalized_right_overlap: float
    biorthogonal_overlap: float

    @property
    def objective(self) -> float:
        return self.residual.total


class StaticXTCTDAModel:
    """Quadratic xTC-TDA model with reference/singles/doubles projections.

    The projected basis must be ordered as reference, singlet singles, then
    selected doubles.  The TDA polynomial acts only in the singles space.
    """

    def __init__(
        self,
        tda: QuadraticTensor,
        projected_hamiltonian: QuadraticTensor,
        number_singles: int,
        *,
        target_root: int = 0,
    ) -> None:
        if tda.constant.shape != (number_singles, number_singles):
            raise ValueError("tda dimension must equal number_singles")
        projected_shape = projected_hamiltonian.constant.shape
        if (
            len(projected_shape) != 2
            or projected_shape[0] != projected_shape[1]
            or projected_shape[0] <= 1 + number_singles
        ):
            raise ValueError(
                "projected Hamiltonian must include reference, singles, "
                "and at least one double"
            )
        if target_root < 0 or target_root >= number_singles:
            raise ValueError("target_root is out of range")
        self.tda = tda
        self.projected_hamiltonian = projected_hamiltonian
        self.number_singles = number_singles
        self.target_root = target_root

    def evaluate(
        self,
        scalar_q: float,
        anchor: StaticXTCTDAEvaluation | None = None,
    ) -> StaticXTCTDAEvaluation:
        eigensystem = solve_biorthogonal(self.tda.value(scalar_q))
        if anchor is None:
            target_index = self.target_root
            right_overlap = 1.0
            biorthogonal_overlap = 1.0
        else:
            overlap = np.abs(
                anchor.eigensystem.left.conj().T @ eigensystem.right
            )
            rows, columns = linear_sum_assignment(-overlap)
            assignment = np.empty(len(rows), dtype=np.int64)
            assignment[rows] = columns
            target_index = int(assignment[anchor.target_index])
            previous = anchor.eigensystem.right[:, anchor.target_index]
            current = eigensystem.right[:, target_index]
            right_overlap = float(
                abs(np.vdot(previous, current))
                / (np.linalg.norm(previous) * np.linalg.norm(current))
            )
            right_overlap = min(right_overlap, 1.0)
            biorthogonal_overlap = float(
                overlap[anchor.target_index, target_index]
            )

        matrix = self.projected_hamiltonian.value(scalar_q)
        derivative = self.projected_hamiltonian.derivative(scalar_q)
        singles = slice(1, 1 + self.number_singles)
        doubles = slice(1 + self.number_singles, matrix.shape[0])
        right = eigensystem.right[:, target_index]
        left = eigensystem.left[:, target_index]
        residual = BalancedResidual(
            target_right=matrix[doubles, singles] @ right,
            target_left=left.conj().T @ matrix[singles, doubles],
            reference_right=matrix[doubles, 0],
            reference_left=matrix[0, doubles],
            target_right_derivative=derivative[doubles, singles] @ right,
            target_left_derivative=(
                left.conj().T @ derivative[singles, doubles]
            ),
            reference_right_derivative=derivative[doubles, 0],
            reference_left_derivative=derivative[0, doubles],
        )
        return StaticXTCTDAEvaluation(
            scalar_q=float(scalar_q),
            eigensystem=eigensystem,
            target_index=target_index,
            target_eigenvalue=eigensystem.eigenvalues[target_index],
            residual=residual,
            normalized_right_overlap=right_overlap,
            biorthogonal_overlap=biorthogonal_overlap,
        )


def numerical_gates_pass(
    evaluation: StaticXTCTDAEvaluation,
    *,
    minimum_overlap: float = 0.90,
    maximum_imaginary: float = 1.0e-10,
    maximum_residual: float = 1.0e-10,
    maximum_biorthogonality_error: float = 1.0e-10,
    maximum_condition_number: float = 1.0e6,
) -> bool:
    """Check the tracked root before accepting an optimizer step."""

    root = evaluation.target_index
    return bool(
        evaluation.normalized_right_overlap >= minimum_overlap
        and abs(evaluation.target_eigenvalue.imag) <= maximum_imaginary
        and evaluation.eigensystem.right_residual[root] <= maximum_residual
        and evaluation.eigensystem.left_residual[root] <= maximum_residual
        and evaluation.eigensystem.biorthogonality_error
        <= maximum_biorthogonality_error
        and evaluation.eigensystem.condition_numbers[root]
        <= maximum_condition_number
    )


def actual_objective_derivative(
    model: StaticXTCTDAModel,
    evaluation: StaticXTCTDAEvaluation,
    *,
    step: float = 1.0e-6,
    bounds: tuple[float, float] = (-np.inf, np.inf),
) -> float:
    """Central derivative of the rediagonalized tracked form-B objective."""

    lower, upper = bounds
    available = min(
        step,
        evaluation.scalar_q - lower,
        upper - evaluation.scalar_q,
    )
    if available <= 0.0:
        raise ValueError("central derivative is unavailable at the bound")
    plus = model.evaluate(
        evaluation.scalar_q + available,
        anchor=evaluation,
    )
    minus = model.evaluate(
        evaluation.scalar_q - available,
        anchor=evaluation,
    )
    return (plus.objective - minus.objective) / (2.0 * available)


@dataclass(frozen=True)
class ScalarOptimizationResult:
    """Result of the safeguarded one-mode form-B macroiteration."""

    start_q: float
    converged: bool
    final: StaticXTCTDAEvaluation
    history: tuple[dict[str, float | int | bool | str], ...]


def optimize_form_b_scalar(
    model: StaticXTCTDAModel,
    start_q: float,
    *,
    bounds: tuple[float, float] = (-1.0, 1.0),
    regularization: float = 1.0e-8,
    initial_trust_radius: float = 0.25,
    minimum_trust_radius: float = 1.0e-6,
    maximum_trust_radius: float = 0.50,
    minimum_ratio: float = 0.10,
    maximum_iterations: int = 50,
    q_tolerance: float = 1.0e-8,
    loss_tolerance: float = 1.0e-10,
    derivative_tolerance: float = 1.0e-8,
) -> ScalarOptimizationResult:
    """Optimize canonical form B with frozen-vector steps and a safeguard.

    The deterministic inner proposal is attempted first.  If the
    rediagonalized objective rejects it, a bounded minimization of the actual
    tracked objective is used inside the same trust interval.
    """

    lower, upper = bounds
    if lower >= upper:
        raise ValueError("bounds must be increasing")
    if not lower <= start_q <= upper:
        raise ValueError("start_q is outside bounds")
    if minimum_trust_radius <= 0.0:
        raise ValueError("minimum_trust_radius must be positive")
    current = model.evaluate(start_q)
    trust_radius = initial_trust_radius
    history: list[dict[str, float | int | bool | str]] = []
    converged = False

    for iteration in range(maximum_iterations):
        matrix, gradient = current.residual.inner_linear_model()
        raw_delta = -gradient / (matrix + regularization)
        accepted = False
        while trust_radius >= minimum_trust_radius:
            delta_q = float(np.clip(raw_delta, -trust_radius, trust_radius))
            candidate_q = float(
                np.clip(current.scalar_q + delta_q, lower, upper)
            )
            delta_q = candidate_q - current.scalar_q
            predicted = (
                current.objective
                - current.residual.frozen_linearized_loss(delta_q)
            )
            candidate = model.evaluate(candidate_q, anchor=current)
            actual = current.objective - candidate.objective
            ratio = actual / predicted if predicted > 0.0 else -np.inf
            gates = numerical_gates_pass(candidate)
            primary_record: dict[str, float | int | bool | str] = {
                "iteration": iteration,
                "step_source": "frozen_response_primary",
                "q_before": current.scalar_q,
                "q_candidate": candidate.scalar_q,
                "loss_before": current.objective,
                "loss_candidate": candidate.objective,
                "raw_delta_q": raw_delta,
                "delta_q": delta_q,
                "trust_radius": trust_radius,
                "predicted_reduction": predicted,
                "actual_reduction": actual,
                "reduction_ratio": float(ratio),
                "root_overlap": candidate.normalized_right_overlap,
                "numerical_gates_pass": gates,
            }
            if (
                abs(delta_q) >= q_tolerance
                and actual > 0.0
                and ratio >= minimum_ratio
                and gates
            ):
                primary_record["accepted"] = True
                primary_record["reason"] = "accepted_primary"
                history.append(primary_record)
                previous_loss = current.objective
                current = candidate
                accepted = True
                if ratio > 0.75 and abs(delta_q) >= 0.99 * trust_radius:
                    trust_radius = min(
                        2.0 * trust_radius,
                        maximum_trust_radius,
                    )
                if (
                    abs(delta_q) < q_tolerance
                    and abs(previous_loss - current.objective)
                    < loss_tolerance
                ):
                    converged = True
                break
            primary_record["accepted"] = False
            primary_record["reason"] = "rejected_primary"
            history.append(primary_record)

            fallback_lower = max(lower, current.scalar_q - trust_radius)
            fallback_upper = min(upper, current.scalar_q + trust_radius)
            fallback_result = minimize_scalar(
                lambda scalar_q: model.evaluate(
                    float(scalar_q),
                    anchor=current,
                ).objective,
                bounds=(fallback_lower, fallback_upper),
                method="bounded",
                options={"xatol": 1.0e-12},
            )
            if not fallback_result.success:
                raise RuntimeError("actual-objective fallback failed")
            fallback = model.evaluate(
                float(fallback_result.x),
                anchor=current,
            )
            fallback_delta = fallback.scalar_q - current.scalar_q
            fallback_reduction = current.objective - fallback.objective
            fallback_gates = numerical_gates_pass(fallback)
            fallback_record: dict[str, float | int | bool | str] = {
                "iteration": iteration,
                "step_source": "actual_objective_fallback",
                "q_before": current.scalar_q,
                "q_candidate": fallback.scalar_q,
                "loss_before": current.objective,
                "loss_candidate": fallback.objective,
                "delta_q": fallback_delta,
                "trust_radius": trust_radius,
                "actual_reduction": fallback_reduction,
                "root_overlap": fallback.normalized_right_overlap,
                "numerical_gates_pass": fallback_gates,
            }
            if abs(fallback_delta) < q_tolerance:
                derivative = actual_objective_derivative(
                    model,
                    current,
                    bounds=bounds,
                )
                fallback_record["actual_objective_derivative"] = derivative
                if abs(derivative) <= derivative_tolerance and fallback_gates:
                    fallback_record["accepted"] = True
                    fallback_record["reason"] = "fallback_stationary"
                    history.append(fallback_record)
                    converged = True
                    accepted = True
                    break
            if fallback_reduction > 0.0 and fallback_gates:
                fallback_record["accepted"] = True
                fallback_record["reason"] = "accepted_fallback"
                history.append(fallback_record)
                previous_loss = current.objective
                current = fallback
                accepted = True
                if (
                    abs(fallback_delta) < q_tolerance
                    and abs(previous_loss - current.objective)
                    < loss_tolerance
                ):
                    converged = True
                break
            fallback_record["accepted"] = False
            fallback_record["reason"] = "rejected_fallback"
            history.append(fallback_record)
            trust_radius *= 0.5
        if converged or not accepted:
            break
    return ScalarOptimizationResult(
        start_q=float(start_q),
        converged=converged,
        final=current,
        history=tuple(history),
    )


__all__ = [
    "BalancedResidual",
    "BiorthogonalEigensystem",
    "QuadraticTensor",
    "ScalarOptimizationResult",
    "StaticXTCTDAEvaluation",
    "StaticXTCTDAModel",
    "actual_objective_derivative",
    "build_reference_singles_couplings",
    "build_rhf_fock",
    "build_singlet_tda",
    "build_tda_polynomial",
    "excitation_derivative",
    "normalized_right_overlap",
    "numerical_gates_pass",
    "optimize_form_b_scalar",
    "project_hamiltonian",
    "project_hamiltonian_polynomial",
    "solve_biorthogonal",
]
