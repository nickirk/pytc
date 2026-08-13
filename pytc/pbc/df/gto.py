"""JAX-native periodic spherical AO evaluation.

The evaluator mirrors PySCF's real-space lattice sum for ``GTOval_sph`` while
keeping the image loop on the JAX device.  It deliberately supports the
contracted s/p/d/f basis functions used by the periodic ISDF production
fixtures and fails closed for Cartesian or higher-angular-momentum bases.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from pyscf import lib
from pyscf.pbc.gto.eval_gto import _estimate_rcut
from pyscf.pbc.gto.eval_gto import get_lattice_Ls

from pytc.ansatz.gto_spherical import MolGTO_Spherical, eval_ao_spherical

_IMAGE_BLOCK_SIZE = 8


@struct.dataclass
class PeriodicGTOEvaluator:
    """Prepared JAX periodic AO evaluator for one cell and k-point list.

    Array-valued fields are PyTree leaves so one prepared object can be reused
    by eager and jitted calls.  Shape/provenance fields are static metadata.
    """

    molecular: MolGTO_Spherical
    lattice_vectors: jax.Array
    phases: jax.Array
    n_lattice_value: jax.Array
    lmax_value: jax.Array
    rcut_max_value: jax.Array

    @property
    def nao(self):
        return self.molecular.nao

    @property
    def nkpts(self):
        return self.phases.shape[2]

    @property
    def n_lattice(self):
        return int(np.asarray(self.n_lattice_value))

    @property
    def image_block_size(self):
        return self.lattice_vectors.shape[1]

    @property
    def lmax(self):
        return int(np.asarray(self.lmax_value))

    @property
    def rcut_max(self):
        return float(np.asarray(self.rcut_max_value))

    @classmethod
    def from_cell(cls, cell, kpts):
        """Prepare the exact PySCF-convention lattice inventory.

        PySCF estimates a cutoff per shell, then enumerates every lattice
        vector within the largest cutoff and uses ``exp(+i L.k)`` phases.  We
        use the same stable norm ordering.  Evaluating all shells on that
        inventory retains exponentially small tails that PySCF may screen;
        the parity tests gate their numerical insignificance explicitly.
        """
        if not jax.config.read("jax_enable_x64"):
            raise ValueError(
                "PeriodicGTOEvaluator requires jax_enable_x64=True; otherwise "
                "JAX silently downcasts periodic AOs to complex64 instead of "
                "the required complex128."
            )
        if bool(getattr(cell, "cart", False)):
            raise NotImplementedError(
                "PeriodicGTOEvaluator supports spherical AOs only; Cartesian "
                "basis functions are not implemented."
            )
        angular = [int(cell.bas_angular(shell)) for shell in range(cell.nbas)]
        lmax = max(angular, default=0)
        if lmax > 3:
            raise NotImplementedError(
                "PeriodicGTOEvaluator supports angular momentum through f "
                f"(l <= 3), got lmax={lmax}."
            )

        kpts_np = np.asarray(kpts, dtype=np.float64).reshape(-1, 3)
        rcut = np.asarray(_estimate_rcut(cell, 0), dtype=np.float64)
        lattice_vectors = get_lattice_Ls(cell, rcut=float(rcut.max()))
        order = np.argsort(lib.norm(lattice_vectors, axis=1), kind="stable")
        lattice_vectors = np.asarray(lattice_vectors[order], dtype=np.float64)
        phases = np.exp(1j * lattice_vectors @ kpts_np.T).astype(np.complex128)
        n_lattice = lattice_vectors.shape[0]
        padding = (-n_lattice) % _IMAGE_BLOCK_SIZE
        lattice_vectors = np.pad(lattice_vectors, ((0, padding), (0, 0)))
        phases = np.pad(phases, ((0, padding), (0, 0)))
        lattice_vectors = lattice_vectors.reshape(-1, _IMAGE_BLOCK_SIZE, 3)
        phases = phases.reshape(-1, _IMAGE_BLOCK_SIZE, kpts_np.shape[0])

        return cls(
            molecular=MolGTO_Spherical.create(cell),
            lattice_vectors=jnp.asarray(lattice_vectors, dtype=jnp.float64),
            phases=jnp.asarray(phases, dtype=jnp.complex128),
            n_lattice_value=jnp.asarray(n_lattice, dtype=jnp.int32),
            lmax_value=jnp.asarray(lmax, dtype=jnp.int32),
            rcut_max_value=jnp.asarray(rcut.max(), dtype=jnp.float64),
        )

    def provenance(self):
        return {
            "backend": "jax_periodic_spherical_gto",
            "phase_convention": "exp(+i L.k)",
            "n_lattice": self.n_lattice,
            "image_block_size": self.image_block_size,
            "lmax": self.lmax,
            "rcut_max": self.rcut_max,
        }


def _eval_periodic_ao(evaluator: PeriodicGTOEvaluator, coords: jax.Array):
    """Evaluate ``(Nk, Ngrid, Nao)`` periodic AOs on the JAX device.

    The lattice sum is a ``lax.scan`` so residency is bounded by one molecular
    AO image plus the accumulated output, rather than ``Nimage`` AO blocks.
    """
    coords = jnp.asarray(coords, dtype=jnp.float64)
    initial = jnp.zeros(
        (evaluator.nkpts, coords.shape[0], evaluator.nao),
        dtype=jnp.complex128,
    )

    def add_image_block(total, image_block):
        lattice_vectors, phases = image_block
        shifted = (
            coords[None, :, :] - lattice_vectors[:, None, :]
        ).reshape(-1, 3)
        molecular_ao = eval_ao_spherical(
            evaluator.molecular, shifted
        ).reshape(
            evaluator.image_block_size, coords.shape[0], evaluator.nao
        )
        total = total + jnp.einsum(
            "lk,lga->kga", phases, molecular_ao, optimize=True
        )
        return total, None

    result, _ = jax.lax.scan(
        add_image_block, initial, (evaluator.lattice_vectors, evaluator.phases)
    )
    return result


eval_periodic_ao = jax.jit(_eval_periodic_ao)
