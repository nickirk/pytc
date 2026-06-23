from pytc.jastrow import Jastrow
from functools import partial
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from dataclasses import dataclass
import jax
from flax import struct
from typing import List, Any


@dataclass
class DTNTermEE:
    """Pure electron-electron term: c * r12^o * C_ee(r12)."""
    o: int
    c: float


@dataclass
class DTNTermEN:
    """Pure electron-nucleus term (one-body): c * r_eI^k * C_en(r_eI).

    CASINO TERM 2, Rank [1,1].  This is a one-body function of a single
    electron-nucleus distance.  When embedded in the pair-sum framework
    the code symmetrises as  c * [r1I^k * C(r1I) + r2I^k * C(r2I)] / (N-1)
    so that the pair sum reproduces the correct one-body total.
    """
    k: int
    c: float


@dataclass
class DTNTermEEN:
    """Mixed electron-electron-nucleus term.

    c * r12^n * (r1I^l*r2I^m + r1I^m*r2I^l) * C_en(r1I) * C_en(r2I)
    All three cutoffs per DTN paper Eq. 21.
    l >= m stored; l != m pairs symmetrized in _compute.
    """
    n: int  # power of r12 (electron-electron distance)
    l: int  # power of r1I (electron 1 to nucleus)
    m: int  # power of r2I (electron 2 to nucleus)
    c: float


def gen_ee_terms(max_ee_degree: int = 9, init_val: float = 1e-5):
    """Generate EE terms up to a given max degree.
    
    The o=1 term handles the e-e cusp and is fixed to 0.5.
    Generates terms up to max_ee_degree (inclusive).
    """
    terms = [DTNTermEE(1, 0.5)]
    for o in range(2, max_ee_degree + 1):
        terms.append(DTNTermEE(o, init_val))
    return terms


def gen_en_terms(max_en_degree: int = 9, init_val: float = 1e-5):
    """Generate EN terms up to a given max degree.
    
    Starts from k=2 to preserve the exact e-n cusp from SCF orbitals.
    Generates terms up to max_en_degree (inclusive).
    """
    return [DTNTermEN(k, init_val) for k in range(2, max_en_degree + 1)]


def gen_een_terms(max_een_degree: int, init_val: float = 1e-5):
    """Generate EEN terms systematically up to a given max degree.

    Generates all ``DTNTermEEN(n, l, m)`` satisfying:
      - ``n >= 2``, ``l >= m >= 2`` (to strictly preserve unconstrained e-e and e-n cusps)
      - ``max(n, l, m) <= max_een_degree``

    Args:
        max_een_degree: Maximum value allowed for any individual power index
            ``n``, ``l``, or ``m``.
        init_val: Initial coefficient value for all generated terms.

    Returns:
        List of :class:`DTNTermEEN` instances, ordered by ``(n, l, m)``.
    """
    terms = []
    for n in range(2, max_een_degree + 1):
        for l in range(2, max_een_degree + 1):
            for m in range(2, l + 1):
                terms.append(DTNTermEEN(n, l, m, init_val))
    return terms


@struct.dataclass
class DTN(Jastrow):
    """Drummond-Towler-Needs Jastrow factor.

    Separates correlation into three independent channels:

      u = u_ee(r12) + Σ_I u_en(r1I, r2I) + Σ_I u_een(r1I, r2I, r12)

    1. **e-e terms**: Pure electron-electron polynomials in r12 with
       cutoff C_ee(r12).  This is evaluated ONCE globally, independent
       of the nuclei. The cusp term lives here and gives
       du/dr12|_{r12→0} = 0.5 exactly.

    2. **e-n terms**: Pure electron-nucleus one-body polynomials in r_eI
       with cutoff C_en(r_eI). Evaluated per atom type.  Since this is a
       one-body term embedded in a pair sum, each pair contributes
       [χ(r1I) + χ(r2I)] / (N-1) so the pair sum recovers the correct
       one-body total Σ_I Σ_i χ_I(r_iI).

    3. **e-e-n terms**: Mixed three-body polynomials with all three
       cutoffs C_en(r1I) * C_en(r2I) * C_ee(r12). Evaluated per atom type.

    Reference:
        Drummond, Towler, Needs, Phys. Rev. B 70, 235119 (2004).
    """
    nuclear_pos: jax.Array
    nuclear_charges: jax.Array
    atom_type_map: jax.Array
    unique_charges: jax.Array

    # EE term arrays (global)
    _ee_term_o: jax.Array
    _ee_cusp_mask: jax.Array

    # EN term arrays (per atom type)
    _en_term_k: jax.Array

    # EEN term arrays (per atom type) - CASINO c_{n,l,m} notation
    _een_term_n: jax.Array    # r12 power (n in c_{n,l,m})
    _een_term_l: jax.Array    # r1I power (l in c_{n,l,m})
    _een_term_m: jax.Array    # r2I power (m in c_{n,l,m})
    _een_delta_factor: jax.Array  # symmetry factor (1.0 for non-symmetrized)

    nelectron: int = struct.field(pytree_node=False)
    natom: int = struct.field(pytree_node=False)
    n_types: int = struct.field(pytree_node=False)
    n_ee_terms: int = struct.field(pytree_node=False)
    n_en_terms: int = struct.field(pytree_node=False)
    n_een_terms: int = struct.field(pytree_node=False)
    max_degree: int = struct.field(pytree_node=False)
    epsilon: float = struct.field(pytree_node=False, default=1e-8)
    ee_terms: List[DTNTermEE] = struct.field(pytree_node=False, default=None)
    en_terms_per_type: List[List[DTNTermEN]] = struct.field(pytree_node=False, default=None)
    een_terms_per_type: List[List[DTNTermEEN]] = struct.field(pytree_node=False, default=None)
    nuclei_by_type: List[jax.Array] = struct.field(default=None)
    name: str = struct.field(pytree_node=False, default=None)
    max_rc_en: float = struct.field(pytree_node=False, default=8.0)
    max_rc_ee: float = struct.field(pytree_node=False, default=8.0)

    @classmethod
    def create(cls, mol, ee_terms=None, en_terms=None, een_terms=None,
               max_een_degree=None,
               epsilon=1e-8, name=None, max_rc_en=8.0, max_rc_ee=8.0):
        nelectron = mol.nelectron
        nuclear_pos = jnp.array(mol.atom_coords())
        nuclear_charges = jnp.array(mol.atom_charges())
        natom = len(nuclear_charges)

        unique_charges = jnp.sort(jnp.unique(nuclear_charges))
        n_types = len(unique_charges)

        atom_type_map = cls._build_atom_type_map(nuclear_charges, unique_charges, natom)

        ee_terms, en_terms, een_terms, n_ee, n_en, n_een = cls._resolve_terms(
            ee_terms, en_terms, een_terms, max_een_degree, n_types)

        term_arrays, max_degree = cls._build_term_arrays(
            ee_terms, en_terms, een_terms, n_types, n_en, n_een)

        nuclei_by_type = cls._build_nuclei_by_type(nuclear_pos, atom_type_map, n_types)

        return cls(
            nuclear_pos=nuclear_pos,
            nuclear_charges=nuclear_charges,
            atom_type_map=atom_type_map,
            unique_charges=unique_charges,
            _een_term_n=term_arrays['_een_term_n'],
            _een_term_l=term_arrays['_een_term_l'],
            _een_term_m=term_arrays['_een_term_m'],
            _een_delta_factor=term_arrays['_een_delta_factor'],
            _ee_term_o=term_arrays['_ee_term_o'],
            _ee_cusp_mask=term_arrays['_ee_cusp_mask'],
            _en_term_k=term_arrays['_en_term_k'],
            nelectron=nelectron,
            natom=natom,
            n_types=n_types,
            n_ee_terms=n_ee,
            n_en_terms=n_en,
            n_een_terms=n_een,
            max_degree=max_degree,
            epsilon=epsilon,
            ee_terms=ee_terms,
            en_terms_per_type=en_terms,
            een_terms_per_type=een_terms,
            nuclei_by_type=nuclei_by_type,
            name=name,
            max_rc_en=max_rc_en,
            max_rc_ee=max_rc_ee,
        )

    @staticmethod
    def _build_atom_type_map(nuclear_charges, unique_charges, natom):
        atom_type_map = jnp.zeros(natom, dtype=jnp.int32)
        for i, charge in enumerate(nuclear_charges):
            type_idx = jnp.where(unique_charges == charge)[0][0].astype(jnp.int32)
            atom_type_map = atom_type_map.at[i].set(type_idx)
        return atom_type_map

    @staticmethod
    def _check_uniform_terms(terms_list, label):
        if not isinstance(terms_list, list) or (len(terms_list) > 0 and not isinstance(terms_list[0], list)):
            raise TypeError(f"{label} terms must be a list of lists (one list per atom type).")
        lengths = tuple(len(t) for t in terms_list)
        if len(lengths) > 0 and len(set(lengths)) != 1:
            raise ValueError(
                f"DTN requires the same number of {label} terms per atom type."
            )
        return lengths[0] if lengths else 0

    @staticmethod
    def _resolve_terms(ee_terms, en_terms, een_terms, max_een_degree, n_types):
        any_supplied = (ee_terms is not None) or (en_terms is not None) or (een_terms is not None)

        if not any_supplied:
            ee_terms = gen_ee_terms()
            en_terms = [gen_en_terms() for _ in range(n_types)]

            if max_een_degree is not None:
                een_terms = [
                    gen_een_terms(max_een_degree)
                    for _ in range(n_types)
                ]
            else:
                een_terms_clean = []
                for n, l, m in [
                    (2, 4, 2),
                    (3, 2, 2), (3, 3, 2), (3, 3, 3), (3, 4, 2), (3, 4, 3), (3, 4, 4),
                    (4, 2, 2), (4, 3, 2), (4, 3, 3), (4, 4, 2), (4, 4, 3), (4, 4, 4),
                ]:
                    een_terms_clean.append(DTNTermEEN(n, l, m, 1e-5))
                een_terms = [een_terms_clean for _ in range(n_types)]
        else:
            if ee_terms is None:
                ee_terms = []
            if en_terms is None:
                en_terms = [[] for _ in range(n_types)]
            if een_terms is None:
                if max_een_degree is not None:
                    een_terms = [gen_een_terms(max_een_degree) for _ in range(n_types)]
                else:
                    een_terms = [[] for _ in range(n_types)]

        if en_terms and not isinstance(en_terms[0], list):
            en_terms = [en_terms for _ in range(n_types)]
        if een_terms and not isinstance(een_terms[0], list):
            een_terms = [een_terms for _ in range(n_types)]

        n_ee = len(ee_terms)
        n_en = DTN._check_uniform_terms(en_terms, "EN")
        n_een = DTN._check_uniform_terms(een_terms, "EEN")

        return ee_terms, en_terms, een_terms, n_ee, n_en, n_een

    @staticmethod
    def _build_term_arrays(ee_terms, en_terms, een_terms, n_types, n_en, n_een):
        max_degree = 0

        if ee_terms:
            max_degree = max(max_degree, max(t.o for t in ee_terms))
            _ee_term_o = jnp.array([t.o for t in ee_terms], dtype=jnp.int32)
            _ee_cusp_mask = (_ee_term_o == 1)
        else:
            _ee_term_o = jnp.zeros(0, dtype=jnp.int32)
            _ee_cusp_mask = jnp.zeros(0, dtype=bool)

        en_k = []
        for type_terms in en_terms:
            en_k.append([t.k for t in type_terms])
            if type_terms:
                max_degree = max(max_degree, max(t.k for t in type_terms))
        _en_term_k = jnp.array(en_k, dtype=jnp.int32) if n_en > 0 else jnp.zeros((n_types, 0), dtype=jnp.int32)

        een_n_array, een_l_array, een_m_array = [], [], []
        for type_terms in een_terms:
            een_n_array.append([t.n for t in type_terms])
            een_l_array.append([t.l for t in type_terms])
            een_m_array.append([t.m for t in type_terms])
            if type_terms:
                cur = max(max(t.n for t in type_terms),
                          max(t.l for t in type_terms),
                          max(t.m for t in type_terms))
                max_degree = max(max_degree, cur)

        if n_een > 0:
            _een_term_n = jnp.array(een_n_array, dtype=jnp.int32)
            _een_term_l = jnp.array(een_l_array, dtype=jnp.int32)
            _een_term_m = jnp.array(een_m_array, dtype=jnp.int32)
            _een_delta_factor = jnp.ones((n_types, n_een))
        else:
            _een_term_n = jnp.zeros((n_types, 0), dtype=jnp.int32)
            _een_term_l = jnp.zeros((n_types, 0), dtype=jnp.int32)
            _een_term_m = jnp.zeros((n_types, 0), dtype=jnp.int32)
            _een_delta_factor = jnp.zeros((n_types, 0))

        return {
            '_ee_term_o': _ee_term_o,
            '_ee_cusp_mask': _ee_cusp_mask,
            '_en_term_k': _en_term_k,
            '_een_term_n': _een_term_n,
            '_een_term_l': _een_term_l,
            '_een_term_m': _een_term_m,
            '_een_delta_factor': _een_delta_factor,
        }, max_degree

    @staticmethod
    def _build_nuclei_by_type(nuclear_pos, atom_type_map, n_types):
        nuclei_by_type = []
        for i in range(n_types):
            mask_np = jnp.array(atom_type_map) == i
            nuclei_group = nuclear_pos[jnp.array(mask_np)]
            nuclei_by_type.append(nuclei_group)
        return nuclei_by_type

    def _safe_norm(self, x):
        return jnp.sqrt(jnp.sum(x * x, axis=-1) + self.epsilon)

    def _cutoff_envelope(self, r, rc):
        """Smooth cutoff C(r) = (1 - r/rc)^3 for r < rc, else 0."""
        x = r / rc
        return jnp.where(r < rc, (1.0 - x) ** 3, 0.0)

    def _raw_distance(self, r_a, r_b):
        """Euclidean distance between two position vectors."""
        return self._safe_norm(r_a - r_b)

    def _get_powers(self, x, degree):
        """Build power table [x^0, x^1, ..., x^degree]."""
        exponents = jnp.arange(degree + 1)
        safe_x = jnp.where(x == 0.0, 1.0, x)
        powers = jnp.power(safe_x[..., None], exponents)
        mask = (x == 0.0)[..., None] & (exponents > 0)
        return jnp.where(mask, 0.0, powers)

    def init_params(self, **kwargs):
        rc_en_raw = jnp.zeros(self.n_types)
        rc_ee_raw = jnp.zeros(1)   # Global

        c_ee_raw = jnp.array(
            [t.c for t in self.ee_terms]
        ) if self.n_ee_terms > 0 else jnp.zeros(0)

        c_en_raw = jnp.array([
            [t.c for t in type_terms]
            for type_terms in self.en_terms_per_type
        ]) if self.n_en_terms > 0 else jnp.zeros((self.n_types, 0))

        c_een_raw = jnp.array([
            [t.c for t in type_terms]
            for type_terms in self.een_terms_per_type
        ]) if self.n_een_terms > 0 else jnp.zeros((self.n_types, 0))

        return {
            'rc_en_raw': rc_en_raw,
            'rc_ee_raw': rc_ee_raw,
            'c_ee_raw': c_ee_raw,
            'c_en_raw': c_en_raw,
            'c_een_raw': c_een_raw,
        }

    def _compute_forward(self, r1, r2, params):
        rc_en = self.max_rc_en * nn.sigmoid(params['rc_en_raw'])
        rc_ee = self.max_rc_ee * nn.sigmoid(params['rc_ee_raw'][0])

        c_ee = jnp.where(self._ee_cusp_mask, 0.5, jnp.array(params['c_ee_raw']))
        c_en = jnp.array(params['c_en_raw'])
        c_een = jnp.array(params['c_een_raw'])

        # --- GLOBAL EVALUATIONS ---
        r12 = self._raw_distance(r1, r2)
        C_r12 = self._cutoff_envelope(r12, rc_ee)
        p_r12 = self._get_powers(r12, self.max_degree)

        # 1. EE term: Σ c_k * r12^o * C_ee(r12)
        if self.n_ee_terms > 0:
            ee_vals = p_r12[self._ee_term_o] * C_r12
            ee_total = jnp.sum(c_ee * ee_vals)
        else:
            ee_total = 0.0

        # --- PER-NUCLEUS EVALUATIONS ---
        def compute_atom(atom_idx):
            type_idx = self.atom_type_map[atom_idx]
            nuc_pos = self.nuclear_pos[atom_idx]

            rc_en_I = rc_en[type_idx]

            r1I = self._raw_distance(r1, nuc_pos)
            r2I = self._raw_distance(r2, nuc_pos)

            C_r1I = self._cutoff_envelope(r1I, rc_en_I)
            C_r2I = self._cutoff_envelope(r2I, rc_en_I)

            p_r1I = self._get_powers(r1I, self.max_degree)
            p_r2I = self._get_powers(r2I, self.max_degree)

            # 2. EN: Σ c_k * [r1I^k * C_en(r1I) + r2I^k * C_en(r2I)] / (N-1)
            # CASINO TERM 2, Rank [1,1]: one-body term.  Each electron's
            # contribution uses only its own cutoff envelope.  The factor
            # 1/(N-1) compensates for each electron appearing in (N-1) pairs.
            en_k_idx = self._en_term_k[type_idx]
            en_vals = (p_r1I[en_k_idx] * C_r1I + p_r2I[en_k_idx] * C_r2I)
            en_total = jnp.sum(c_en[type_idx, :] * en_vals) / (self.nelectron - 1)

            # 3. EEN: Σ c_{n,l,m} * r12^n * (r1I^l*r2I^m + r1I^m*r2I^l)
            #         * C_en(r1I) * C_en(r2I) * C_ee(r12)
            # DTN paper Eq. 21: all three cutoffs (2 EN + 1 EE).
            # Parameters stored with l >= m; l != m pairs symmetrized explicitly.
            n_idx = self._een_term_n[type_idx]    # r12 power
            l_idx = self._een_term_l[type_idx]    # r1I power (l >= m)
            m_idx = self._een_term_m[type_idx]    # r2I power

            v_r12_n = p_r12[n_idx]    # r12^n
            v_r1I_l = p_r1I[l_idx]    # r1I^l
            v_r2I_m = p_r2I[m_idx]    # r2I^m
            v_r1I_m = p_r1I[m_idx]    # r1I^m (counterpart for symmetrization)
            v_r2I_l = p_r2I[l_idx]    # r2I^l (counterpart for symmetrization)

            l_eq_m = (l_idx == m_idx)
            een_poly = jnp.where(
                l_eq_m,
                v_r12_n * v_r1I_l * v_r2I_m,                         # l == m
                v_r12_n * (v_r1I_l * v_r2I_m + v_r1I_m * v_r2I_l),  # l != m: symmetrized
            )
            een_envelope = C_r1I * C_r2I   # CASINO convention: only EN cutoffs (no C_r12)
            een_total = jnp.sum(c_een[type_idx, :] * een_poly * een_envelope)

            return en_total + een_total

        atom_contributions = jax.vmap(compute_atom)(jnp.arange(self.natom))
        return ee_total + jnp.sum(atom_contributions)

    def _compute(self, r1, r2, params):
        return self._compute_forward(r1, r2, params)

    def flatten_params(self, params):
        return jnp.concatenate([
            params['rc_en_raw'].ravel(),
            params['rc_ee_raw'].ravel(),
            params['c_ee_raw'].ravel(),
            params['c_en_raw'].ravel(),
            params['c_een_raw'].ravel(),
        ])

    def unflatten_params(self, flat_params):
        idx = 0

        rc_en_raw = flat_params[idx:idx + self.n_types]
        idx += self.n_types

        rc_ee_raw = flat_params[idx:idx + 1]
        idx += 1

        c_ee_raw = flat_params[idx:idx + self.n_ee_terms]
        idx += self.n_ee_terms

        en_size = self.n_types * self.n_en_terms
        c_en_raw = flat_params[idx:idx + en_size].reshape(self.n_types, self.n_en_terms)
        idx += en_size

        een_size = self.n_types * self.n_een_terms
        c_een_raw = flat_params[idx:idx + een_size].reshape(self.n_types, self.n_een_terms)
        idx += een_size

        return {
            'rc_en_raw': rc_en_raw,
            'rc_ee_raw': rc_ee_raw,
            'c_ee_raw': c_ee_raw,
            'c_en_raw': c_en_raw,
            'c_een_raw': c_een_raw,
        }
