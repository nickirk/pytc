import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from .jastrow import Jastrow
from functools import partial
import jax

class NuclearCusp(Jastrow):
    def __init__(self, mol, name=None, n_radial=1000):
        """Initialize nuclear cusp correction.
        
        Args:
            n_radial: Number of radial grid points for orbital evaluation
        """
        super().__init__(name=name)
        self.n_radial = n_radial
        self.nelectron = mol.nelectron
        self.setup_for_molecule(mol)
        
        
    def setup_for_molecule(self, mol, mo_coeff=None):
        """Setup orbital evaluators for given molecule.
        This is separate from __init__ to avoid storing large arrays.
        
        Args:
            mol: PySCF Mole object
            mo_coeff: Molecular orbital coefficients
        """
        # Convert coordinates and charges to JAX arrays
        self.coords = jnp.array(mol.atom_coords())
        self.charges = jnp.array(mol.atom_charges())
        self.n_nuclei = len(self.charges)
        
        # Convert atomic charges to indices using array ops
        unique_Z = jnp.sort(jnp.unique(self.charges))
        self.unique_Z = unique_Z
        # Create reverse mapping array: Z -> idx
        max_Z = int(jnp.max(unique_Z))
        Z_to_idx = -jnp.ones(max_Z + 1, dtype=jnp.int32)
        # Use dynamic_update_slice or scatter to set values
        for i, Z in enumerate(unique_Z):
            Z_to_idx = Z_to_idx.at[int(Z)].set(i)
        self.Z_to_idx = Z_to_idx  # Now an array instead of dict
        self.n_types = len(unique_Z)

        # Create radial grids for each nucleus
        r_grids = []
        ao_values = []
        self.s_indices_per_atom = []
        
        for atom_id in range(self.n_nuclei):
            r_grid = jnp.linspace(1e-8, 1.5, self.n_radial)
            coords = jnp.zeros((self.n_radial, 3))
            coords = coords.at[:,0].set(r_grid)
            coords = coords + self.coords[atom_id]
            
            # Get all basis function indices for this atom
            shell_ids = []
            for i in range(mol.nbas):
                if mol.bas_atom(i) == atom_id:
                    shell_ids.append(i)
            
            # Find s-type shells and their AO indices
            s_shells = []
            ao_idx_start = 0
            for i in range(mol.nbas):
                if i in shell_ids and mol.bas_angular(i) == 0:
                    s_shells.append(i)
                if i < shell_ids[0]:  # Count AOs before this atom
                    nctr = mol.bas_nctr(i)
                    l = mol.bas_angular(i)
                    ao_idx_start += (2*l + 1) * nctr
            
            # Store s-type AO indices for this atom
            s_ao_indices = []
            for shell_id in s_shells:
                nctr = mol.bas_nctr(shell_id)
                for c in range(nctr):
                    s_ao_indices.append(ao_idx_start)
                    ao_idx_start += 1
            self.s_indices_per_atom.append(jnp.array(s_ao_indices))
            
            # Evaluate s-type AOs using shls_slice
            if s_shells:  # Only evaluate if we have s-type shells
                shls_slice = (min(s_shells), max(s_shells) + 1)
                ao_values_r = mol.eval_gto('GTOval_sph', coords, shls_slice=shls_slice)
                # Ensure we have correct shape (ngrids, nao)
                if ao_values_r.ndim == 1:
                    ao_values_r = ao_values_r.reshape(-1, 1)
            else:
                # Create empty array if no s-type orbitals
                ao_values_r = jnp.zeros((self.n_radial, 0))
                
            r_grids.append(r_grid)
            ao_values.append(ao_values_r)
        
        self.r_grids = jnp.array(r_grids)
        # Store as list instead of JAX array since shapes may differ
        self.ao_values = ao_values  # Changed from jnp.array(ao_values)
        
        # Instead of MO transformation, just use 1s orbital values
        sao_sums = []
        for atom_id in range(self.n_nuclei):
            s_ao_vals = jnp.array(self.ao_values[atom_id])  # (n_radial, n_s_orbs)
            # For now, just take the first s-orbital (1s) contribution
            # Assuming first s-orbital in the basis set is 1s
            sao_sum = s_ao_vals[:, 0]  # Only use 1s orbital
            sao_sums.append(sao_sum)
        
        # Instead of storing CubicSpline objects, store their coefficients
        self.spline_coeffs = []
        self.spline_xs = []
        for i in range(self.n_nuclei):
            x = np.array(self.r_grids[i])
            y = np.array(sao_sums[i])
            spline = CubicSpline(x, y, bc_type='natural')
            dx = np.diff(x)
            # Convert coefficients to segment form for each interval
            # SciPy stores coefficients in descending power order: [a3, a2, a1, a0]
            # where p(x) = a3*(x-x0)^3 + a2*(x-x0)^2 + a1*(x-x0) + a0
            self.spline_xs.append(jnp.array(x))
            self.spline_coeffs.append(jnp.array(spline.c))
        
        # Add mapping from Z_idx to first nucleus of that type
        self.Z_idx_to_nucleus = []
        for Z_idx, Z in enumerate(self.unique_Z):
            nucleus_idx = int(np.where(self.charges == Z)[0][0])
            self.Z_idx_to_nucleus.append(nucleus_idx)
        self.Z_idx_to_nucleus = jnp.array(self.Z_idx_to_nucleus)
        
    def init_params(self):
        """Initialize parameter dictionary structure."""
        # Initialize parameters for each unique nuclear type
        params = {
            'rc': jnp.array([1.0/float(Z) for Z in self.unique_Z]),  # Use unique_Z array
            'X4': jnp.zeros(self.n_types),  # Only store X4 (phi_0) values
        }
        
        # Initialize X4 and compute alpha coefficients for each nucleus type
        for Z_idx, Z in enumerate(self.unique_Z):
            # Find first nucleus of this type
            nucleus_idx = jnp.where(self.charges == Z)[0][0]
            phi_0 = self.eval_mo_at_r(nucleus_idx, 0.0)
            params['X4'] = params['X4'].at[Z_idx].set(jnp.log(abs(phi_0))*1.1)
        
        # Update all alpha coefficients
        #self._update_alpha_coeffs(params)
        self._validate_params(params)
        return params
    
    def _validate_params(self, params):
        """Validate parameter shapes."""
        assert params['rc'].shape == (self.n_types,), f"rc shape {params['rc'].shape} != {(self.n_types,)}"
        assert params['X4'].shape == (self.n_types,), f"X4 shape {params['X4'].shape} != {(self.n_types,)}"

    def _compute_X_values(self, Z_idx, rc, X4):
        """Compute all X values given rc and X4."""
        Z = self.unique_Z[Z_idx]
        # Use pre-computed mapping instead of jnp.where
        nucleus_idx = self.Z_idx_to_nucleus[Z_idx]
        
        phi_rc_vals = self._get_phi_s_derivatives(nucleus_idx, rc)
        
        X = jnp.zeros(5)
        X = X.at[0].set(jnp.log(abs(phi_rc_vals[0])))  # X₁ = ln|φ(rc)|
        X = X.at[1].set(phi_rc_vals[1]/phi_rc_vals[0])  # X₂ = φ'(rc)/φ(rc)
        X = X.at[2].set(phi_rc_vals[2]/phi_rc_vals[0])  # X₃ = φ''(rc)/φ(rc)
        X = X.at[3].set(-Z)  # X₄ = -Z (cusp condition)
        X = X.at[4].set(X4)  # X₅ = ln|φ(0)|
        
        return X

    def _update_alpha_coeffs(self, params):
        """Compute but don't store polynomial coefficients."""
        poly_coeffs = jnp.zeros((self.n_types, 5))
        
        for Z_idx, Z in enumerate(self.unique_Z):
            rc = params['rc'][Z_idx]
            X4 = params['X4'][Z_idx]
            
            # Compute X values and alpha coefficients
            X = self._compute_X_values(Z_idx, rc, X4)
            alpha = self._compute_alpha_coeffs(Z, rc, X)
            poly_coeffs = poly_coeffs.at[Z_idx].set(alpha)
        
        return poly_coeffs  # Just return without storing in params

    def _cutoff_function(self, r, rc):
        """Smooth cutoff function using inverse polynomial.
        
        Args:
            r: Distance from nucleus
            rc: Cutoff radius
        """
        n = 3
        return 1.0 / (1.0 + (r/rc)**n)
    
    def _eval_poly(self, r, coeffs):
        """Evaluate polynomial Σ(coeffs[l]*r^l)."""
        powers = jnp.arange(len(coeffs))
        return jnp.sum(coeffs * (r**powers))
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute(self, r1, r2, params):
        """Compute nuclear cusp correction for a single electron."""
        # Compute polynomial coefficients from current params
        poly_coeffs = jnp.zeros((self.n_types, 5))
        for Z_idx, Z in enumerate(self.unique_Z):
            rc = params['rc'][Z_idx]
            X4 = params['X4'][Z_idx]
            X = self._compute_X_values(Z_idx, rc, X4)
            alpha = self._compute_alpha_coeffs(Z, rc, X)
            poly_coeffs = poly_coeffs.at[Z_idx].set(alpha)

        def scan_nuclei(carry, nucleus_idx):
            total = carry
            # Get distance from electron to this nucleus
            dr = r1 - self.coords[nucleus_idx]
            r = jnp.sqrt(jnp.sum(dr**2))
            
            # Use array indexing instead of dictionary lookup
            Z = self.charges[nucleus_idx]
            Z_idx = self.Z_to_idx[Z.astype(jnp.int32)]
            rc = params['rc'][Z_idx]
            
            # Use where to conditionally evaluate only when r <= rc
            def evaluate_contribution(r):
                # Use computed poly_coeffs instead of params
                coeffs = poly_coeffs[Z_idx]
                
                # Compute φ_cusp = exp(poly(r))
                poly_val = jnp.where(r<=rc, self._eval_poly(r, coeffs), 0.0)
                phi_cusp = jnp.exp(poly_val)
                
                # Get φ_s value with numerical safeguard
                phi_s = jnp.where(r<=rc, self.eval_mo_at_r(nucleus_idx, r), 1.0)
                
                # Add small constants to prevent division by zero or log(0)
                eps = 0.0
                ratio = (phi_cusp + eps)/(phi_s + eps)
                # Use log1p for better numerical stability when ratio is close to 1
                log_term = jnp.log(ratio)
                
                # Combine using cutoff
                cutoff = self._cutoff_function(r, rc)
                return log_term * cutoff
            
            # Only evaluate when r <= rc, otherwise return 0
            contrib = jnp.where(r <= rc, 
                              evaluate_contribution(r),
                              0.0)
            
            return total + contrib, None
        
        # Sum over all nuclei
        total, _ = jax.lax.scan(scan_nuclei, 0.0, jnp.arange(self.n_nuclei))
        
        return total/(self.nelectron-1)  # Normalize by nelectron - 1
    
    # overwrite get_log_grads_r2 to give the same value as get_log_grads_r1 from
    # parent class
    #def get_log_grads_r1(self, r1, r2, params):
    #    grad_u, lap_u = super().get_log_grads_r1(r1, r2, params)
    #    return grad_u, lap_u

    def get_log_grads_r2(self, r1, r2, params):
        return self.get_log_grads_r1(r2, r1, params)

    def eval_mo_at_r(self, nucleus_idx, r):
        """JAX-compatible cubic spline evaluation."""
        # Handle scalar vs array inputs differently
        r_is_array = hasattr(r, 'shape') and r.ndim > 0
        
        # Stack all spline data for vectorized operations
        xs = jnp.stack(self.spline_xs)
        coeffs = jnp.stack(self.spline_coeffs)
        
        # Select data for this nucleus
        mask = jnp.arange(self.n_nuclei) == nucleus_idx
        x = jnp.sum(xs * mask[:, None], axis=0)
        c = jnp.sum(coeffs * mask[:, None, None], axis=0)
        
        # Process differently based on input type
        if r_is_array:
            # Vectorized processing for array inputs
            dx = x[1] - x[0]
            
            # Compute all indices and t values at once (JAX-traceable)
            indices = jnp.clip((r - x[0]) / dx, 0, len(x)-2)
            indices_int = jnp.floor(indices).astype(jnp.int32)
            
            # Calculate local coordinates relative to left endpoint
            x_i = jnp.take(x, indices_int)
            t = r - x_i
            
            # Use vmap to apply the spline evaluation to each element
            def eval_spline_at_idx(idx, t):
                # Get coefficients for this interval - use JAX-friendly indexing
                # SciPy's coefficients are stored in descending power order:
                # c[0,idx] is coefficient of (x-x_i)³
                # c[1,idx] is coefficient of (x-x_i)²
                # c[2,idx] is coefficient of (x-x_i)
                # c[3,idx] is constant term
                return c[3,idx] + t*(c[2,idx] + t*(c[1,idx] + t*c[0,idx]))
            
            # Use vmap instead of fori_loop for better traceability
            values = jax.vmap(eval_spline_at_idx)(indices_int, t)
            return values
        else:
            # Scalar processing - use same logic as array processing
            dx = x[1] - x[0]
            # Compute index and t value same as array case
            index = jnp.clip((r - x[0]) / dx, 0, len(x)-2)
            indices_int = jnp.floor(index).astype(jnp.int32)
            # Calculate local coordinate
            x_i = jnp.take(x, indices_int)
            t = r - x_i
            # Use same coefficient order as array case
            return c[3,indices_int] + t*(c[2,indices_int] + t*(c[1,indices_int] + t*c[0,indices_int]))
    
    def _get_phi_s_derivatives(self, nucleus_idx, r):
        """JAX-compatible derivatives computation."""
        # Convert inputs to arrays and combine spline data
        xs = jnp.stack(self.spline_xs)
        coeffs = jnp.stack(self.spline_coeffs)
        
        # Select data for this nucleus using where/multiply
        mask = jnp.arange(self.n_nuclei) == nucleus_idx
        x = jnp.sum(xs * mask[:, None], axis=0)
        c = jnp.sum(coeffs * mask[:, None, None], axis=0)
        
        # Find interval using safe integer operations
        dx = x[1] - x[0]
        index = jnp.clip((r - x[0]) / dx, 0, len(x)-2)
        index_int = jnp.floor(index).astype(jnp.int32)
        
        # Get local coordinate
        t = r - x[index_int]  # Note: not normalized by dx here
        
        # Get coefficients for this interval
        c0 = jnp.sum(c[0] * (jnp.arange(len(c[0])) == index_int))  # cubic term
        c1 = jnp.sum(c[1] * (jnp.arange(len(c[1])) == index_int))  # quadratic term
        c2 = jnp.sum(c[2] * (jnp.arange(len(c[2])) == index_int))  # linear term
        c3 = jnp.sum(c[3] * (jnp.arange(len(c[3])) == index_int))  # constant term
        
        # Value: p(t) = c0*t³ + c1*t² + c2*t + c3
        phi = c3 + t*(c2 + t*(c1 + t*c0))
        
        # First derivative: p'(t) = 3c0*t² + 2c1*t + c2
        phi_d1 = c2 + t*(2*c1 + t*3*c0)
        
        # Second derivative: p''(t) = 6c0*t + 2c1
        phi_d2 = 2*c1 + t*6*c0
        
        return jnp.array([phi, phi_d1, phi_d2])

    def _compute_alpha_coeffs(self, Z, rc, X_vals):
        """Compute α coefficients from X values and rc.
        
        Args:
            Z: Nuclear charge
            rc: Cutoff radius
            X_vals: Array of X1-X5 values
            
        Returns:
            Array of α coefficients [α₀, α₁, α₂, α₃, α₄]
        """
        X1, X2, X3, X4, X5 = X_vals
        
        alpha = jnp.zeros(5)
        # α₀ = X₅
        alpha = alpha.at[0].set(X5)
        # α₁ = X₄
        alpha = alpha.at[1].set(X4)
        # α₂ = 6X₁/rc² - 3X₂/rc + X₃/2 - 3X₄/rc - 6X₅/rc² - X₂²/2
        alpha = alpha.at[2].set(
            6*X1/rc**2 - 3*X2/rc + X3/2 - 3*X4/rc - 6*X5/rc**2 - X2**2/2
        )
        # α₃ = -8X₁/rc³ + 5X₂/rc² - X₃/rc + 3X₄/rc² + 8X₅/rc³ + X₂²/rc
        alpha = alpha.at[3].set(
            -8*X1/rc**3 + 5*X2/rc**2 - X3/rc + 3*X4/rc**2 + 8*X5/rc**3 + X2**2/rc
        )
        # α₄ = 3X₁/rc⁴ - 2X₂/rc³ + X₃/(2rc²) - X₄/rc³ - 3X₅/rc⁴ - X₂²/(2rc²)
        alpha = alpha.at[4].set(
            3*X1/rc**4 - 2*X2/rc**3 + X3/(2*rc**2) - X4/rc**3 - 3*X5/rc**4 - X2**2/(2*rc**2)
        )
        return alpha
