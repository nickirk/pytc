# Interpolative Separable Density Fitting (ISDF) for Time-Dependent Density Functional Theory

This document outlines the theoretical background, computational bottlenecks of traditional Gaussian Density Fitting (DF) TDDFT, and the detailed mathematical and algorithmic implementation of ISDF-accelerated TDDFT within `pytc`.

---

## 1. Background on Traditional Gaussian DF TDDFT

Time-Dependent Density Functional Theory (TDDFT) in the linear response regime is typically solved via the Casida equations (or Davidson diagonalization), which seek the excitation energies $\omega$ and transition amplitudes $(X, Y)$ by solving:

$$
\begin{pmatrix} A & B \\ B & A \end{pmatrix}
\begin{pmatrix} X \\ Y \end{pmatrix}
= \omega
\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
\begin{pmatrix} X \\ Y \end{pmatrix}
$$

The orbital rotation matrices $A$ and $B$ are defined by the difference in orbital energies ($\epsilon_a - \epsilon_i$) and the two-electron coupling matrix $K$:

$$
(A+B)_{ia, jb} = (\epsilon_a - \epsilon_i) \delta_{ij} \delta_{ab} + 2 (ia|jb) - c_X (ib|ja) + 2 f_{xc}^{ia, jb}
$$

### The Coupling Terms

The computational bottleneck in conventional TDDFT forms is the evaluation of the matrix-vector products with the two-electron integrals. For a trial vector $z_{jb}$:

1. **Coulomb ($J$) Contraction:**
   $$ V_{ia}^J = \sum_{jb} (ia|jb) z_{jb} $$

2. **Exact Exchange ($K$) Contraction:**
   $$ V_{ia}^K = \sum_{jb} (ib|ja) z_{jb} $$
   *(Applies the non-local exchange potential; required for hybrid functionally like PBE0 or range-separated hybrids like $\omega$B97M-V)*

3. **Exchange-Correlation ($f_{xc}$) Contraction:**
   $$ V_{ia}^{xc} = \sum_{jb} \int d\mathbf{r} \, \phi_i(\mathbf{r}) \phi_a(\mathbf{r}) f_{xc}(\mathbf{r}) \phi_j(\mathbf{r}) \phi_b(\mathbf{r}) z_{jb} $$
   *(Captures complex quantum correlations based on the functional derivative of the continuous XC density grid)*

### Computational Scaling of Gaussian DF

To avoid evaluating $\mathcal{O}(N_{basis}^4)$ analytical four-center integrals $(ia|jb)$, Gaussian Density Fitting (DF) approximates orbital products using an auxiliary Gaussian basis set $|P\rangle$:

$$ \phi_i(\mathbf{r}) \phi_a(\mathbf{r}) \approx \sum_P C_{ia}^P \chi_P(\mathbf{r}) $$

This transforms the two-electron integrals to:

$$ (ia|jb) \approx \sum_{PQ} (ia|P) [J^{-1}]_{PQ} (Q|jb) = \sum_{P} L_{ia}^P L_{jb}^P $$

Where $L_{ia}^P$ are the three-center DF tensors. 
Even with Gaussian DF, the scaling of the Davidson matrix-vector products remains $\mathcal{O}(N_{aux} N_{occ} N_{vir} N_{iter})$ for the Coulomb term, which translates to a formal dense scaling of **$\mathcal{O}(N^3)$ per Davidson iteration**.
The exact exchange ($K$) term is even more demanding, requiring tensor contractions of shapes $(N_{aux}, N_{occ}, N_{vir})$ against the trial vectors, driving severe memory bottlenecks (scaling memory as $N^3$ and compute FLOPs explicitly as $\mathcal{O}(N^4)$ for exact exchange generation in purely iterative schemes if formed explicitly).

---

## 2. Motivating ISDF

The Interpolative Separable Density Fitting (ISDF) approximation completely bypasses evaluating analytical Gaussian three-center integrals $(ia|P)$. Instead of expanding orbital products into standard atom-centered Gaussians, ISDF spatially factorizes them directly on an optimal real-space grid (the "pivots" $\mathbf{r}_\mu$):

$$ \phi_i(\mathbf{r}) \phi_j(\mathbf{r}) \approx \sum_{\mu=1}^{N_{piv}} \zeta_\mu(\mathbf{r}) \phi_i(\mathbf{r}_\mu) \phi_j(\mathbf{r}_\mu) $$

Here, $\zeta_\mu(\mathbf{r})$ are non-local interpolation functions, and $\phi_i(\mathbf{r}_\mu)$ are simply the physical molecular orbital (MO) coefficients evaluated at the spatial coordinate of pivot $\mu$.

### Overcoming Gaussian DF Bottlenecks

ISDF provides immense mathematical simplifications:
1. **No Three-Center Integrals:** $L_{ia}^P$ is completely avoided. The coupling terms are formed entirely from discrete MOs on the grid.
2. **Memory Decoupling:** Instead of storing a massive $N_{aux} \times N_{occ} \times N_{vir}$ 3D tensor, ISDF stores the MOs directly at the pivots: $C_o$ ($N_{piv} \times N_{occ}$) and $C_v$ ($N_{piv} \times N_{vir}$). Memory drops from $\mathcal{O}(N^3)$ to **$\mathcal{O}(N^2)$**, allowing massive systems to easily fit in RAM.
3. **Hardware Acceleration:** By rendering standard analytical integration into dense matrix-matrix multiplications (DGEMM) mapping pivots to MOs, ISDF maximally exploits modern BLAS optimization and GPU architectures.

### Replacing J and K Contractions

Under ISDF, the four-center contraction is replaced by a compressed spatial integral:

$$ (ia|jb) \approx \sum_{\mu \nu} \phi_i(\mathbf{r}_\nu) \phi_a(\mathbf{r}_\nu) J_{\nu \mu} \phi_j(\mathbf{r}_\mu) \phi_b(\mathbf{r}_\mu) $$

Where the **ISDF Coulomb Kernel ($J_{\nu \mu}$)** is pre-computed entirely from the analytical spatial definitions of the interpolants:
$$ J_{\nu \mu} = \iint d\mathbf{r} d\mathbf{r}' \frac{\zeta_\nu(\mathbf{r}) \zeta_\mu(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} $$

For Range-Separated Hybrids (RSH), the Long-Range (LR) exact exchange requires a complementary Long-Range Coulomb Kernel ($J_{\nu \mu}^{LR}$) modulated by the error function:
$$ J_{\nu \mu}^{LR} = \iint d\mathbf{r} d\mathbf{r}' \frac{\text{erf}(\omega |\mathbf{r} - \mathbf{r}'|)}{|\mathbf{r} - \mathbf{r}'|} \zeta_\nu(\mathbf{r}) \zeta_\mu(\mathbf{r}') $$

#### Floating Gaussian Basis Evaluation

While the above formulation integrates $J_{\nu\mu}$ directly on the numerical grid, evaluating the exact Coulomb interaction in real-space pairs scales at a prohibitive $\mathcal{O}(N_{grid}^2)$. To bypass this massive computational bottleneck, `pytc` employs an auxiliary **Floating Gaussian Basis** constructed dynamically around the extracted ISDF pivot coordinates.

Instead of generic atom-centered bases, uncontracted s-type Gaussian functions are centered exactly at the chosen interpolant pivots ($\mathbf{r}_\mu$). The exponents ($\alpha_\mu$) are dynamically determined by the local pivot density via a Nearest-Neighbor KDTree lookup:
$$ \alpha_\mu = \frac{\gamma}{h_\mu^2} $$
Where $h_\mu$ is the distance to the nearest neighbor (or local mean radius) and $\gamma$ is a coverage factor. To enhance basis flexibility, multiple $\gamma$ values (e.g., $0.25, 0.5, 1.0$) can be used to generate multiple s-functions per pivot.

By projecting the numerical ISDF interpolants $\zeta_\mu(\mathbf{r})$ onto this compact floating basis set using an SVD-based pseudo-inverse, the discrete ISDF kernels are evaluated using mathematically exact, rapid analytical PySCF $(P|Q)$ Coulomb integrals. This reduces the kernel construction cost dramatically while simultaneously eliminating grid-based singularity noise for the long-range tails.

**SVD-Based Pseudo-Inverse Projection:**
The projection solves the linear system representing the overlap of the dynamically constructed auxiliary basis function ($\chi_P$) and the target ISDF interpolating functions ($\zeta_\mu$). Instead of a direct inverse, an SVD-based pseudo-inverse is employed to evaluate the expansion coefficients ($d$).

1. **Auxiliary Metric Matrix ($S_{PQ}$)**: First, it evaluates the overlap of the uncontracted gaussians defined dynamically at the pivot centers: 
   $$ S_{PQ} = \sum_{r} w(r) \chi_P(r) \chi_Q(r) $$
2. **Overlap Vector ($V_{P\mu}$)**: It computes the projection of the auxiliary basis onto the original ISDF interpolants across the real-space grid: 
   $$ V_{P\mu} = \sum_{r} w(r) \chi_P(r) \zeta_\mu(r) $$
3. **Truncated SVD**: Since the floating uncontracted s-functions can form a nearly linearly dependent basis (especially with multiple gammas), $S_{PQ}$ is often ill-conditioned. Scipy's `linalg.svd` factors $S_{PQ} = U \Sigma V^H$. A strict cutoff threshold zeroes out singular values below the limit $s_0 \times \text{rcond}$.
4. **Coefficient Matrix ($d$)**: Using the stable pseudoinverse $S_{PQ}^{+}$, the robust projection coefficients are constructed:
   $$ d = S_{PQ}^{+} V_{P\mu} $$
5. **Analytical J-Kernel**: The matrix is reconstructed exactly using analytical 2-center 2-electron integrals $J_{PQ} = (P|Q)$, circumventing grid-based integration errors for the long-range Coulomb singularity:
   $$ J_{\mu\nu} = d^T J_{PQ} d $$

### ISDF Exchange-Correlation kernels ($f_{xc}$)
Just like the Coulomb kernel, the TDDFT XC response matrix $f_{xc}$ can be compressed from the massive $N_{grid} \times N_{grid}$ grid dimensions down to the compact $N_{piv} \times N_{piv}$ auxiliary space.

**LDA Kernel:**
The local density approximation depends solely on the density $\rho$.
$$ [V_{xc}^{LDA}]_{\nu \mu} = \sum_{r} \zeta_\nu(\mathbf{r}) w(\mathbf{r}) f_{xc}^{LDA}(\mathbf{r}) \zeta_\mu(\mathbf{r}) $$

**GGA Kernel:**
Generalized Gradient Approximations (e.g. PBE) depend on both the density $\rho$ and its gradient $\nabla \rho$. When evaluating linear response derivatives on the grid, evaluating the product rule generates highly oscillatory functions $\nabla(\phi_i \phi_a) = \phi_i \nabla \phi_a + \phi_a \nabla \phi_i$. 

Interpolating gradients using standard scalar density interpolators ($\zeta_\mu^{\phi}$) yields catastrophic numerical errors and non-positive-definite solutions. `pytc` solves this by explicitly generating a unique basis of **Gradient Interpolators** ($\zeta_\mu^{\nabla}$), and packing the full multi-dimensional coupling tensor across the 4 physical density dimensions ($x, y, z, \rho$):
$$ [V_{fxc}^{GGA}]_{y x}^{\nu \mu} = \sum_{r} \zeta_{\nu, y}(\mathbf{r}) w(\mathbf{r}) f_{xy}^{GGA}(\mathbf{r}) \zeta_{\mu, x}(\mathbf{r}) $$
*(Where $\zeta_{\mu, x=0} = \zeta^{\phi}$ and $\zeta_{\mu, x \in (1,2,3)} = \zeta^{\nabla}$)*

### Explicit ISDF Contraction Equations and Scaling

Extracting physical eigenvalues in the Davidson eigensolver requires applying these integrals to the dense transition trial vectors $z_{ia}$. In ISDF, this is done through a sequence of successive Dense Matrix-Matrix Multiplications (BLAS DGEMM), transitioning vectors between the Molecular Orbital basis and the ISDF Auxiliary space.

**1. Coulomb (J) Contraction:**
$$ V_{ia}^J = \sum_{\mu} \phi_i(\mathbf{r}_\mu) \phi_a(\mathbf{r}_\mu) \sum_{\nu} J_{\mu \nu} \sum_{jb} \phi_j(\mathbf{r}_\nu) \phi_b(\mathbf{r}_\nu) z_{jb} $$
*Algorithmic sequence:*
1. $z_b^\nu = \sum_j \phi_j(\mathbf{r}_\nu) z_{jb}$ (*$\mathcal{O}(N_{occ} N_{vir} N_{piv})$ scaling operation mapping $N_{occ} \rightarrow N_{piv}$*)
2. $\rho^\nu = \sum_b \phi_b(\mathbf{r}_\nu) z_b^\nu$ (*$\mathcal{O}(N_{vir} N_{piv})$ Hadamard product forming the physical density response*)
3. $u^\mu = \sum_\nu J_{\mu \nu} \rho^\nu$ (*$\mathcal{O}(N_{piv}^2)$ kernel application*)
4. $w_a^\mu = \phi_a(\mathbf{r}_\mu) u^\mu$ (*$\mathcal{O}(N_{vir} N_{piv})$ scalar broadcast*)
5. $V_{ia}^J = \sum_\mu \phi_i(\mathbf{r}_\mu) w_a^\mu$ (*$\mathcal{O}(N_{occ} N_{vir} N_{piv})$ back-projection $N_{piv} \rightarrow N_{occ}$*)

**2. Exact Exchange (K) Contraction:**
$$ V_{ia}^K = \sum_{\mu} \phi_i(\mathbf{r}_\mu) \sum_{\nu} \phi_a(\mathbf{r}_\nu) J_{\mu \nu} \sum_{jb} \phi_j(\mathbf{r}_\mu) \phi_b(\mathbf{r}_\nu) z_{jb} $$
*Algorithmic sequence:*
1. $z_j^\nu = \sum_b \phi_b(\mathbf{r}_\nu) z_{jb}$ (*$\mathcal{O}(N_{occ} N_{vir} N_{piv})$ scaling DGEMM*)
2. $\tilde{Z}_j^\mu = \phi_j(\mathbf{r}_\mu) z_j^\nu$ (*$\mathcal{O}(N_{occ} N_{piv}^2)$ Kronecker product matching density tails*)
3. $U_a^\mu = \sum_\nu J_{\mu \nu} \tilde{Z}_j^\mu$ (*$\mathcal{O}(N_{occ} N_{piv}^2)$ non-local kernel expansion*)
4. $w_a^\nu = \sum_j \dots$ (*Broadcast/reshape steps mapping $U$ into target form*)
5. $V_{ia}^K = \sum_\mu w_{a}^\mu \phi_i(\mathbf{r}_\mu)$ (*$\mathcal{O}(N_{occ} N_{vir} N_{piv})$ DGEMM back-projection*)

LDA and GGA $f_{xc}$ contractions are structurally identical to the $J$ sequence above, simply replacing the pure $1/r$ interaction $J_{\mu \nu}$ with the highly localized functional derivative blocks $V_{\mu \nu}^{LDA}$ or $V_{yx}^{\nu \mu, GGA}$. 

> [!NOTE] 
> **Scaling Analysis:**
> The formal algorithmic order of operations for TDHF or TDDFT in ISDF is rigidly bound by **$\mathcal{O}(N_{occ} N_{piv} N_{vir}) \propto \mathcal{O}(N^3)$**. While ISDF is occasionally mischaracterized in quantum chemistry literature as strictly a "Linear Scaling" algorithm because it bypasses evaluating the massive $\mathcal{O}(N^4)$ integral formulation $(ia|jb)$, reconstructing the dense 2D transition density $z_{ia}$ through the grid basis mathematically necessitates block multiplications scaling as $N^3$. The immense acceleration in ISDF arises exclusively because evaluating dense matrix multiplications via hardware-optimized basic linear algebra (BLAS DGEMM) exhibits phenomenally low prefactors, entirely avoiding the index permutations explicitly required by typical 4D Gaussian tensor algebras.

---

## 3. Numerical Ill-Conditioning and Regularization

ISDF is fundamentally rooted in solving the normal equations to identify the optimal interpolation coefficients $C$:
$$ (P^T P) C = P^T A $$
*(Where $P$ represents the evaluated atomic orbitals on the sub-sampled grid)*

When evaluating highly diffuse or polarized basis sets (such as `def2-svpd`), the matrix $P^T P$ becomes severely ill-conditioned (condition numbers exceeding $10^{16}$). This leads to catastrophic cancellations, rank-deficiency inside the Cholesky solver, and ultimately produces unphysical negative eigenvalues or infinities in the solved interpolants ($\zeta_\mu$). 

### Tikhonov Regularization (`rcond`)

To stabilize the interpolation, `pytc` applies Tikhonov regularization (Ridge Regression) internally during the `solve_normal_equations_batch` process:
$$ (P^T P + \lambda I) C = P^T A $$

The regularization parameter $\lambda$ is scaled by the mean diagonal of the overlap matrix and controlled by the parameter **`isdf_rcond`**.

**Optimal Tuning:**
- Setting `rcond` too high (e.g., $10^{-3}$) artificially smears the density representation. While it prevents crashing, it severely degrades the physical accuracy of the simulated TDDFT spectra (Mean Absolute Error skyrocketing to >0.15 eV).
- Setting `rcond` too low (e.g., $10^{-14}$) allows ill-conditioning to dominate, leading to NaN results or `FloatingPointError (Negative Eigenvalues in Davidson)`.
- **The optimal empirical value for analytical exactness in def2-svpd basis sets is $10^{-6}$ or $10^{-7}$.** At these values, numerical stability is maintained while achieving MAEs against rigorous analytical PySCF results strictly below $0.01$ eV across all primary TDDFT functions (LDA, PBE, and PBE0).
