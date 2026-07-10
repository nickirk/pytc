"""Deterministic water-cluster geometry generator for the VMC scaling benchmark
(task #1, #pro-pytc-efficiency-refactor). Timing/scaling systems only --
consistency and a genuine multi-element/multi-orientation structure matter,
not physical realism.

Recipe (per Felix's spec):
- Rigid experimental monomer: r(OH) = 0.9572 Angstrom, angle(HOH) = 104.52 deg.
- O atoms placed on a simple cubic grid, O-O spacing = 2.9 Angstrom.
- Every other monomer (checkerboard by grid-index parity) rotated 90 degrees
  about the z-axis relative to the base orientation, so no H...H contact is
  too short.
- Grid shape chosen per n_monomers: 2 -> (2,1,1), 4 -> (2,2,1), 8 -> (2,2,2),
  16 -> (4,2,2), 25 -> (5,5,1).
- No RNG anywhere -- fully reproducible from (n_monomers) alone.

Usage:
    python gen_water_cluster.py <n_monomers> -o <output.xyz>
"""
import argparse
import numpy as np

R_OH = 0.9572  # Angstrom
ANGLE_HOH_DEG = 104.52
OO_SPACING = 2.9  # Angstrom
MIN_DIST_ANGSTROM = 1.5  # sanity-check floor for any interatomic distance

# Grid shape (nx, ny, nz) per supported cluster size.
GRID_SHAPES = {
    2: (2, 1, 1),
    4: (2, 2, 1),
    8: (2, 2, 2),
    16: (4, 2, 2),
    25: (5, 5, 1),
}


def _base_monomer():
    """Single water monomer in its own frame: O at origin, molecule in the
    xz-plane, bisector of HOH along +x."""
    half_angle = np.radians(ANGLE_HOH_DEG) / 2.0
    o = np.array([0.0, 0.0, 0.0])
    h1 = np.array([R_OH * np.cos(half_angle), 0.0, R_OH * np.sin(half_angle)])
    h2 = np.array([R_OH * np.cos(half_angle), 0.0, -R_OH * np.sin(half_angle)])
    return o, h1, h2


def _rotate_z(vec, degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot @ vec


def generate_water_cluster(n_monomers):
    if n_monomers not in GRID_SHAPES:
        raise ValueError(
            f"n_monomers={n_monomers} not in supported set {sorted(GRID_SHAPES)}"
        )
    nx, ny, nz = GRID_SHAPES[n_monomers]

    atoms = []  # list of (element, x, y, z, monomer_index)
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if idx >= n_monomers:
                    break
                center = np.array(
                    [ix * OO_SPACING, iy * OO_SPACING, iz * OO_SPACING]
                )
                o, h1, h2 = _base_monomer()
                # Checkerboard rotation: alternate 90 deg by parity of the
                # flat grid index so neighboring monomers don't clash.
                if idx % 2 == 1:
                    o = _rotate_z(o, 90.0)
                    h1 = _rotate_z(h1, 90.0)
                    h2 = _rotate_z(h2, 90.0)
                atoms.append(("O", *(center + o), idx))
                atoms.append(("H", *(center + h1), idx))
                atoms.append(("H", *(center + h2), idx))
                idx += 1

    assert idx == n_monomers, f"grid shape {GRID_SHAPES[n_monomers]} only placed {idx}"
    return atoms


def _min_intermonomer_distance(atoms):
    """Minimum distance between atoms belonging to DIFFERENT monomers.
    Intra-molecular O-H bonds (~0.96A) are expected and excluded."""
    coords = np.array([a[1:4] for a in atoms])
    monomer = [a[4] for a in atoms]
    n = len(coords)
    dmin = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if monomer[i] == monomer[j]:
                continue
            d = np.linalg.norm(coords[i] - coords[j])
            if d < dmin:
                dmin = d
    return dmin


def write_xyz(atoms, output_file, n_monomers):
    with open(output_file, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(
            f"Water cluster (H2O){n_monomers}, cubic grid O-O={OO_SPACING}A, "
            f"checkerboard 90deg rotation, r(OH)={R_OH}A, "
            f"angle(HOH)={ANGLE_HOH_DEG}deg\n"
        )
        for el, x, y, z, _monomer_idx in atoms:
            f.write(f"{el} {x:>12.6f} {y:>12.6f} {z:>12.6f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a deterministic (H2O)n cluster XYZ for VMC scaling benchmarks."
    )
    parser.add_argument(
        "n_monomers", type=int, choices=sorted(GRID_SHAPES), help="Number of H2O monomers"
    )
    parser.add_argument("-o", "--output", default=None, help="Output XYZ filename")
    args = parser.parse_args()

    out = args.output or f"water{args.n_monomers}.xyz"
    atoms = generate_water_cluster(args.n_monomers)
    dmin = _min_intermonomer_distance(atoms)
    if dmin < MIN_DIST_ANGSTROM:
        raise RuntimeError(
            f"Sanity check failed: min inter-monomer distance {dmin:.3f}A "
            f"< floor {MIN_DIST_ANGSTROM}A for n_monomers={args.n_monomers}"
        )
    write_xyz(atoms, out, args.n_monomers)
    print(
        f"(H2O){args.n_monomers}: {len(atoms)} atoms, {10*args.n_monomers} electrons "
        f"(RHF), min inter-monomer distance = {dmin:.3f} A -> {out}"
    )
