import numpy as np
from pyscf import gto, scf, ao2mo
from pytc.xtc import XTC

def write_fcidump(filename, mol):
    """
    Write integrals in FCI dump format.

    Parameters:
    filename (str): The name of the output file.
    mol (Mole): The PySCF molecule object.
    """
    # Perform SCF calculation
    mf = scf.RHF(mol)
    mf.kernel()

    # Create XTC object
    my_xtc = XTC(mf, None, grid_lvl=2)

    # Extract 1body and 2body integrals and ecore from the XTC object
    h1e = my_xtc.get_1b()
    h2e = my_xtc.get_2b()
    ecore = my_xtc.get_const()

    # Get the number of orbitals
    n_orb = h1e.shape[0]

    with open(filename, 'w') as f:
        # Write header
        f.write('&FCI\n')
        f.write('NORB= {}\n'.format(n_orb))
        f.write('NELEC= {}\n'.format(mol.nelectron))
        f.write('MS2= 0\n')
        f.write('ORBSYM= {}\n'.format('1 ' * n_orb))
        f.write('ISYM= 1\n')
        f.write('&END\n')

        # Write 1-body integrals
        for i in range(n_orb):
            for j in range(n_orb):
                if abs(h1e[i, j]) > 1e-15:
                    f.write('{:22.15E} {:3d} {:3d} {:3d} {:3d}\n'.format(h1e[i, j], i + 1, j + 1, 0, 0))

        # Write 2-body integrals
        for i in range(n_orb):
            for j in range(n_orb):
                for k in range(n_orb):
                    for l in range(n_orb):
                        if abs(h2e[i, j, k, l]) > 1e-15:
                            f.write('{:22.15E} {:3d} {:3d} {:3d} {:3d}\n'.format(h2e[i, j, k, l], i + 1, j + 1, k + 1, l + 1))

        # Write core energy
        f.write('{:22.15E} {:3d} {:3d} {:3d} {:3d}\n'.format(ecore, 0, 0, 0, 0))
