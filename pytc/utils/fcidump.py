import numpy as np

def write_fcidump(filename, h1e, h2e, ecore, n_orb, n_elec):
    """
    Write integrals in FCI dump format.

    Parameters:
    filename (str): The name of the output file.
    h1e (ndarray): 1-body integrals.
    h2e (ndarray): 2-body integrals.
    ecore (float): Core energy.
    n_orb (int): Number of orbitals.
    n_elec (int): Number of electrons.
    """
    with open(filename, 'w') as f:
        # Write header
        f.write('&FCI\n')
        f.write('NORB= {}\n'.format(n_orb))
        f.write('NELEC= {}\n'.format(n_elec))
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
