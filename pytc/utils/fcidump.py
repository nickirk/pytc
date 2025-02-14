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
        # Write header in the new format
        header = "&FCI NORB={:4d},NELEC= {:d},MS2=0,\n".format(n_orb, n_elec)
        header += "  ORBSYM={}\n".format("1," * n_orb)
        header += "  ISYM=1,\n &END\n"
        f.write(header)

        # Write 2-body integrals
        for i in range(n_orb):
            for j in range(n_orb):
                for k in range(n_orb):
                    for l in range(n_orb):
                        if abs(h2e[i, j, k, l]) > 1e-15:
                            f.write('{:22.15E} {:3d} {:3d} {:3d} {:3d}\n'.format(h2e[i, j, k, l], i + 1, j + 1, k + 1, l + 1))

        # Write 1-body integrals
        for i in range(n_orb):
            for j in range(i,n_orb):
                if abs(h1e[i, j]) > 1e-15:
                    f.write('{:22.15E} {:3d} {:3d} {:3d} {:3d}\n'.format(h1e[i, j], i + 1, j + 1, 0, 0))

        
        # Write core energy
        f.write('{:22.15E} {:3d} {:3d} {:3d} {:3d}\n'.format(ecore, 0, 0, 0, 0))


def read_fcidump(filename):
    """
    Reads a FCIDUMP file and extracts electronic structure parameters, one-electron integrals,
    two-electron integrals, and the core energy.
    The FCIDUMP file is expected to contain header information with keywords 'norb', 'nelec',
    and 'ms2', followed by integral values. The header is processed until a line containing
    either "&end" or "/" is found. After this, the file lines are parsed to populate:
        - Core energy (ecore)
        - One-electron integrals (h1e): a symmetric matrix of size (n_sites x n_sites)
        - Two-electron integrals (g2e): a tensor of size (n_sites x n_sites x n_sites x n_sites)
    Parameters:
        filename (str): The path to the FCIDUMP file to be read.
    Returns:
        tuple:
            n_sites (int): Number of orbitals/sites (extracted from 'norb').
            n_elec (int): Number of electrons (extracted from 'nelec').
            ecore (float): Core energy, accumulated from lines where indices sum to -4.
            h1e (numpy.ndarray): One-electron integrals matrix with shape (n_sites, n_sites).
            g2e (numpy.ndarray): Two-electron integrals tensor with shape 
                                 (n_sites, n_sites, n_sites, n_sites).
    """
    with open(filename, 'r') as f:
        lines = [x.lower().strip() for x in f.readlines()]
        lbrk = [il for il, l in enumerate(lines) if "&end" in l or "/" in l][0]
        keys = {'norb': None, 'nelec': None, 'ms2': None}
        for k in keys:
            keys[k] = [int(l.split(k)[1].split('=')[1].split(',')[0]) for l in lines if k in l][0]
        # print(keys)
        n_sites = keys['norb']
        n_elec = keys['nelec']
        spin = keys['ms2']
        h1e = np.zeros((n_sites, n_sites),dtype=float)
        g2e = np.zeros((n_sites, n_sites, n_sites, n_sites),dtype=float)
        ecore = 0
        for l in lines[lbrk + 1:]:
            if len(l.split()) == 0:
                continue
            a, i, j, k, l = l.split()
            i, j, k, l = [int(x) - 1 for x in [i, j, k, l]]
            
            #print(a)
        
            if i + j + k + l == -4:
                ecore += float(a)
            
            elif k + l ==  -2:
                h1e[i, j] = float(a)
                h1e[j, i] = float(a)
            # print(h1e[i,j],h1e[j,i])

            else:
                g2e[i, j, k, l] = float(a)
                # g2e[k, l, i, j] = float(a)


    # print('h1e norm = ', np.linalg.norm(h1e))
    # print('g2e norm = ', np.linalg.norm(g2e))

    return n_sites, n_elec, ecore, h1e, g2e
