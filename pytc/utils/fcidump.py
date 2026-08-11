import numpy as np

def write(filename, h1e, h2e, ecore, n_orb, n_elec):
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
        header = "&FCI NORB={:4d},NELEC= {:d},MS2=0,\n".format(n_orb, n_elec)
        header += "  ORBSYM={}\n".format("1," * n_orb)
        header += "  ISYM=1,\n &END\n"
        f.write(header)

        ii, jj, kk, ll = np.meshgrid(
            np.arange(n_orb), np.arange(n_orb),
            np.arange(n_orb), np.arange(n_orb), indexing='ij')
        ii = ii.ravel(); jj = jj.ravel(); kk = kk.ravel(); ll = ll.ravel()

        pair_le = (ii < kk) | ((ii == kk) & (jj <= ll))
        vals = h2e[ii, jj, kk, ll]
        mask = pair_le & (np.abs(vals) > 1e-15)
        for v, i, j, k, l in zip(vals[mask], ii[mask], jj[mask], kk[mask], ll[mask]):
            f.write('{:22.15E} {:3d} {:3d} {:3d} {:3d}\n'.format(
                v, i + 1, j + 1, k + 1, l + 1))

        i_1b, j_1b = np.triu_indices(n_orb)
        vals_1b = h1e[i_1b, j_1b]
        mask_1b = np.abs(vals_1b) > 1e-15
        for v, i, j in zip(vals_1b[mask_1b], i_1b[mask_1b], j_1b[mask_1b]):
            f.write('{:22.15E} {:3d} {:3d} {:3d} {:3d}\n'.format(
                v, i + 1, j + 1, 0, 0))

        f.write('{:22.15E} {:3d} {:3d} {:3d} {:3d}\n'.format(ecore, 0, 0, 0, 0))


def read(filename):
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
            
        
            if i + j + k + l == -4:
                ecore += float(a)
            
            elif k + l ==  -2:
                h1e[i, j] = float(a)
                h1e[j, i] = float(a)

            else:
                g2e[i, j, k, l] = float(a)
                g2e[k, l, i, j] = float(a)

    return n_sites, n_elec, ecore, h1e, g2e
