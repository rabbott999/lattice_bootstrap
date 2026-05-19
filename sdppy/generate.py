import numpy as np
from mpmath import mp

from .mp_array import MPArray

def generate_test_corr(Nstate, Nt):
    # ## Create correlators
    energies = mp.mpf("0.1") * (MPArray(np.arange(float(Nstate))) + 1.0)
    Ztilde = MPArray(np.ones((Nstate,2)))
    Ztilde[1::2,1] = mp.mpf(1.0)/5 # Z_{k1}, k odd
    Ztilde[2::2,0] = -mp.mpf(1.0)/5 # Z_{k0}, k > 0 even
    Ztilde[0,0] = mp.mpf(1.0)/5 # Z_{00}

    Z_exact = Ztilde

    t = MPArray(np.arange(float(Nt)))

    corr = np.einsum("ka,kb,kt -> tab", np.conj(Z_exact), Z_exact,
                     np.exp(-energies[:,None] * t[None,:]))

    return corr, Z_exact, energies

def generate_test_pick_corr(Nstate, Nt):
    # ## Create correlators
    energies = mp.mpf("0.1") * (MPArray(np.arange(float(Nstate))) + 1.0)
    Ztilde = MPArray(np.ones((Nstate,2)))
    Ztilde[1::2,1] = mp.mpf(1.0)/5 # Z_{k1}, k odd
    Ztilde[2::2,0] = -mp.mpf(1.0)/5 # Z_{k0}, k > 0 even
    Ztilde[0,0] = mp.mpf(1.0)/5 # Z_{00}

    Z_exact = Ztilde

    omegas = 2 * mp.pi * MPArray(np.arange(float(Nt))) / Nt / 2
    omegas[0] = 1e-3 * omegas[1]

    corr = np.einsum("ka,kb,kt -> tab", np.conj(Z_exact), Z_exact,
                     1 / (energies[:,None] - 1j * omegas[None,:]))

    return corr, Z_exact, energies, omegas

def generate_test_cov0(corr, dt_cov, shrinkage):
    from .util import CorrelatorFlattener
    Nt, r, r2 = corr.shape
    assert r == r2

    flattener = CorrelatorFlattener(Nt, r)

    dt = np.abs(MPArray(np.arange(Nt)[:,None] - np.arange(Nt)[None,:]))

    cov0 = corr * np.conj(corr[:,:,:,None,None,None]) \
        * np.exp(-dt/dt_cov)[:,None,None,:,None,None]

    cov0_flat = flattener.flatten_cov(cov0)
    ident = np.eye(*cov0_flat.shape)
    cov0_flat = cov0_flat * shrinkage + (1 - shrinkage) * ident * cov0_flat

    return flattener.unflatten_cov(cov0_flat)
