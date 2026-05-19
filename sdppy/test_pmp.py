import pytest

import numpy as np
from mpmath import mp

from .mp_array import MPArray

@pytest.mark.parametrize("scalar", [True, False])
def test_polynomial_reconstruct(scalar):
    mp.dps = 100

    from .pmp import make_pmp, pmp_to_sdp2
    from .pmp import eval_sum_of_squares_poly, eval_coefficient_poly
    from .solve_sdp import solve_sdp, default_params
    from .generate import generate_test_corr

    ## Parameters
    Nstate = 4
    Nt = Nstate//2
    r = 2
    maximize = False
    poly_basis_type = ("chebyshev", 0, 1)

    omega = mp.mpf("0.3")
    epsilon = mp.mpf("0.05")

    ## Generate correlator
    corr, Z_exact, energies = generate_test_corr(Nstate, Nt)
    if scalar:
        corr = corr[:,:1,:1]
    r = corr.shape[-1]

    numer = MPArray(np.array([1.0*epsilon], dtype=object))
    denom = MPArray(np.array([omega**2 + epsilon**2, -2 * omega, 1.0],
                             dtype=object))

    pmp = make_pmp(corr, numer, denom,
                   maximize=maximize,
                   left_bound=0.0, right_bound=1.0)
    sdp = pmp_to_sdp2(pmp, poly_basis_type=poly_basis_type)

    q, history = solve_sdp(sdp, default_params)
    x, X, y, Y = q

    z = mp.mpf("0.45")

    p0 = eval_sum_of_squares_poly(pmp, Y, z, poly_basis_type=poly_basis_type,
                                  inexact=False)
    p1 = eval_coefficient_poly(pmp, y, z)

    assert np.allclose(p0, p1)
