import os
import numpy as np
import pathlib
import json

import pytest

from mpmath import mp

from .sdp import ReducedSDP
from .test_sdp import _read_test_sdp
from .pmp import reduce_unbounded_odd_pmp
from .util import tree_allclose

from .mp_array import MPArray


@pytest.mark.parametrize("name", ["test_sdp", "noisy_stieltjes_sdp"])
def test_sdp_solve(name):
    mp.dps = 100

    from .solve_sdp import solve_sdp, default_params
    file_path = pathlib.Path(os.path.abspath(__file__))
    root_path = file_path.parent.parent / "test_data"
    results_loc = root_path / name / "results.json"

    # Compute results
    sdp = _read_test_sdp(name)
    q, history = solve_sdp(sdp, params=default_params)
    results = {
        "primal_objective" : history["primal_objective"][-1],
        "dual_objective" : history["dual_objective"][-1]
    }

    with open(results_loc, 'r') as f:
        reference_results = json.load(f)

    for key, ref_value in reference_results.items():
        ref_value = mp.mpf(ref_value)
        value = results[key]

        assert mp.fabs(ref_value - value) < 1e-20


@pytest.mark.parametrize("poly_basis_type", ["hermite", "monomial"])
@pytest.mark.parametrize("scalar", [True, False])
def test_pmp_hamburger(scalar, poly_basis_type):
    mp.dps = 100

    from sdppy.pmp import make_cauchy_pmp, pmp_to_sdp2
    from .solve_sdp import solve_sdp, default_params
    from .hamburgers import MomentProblem
    from .generate import generate_test_corr

    ## Parameters
    Nstate = 8
    Nt = 3
    r = 2

    omega = mp.mpf("0.3")
    epsilon = mp.mpf("0.05")
    z = mp.exp(-omega) + 1j * epsilon

    ## Generate correlator
    corr, Z_exact, energies = generate_test_corr(Nstate, Nt)
    if scalar:
        corr = corr[:,:1,:1]
    r = corr.shape[-1]

    ## Solve PMP via SDP
    # Lower bound/minimum
    pmp = make_cauchy_pmp(corr, mp.exp(-omega), epsilon, maximize=False,
                   poly_basis_type=poly_basis_type)
    sdp = pmp_to_sdp2(pmp)
    sdp = ReducedSDP(sdp)

    q, history = solve_sdp(sdp, params=default_params, verbose=True)
    min_bound = history["primal_objective"][-1]

    # Upper bound/maximum
    pmp = make_cauchy_pmp(corr, mp.exp(-omega), epsilon, maximize=True)
    sdp = pmp_to_sdp2(pmp)
    sdp = ReducedSDP(sdp)

    q, history = solve_sdp(sdp, params=default_params, verbose=True)
    max_bound = -history["primal_objective"][-1]

    ## Solve via Hamburger moment problem
    moment_problem = MomentProblem(corr)
    e0 = np.zeros(r)
    e0[0] = 1.0

    center, radius = moment_problem.get_ball(z).projection(e0)

    assert np.abs(center.imag + radius - max_bound) < 1e-20
    assert np.abs(center.imag - radius - min_bound) < 1e-20

@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("maximize", [True, False])
def test_pmp_stieltjes(scalar, maximize):
    mp.dps = 100

    from sdppy.pmp import make_cauchy_pmp, pmp_to_sdp, pmp_to_sdp2
    from .solve_sdp import solve_sdp, default_params
    from .generate import generate_test_corr

    if "SDPPY_SDPB_EXE" not in os.environ:
        return pytest.skip("SDPB executable not found")

    ## Parameters
    Nstate = 4
    Nt = Nstate

    omega = mp.mpf("0.3")
    epsilon = mp.mpf("0.05")

    ## Generate correlator
    corr, Z_exact, energies = generate_test_corr(Nstate, Nt)
    if scalar:
        corr = corr[:,:1,:1]

    # Lower bound/minimum
    pmp = make_cauchy_pmp(corr, mp.exp(-omega), epsilon, maximize=maximize,
                          left_bound=0.0, poly_basis_type="monomial")
    sdp = pmp_to_sdp(pmp, precision=mp.prec)
    sdp = ReducedSDP(sdp)
    _, history = solve_sdp(sdp, params=default_params, verbose=True)
    min_bound = history["primal_objective"][-1]

    sdp2 = pmp_to_sdp2(pmp)
    _, history = solve_sdp(sdp2, params=default_params, verbose=True)
    min_bound2 = history["primal_objective"][-1]

    assert abs((min_bound - min_bound2)/min_bound) < 1e-30

@pytest.mark.parametrize("scalar", [True, False])
def test_pmp_nevanlinna(scalar):
    mp.dps = 100

    from sdppy.pmp import make_pick_pmp, pmp_to_sdp2
    from .solve_sdp import solve_sdp, default_params
    from .hamburgers import NevanlinnnaPickProblem
    from .generate import generate_test_pick_corr

    ## Parameters
    Nstate = 4
    Nt = Nstate//2
    r = 2

    omega = mp.mpf("0.3")
    epsilon = mp.mpf("0.05")
    z = omega + 1j * epsilon

    ## Generate correlator
    corr, Z_exact, energies, omegas = generate_test_pick_corr(Nstate, Nt)
    if scalar:
        corr = corr[:,:1,:1]
    r = corr.shape[-1]

    numer = MPArray(np.array([1.0*epsilon], dtype=object))
    denom = MPArray(np.array([omega**2 + epsilon**2, -2 * omega, 1.0],
                             dtype=object))

    ## Solve PMP via SDP
    # Lower bound/minimum
    pmp = make_pick_pmp(corr, omegas, numer, denom, maximize=False)
    pmp = reduce_unbounded_odd_pmp(pmp)
    sdp = pmp_to_sdp2(pmp)
    sdp = ReducedSDP(sdp)

    q, history = solve_sdp(sdp, params=default_params, verbose=True)
    min_bound = history["primal_objective"][-1]

    # Upper bound/maximum
    pmp = make_pick_pmp(corr, omegas, numer, denom, maximize=True)
    pmp = reduce_unbounded_odd_pmp(pmp)
    sdp = pmp_to_sdp2(pmp)
    sdp = ReducedSDP(sdp)

    q, history = solve_sdp(sdp, params=default_params, verbose=True)
    max_bound = -history["primal_objective"][-1]

    ## Solve via Nevalinna-Pick inteprolation
    NP_problem = NevanlinnnaPickProblem(1j * omegas, corr)
    e0 = np.zeros(r)
    e0[0] = 1.0

    center, radius = NP_problem.get_ball(z).projection(e0)

    assert np.abs(center.imag + radius - max_bound) < 1e-20
    assert np.abs(center.imag - radius - min_bound) < 1e-20

@pytest.mark.parametrize("poly_basis_type", ["monomial", "hermite"])
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("maximize", [False, True])
def test_noisy_pmp_reconstruct(scalar, poly_basis_type, maximize):
    mp.dps = 150

    from sdppy.pmp import make_noisy_pmp_sdp, make_pmp, pmp_to_sdp2
    from .solve_sdp import solve_sdp, default_params
    from .generate import generate_test_corr, generate_test_cov0
    from .kernels import cauchy_kernel
    from .util import CorrelatorFlattener
    from . import polynomials

    ## Parameters
    Nstate = 8
    Nt = 3

    omega = mp.mpf("0.3")
    epsilon = mp.mpf("0.05")
    sigma0 = mp.mpf("0.9")
    error_scale = mp.mpf("0.1")
    dt_cov = mp.mpf("1.3")
    shrinkage = mp.mpf("0.5")

    numer, denom = cauchy_kernel(mp.exp(-omega), epsilon)
    bounds = [
        (None, None, True) # (-\infty, \infty), kernel is supported
    ]

    ## Generate correlator
    corr, Z_exact, energies = generate_test_corr(Nstate, Nt)
    if scalar:
        corr = corr[:,:1,:1]
    r = corr.shape[-1]
    flattener = CorrelatorFlattener(Nt, r)

    cov0 = generate_test_cov0(corr, dt_cov, shrinkage)
    cov = error_scale * cov0

    ## Solve Noisy SDP
    sdp = make_noisy_pmp_sdp(corr, cov, numer, denom, maximize=maximize,
                             sigma0=sigma0, poly_basis_type=poly_basis_type,
                             bounds=bounds)
    sdp = ReducedSDP(sdp)
    q, history = solve_sdp(sdp, params=default_params, verbose=True)
    bound_noisy = history["dual_objective"][-1]


    ## Reconstruct correlator that provides the optimal bound
    poly_basis = polynomials.make_poly_basis(poly_basis_type, Nt - 1)
    cov = poly_basis.covariance_from_monomial(cov)

    (x, X, y, Y) = q
    Z = Y.blocks[-1]
    cov_flat = flattener.flatten_cov(cov)
    dcorr = -cov_flat @ (Z[:-1,-1] / Z[-1,-1])

    # Check that corelator difference is consistent w/ saturating chi^2 bound
    dcorr_norm = dcorr.T @ np.linalg.inv(cov_flat) @ dcorr
    cutoff = sigma0**2
    assert np.allclose(dcorr_norm, MPArray(cutoff))

    dcorr = poly_basis.coefficients_to_monomial(flattener.unflatten_corr(dcorr))
    corr_bound = corr + dcorr

    ## Solve exact SDP w/ corr_bound
    pmp_exact = make_pmp(corr_bound, numer, denom, maximize=maximize)
    sdp_exact = pmp_to_sdp2(pmp_exact)
    sdp_exact = ReducedSDP(sdp_exact)
    _, history_exact = solve_sdp(sdp_exact, params=default_params,
                                 verbose=True)
    bound_exact = history_exact["dual_objective"][-1]

    assert np.abs(bound_noisy - bound_exact) < 1e-20

@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("maximize", [False, True])
def test_noisy_pick_pmp_reconstruct(scalar, maximize):
    mp.dps = 150

    from sdppy.pmp import make_noisy_pick_pmp_sdp, make_pick_pmp_sdp
    from .solve_sdp import solve_sdp, default_params
    from .generate import generate_test_pick_corr, generate_test_cov0
    from .kernels import cauchy_kernel
    from .util import CorrelatorFlattener, covariance_real, complex_from_components

    ## Parameters
    Nstate = 8
    Nt = 2

    omega = mp.mpf("0.3")
    epsilon = mp.mpf("0.05")
    sigma0 = mp.mpf("0.9")
    error_scale = mp.mpf("0.1")
    dt_cov = mp.mpf("1.3")
    shrinkage = mp.mpf("0.5")

    numer, denom = cauchy_kernel(mp.exp(-omega), epsilon)
    # Note: can't use (-\infty, \infty), since that causes issues with
    # the odd-degree polynomial (see reduce_unbounded_odd_pmp)
    bounds = [
        (0.0, None, True) # (0, \infty), kernel is supported
    ]

    ## Generate correlator
    corr, Z_exact, energies, omega = generate_test_pick_corr(Nstate, Nt)
    if scalar:
        corr = corr[:,:1,:1]
    r = corr.shape[-1]
    flattener = CorrelatorFlattener(Nt, r)

    cov0 = generate_test_cov0(corr, dt_cov, shrinkage)
    cov = error_scale * cov0

    ## Solve Noisy SDP
    sdp = make_noisy_pick_pmp_sdp(corr, omega, cov, numer, denom,
                                  maximize=maximize,
                                  sigma0=sigma0, bounds=bounds)
    sdp = ReducedSDP(sdp)
    q, history = solve_sdp(sdp, params=default_params, verbose=True)
    bound_noisy = history["dual_objective"][-1]


    ## Reconstruct correlator that provides the optimal bound
    (x, X, y, Y) = q
    Z = Y.blocks[-1]
    cov_flat = covariance_real(flattener.flatten_cov(cov))
    dcorr = -cov_flat @ (Z[:-1,-1] / Z[-1,-1])
    dcorr = complex_from_components(dcorr)
    dcorr = flattener.unflatten_corr(dcorr)
    corr_bound = corr + dcorr

    ## Solve exact SDP w/ corr_bound
    sdp_exact = make_pick_pmp_sdp(corr_bound, omega, numer, denom,
                                  maximize=maximize, bounds=bounds)
    sdp_exact = ReducedSDP(sdp_exact)
    _, history_exact = solve_sdp(sdp_exact, params=default_params,
                                 verbose=True)
    bound_exact = history_exact["dual_objective"][-1]

    assert np.abs(bound_noisy - bound_exact) < 1e-20


@pytest.mark.parametrize("scalar", [True, False])
def test_reduced_sdp(scalar):
    mp.dps = 100

    from .pmp import make_pmp, pmp_to_sdp2
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

    numer = MPArray(np.array([1.0*epsilon], dtype=object))
    denom = MPArray(np.array([omega**2 + epsilon**2, -2 * omega, 1.0],
                             dtype=object))

    pmp = make_pmp(corr, numer, denom,
                   maximize=maximize,
                   left_bound=0.0, right_bound=1.0)
    sdp = pmp_to_sdp2(pmp, poly_basis_type=poly_basis_type)
    sdp_red = ReducedSDP(sdp)

    q, history = solve_sdp(sdp, default_params)
    q_red, history_red = solve_sdp(sdp_red, default_params)

    # Check that objectives are equal
    for key in ["primal_objective", "dual_objective"]:
        assert mp.fabs(history_red[key][-1] - history[key][-1]) < 1e-20

    q_recovered = sdp_red.recover_free_variables(q_red)
    assert tree_allclose(q, q_recovered)
