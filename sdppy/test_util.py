import numpy as np

from .generate import generate_test_corr

from .util import CorrelatorFlattener

import pytest

@pytest.mark.parametrize("use_mparray", [True, False])
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("cplx", [True, False])
def test_corr_flatten_corr(scalar, use_mparray, cplx):
    Nstate = 8
    Nt = 8

    batch_shape = (3,5)

    corr, _, _ = generate_test_corr(Nstate, Nt)
    corr = np.broadcast_to(corr, batch_shape + corr.shape)
    if scalar:
        corr = corr[...,:1,:1]
    if not use_mparray:
        corr = corr.astype(np.float64)
    if cplx:
        corr = corr + 1j
    r = corr.shape[-1]

    flattener = CorrelatorFlattener(Nt, r)
    corr_flat = flattener.flatten_corr(corr)
    corr2 = flattener.unflatten_corr(corr_flat)

    assert np.all(corr == corr2)

@pytest.mark.parametrize("use_mparray", [True, False])
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("cplx", [True, False])
def test_corr_flatten_cov(scalar, use_mparray, cplx):
    Nstate = 8
    Nt = 4

    batch_shape = (3,2)

    corr, _, _ = generate_test_corr(Nstate, Nt)
    if scalar:
        corr = corr[...,:1,:1]

    cov = corr[:,:,:,None,None,None] * corr[None,None,None,:,:,:]

    cov = np.broadcast_to(cov, batch_shape + cov.shape)
    if not use_mparray:
        cov = cov.astype(np.float64)
    if cplx:
        cov = cov + 1j
    r = cov.shape[-1]

    flattener = CorrelatorFlattener(Nt, r)
    cov_flat = flattener.flatten_cov(cov)
    cov2 = flattener.unflatten_cov(cov_flat)

    assert np.all(cov == cov2)
