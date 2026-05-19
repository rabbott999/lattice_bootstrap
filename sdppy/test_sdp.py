import os
import pathlib
import tempfile

import numpy as np
import pytest
import functools
import operator

from .sdp import read_sdp, write_sdp, SDP, BlockMatrix, ReducedSDP
from .mp_array import MPArray

def _transpose(x):
    return np.swapaxes(x, axis1=-1, axis2=-2)

# Makes a random, symmetric block matrix that has the dimensions of
# the primal and dual matrices for the given SDP
def _make_random_block_matrix(sdp, rng):
    if isinstance(sdp, ReducedSDP):
        equalities = sdp.sdp.equalities
    else:
        assert isinstance(sdp, SDP)
        equalities = sdp.equalities
    M = []
    for eq in equalities:
        for bilinear_basis in eq.bilinear_bases:
            block_size = bilinear_basis.shape[0] * eq.inner_block_size
            M_b = rng.normal(size=(block_size, block_size))
            M.append((M_b + M_b.T)/2)
    return BlockMatrix(M)

def _read_test_sdp(name):
    from .sdp import read_sdp
    file_path = pathlib.Path(os.path.abspath(__file__))
    root_path = file_path.parent.parent / "test_data"
    return read_sdp(root_path / name)

# The tamplate for passing multiple values comes from
# https://github.com/pytest-dev/pytest/issues/1595
@pytest.fixture(params=["test_sdp", "hamburger_sdp",
                        "stieltjes_sdp",
                        "noisy_stieltjes_sdp"])
def sdp(request):
    return _read_test_sdp(request.param)

@pytest.mark.parametrize("reduced", [False, True])
def test_combine_A(sdp, reduced):
    seed = 0
    rng = np.random.default_rng(seed)

    if reduced:
        sdp = ReducedSDP(sdp)

    A = sdp.make_matrices()
    coeff = rng.normal(size=(len(A),))

    M = sdp.combine_A(coeff)
    M_naive = functools.reduce(operator.add,
        [c_i * A_i for c_i, A_i in zip(coeff, A)])

    for block, block_naive in zip(M.blocks, M_naive.blocks):
        assert np.allclose(block, block_naive)

@pytest.mark.parametrize("reduced", [False, True])
def test_tr_A_vec(sdp, reduced):
    seed = 0
    rng = np.random.default_rng(seed)

    if reduced:
        sdp = ReducedSDP(sdp)

    A = sdp.make_matrices()
    M = _make_random_block_matrix(sdp, rng)

    result = sdp.tr_A_vec(M)
    result_naive = MPArray(np.array([(A_i @ M).trace() for A_i in A]))

    assert np.allclose(result, result_naive)

@pytest.mark.parametrize("reduced", [False, True])
def test_schur_complement(sdp, reduced):
    from .util import trace_product
    seed = 0
    rng = np.random.default_rng(seed)

    if reduced:
        sdp = ReducedSDP(sdp)

    A = sdp.make_matrices()
    X_inv = _make_random_block_matrix(sdp, rng)
    Y = _make_random_block_matrix(sdp, rng)

    S = sdp.compute_schur_complement(X_inv, Y)
    AX = [A_p @ X_inv for A_p in A]
    AY = [A_q @ Y for A_q in A]
    S_naive = [[trace_product(AX_p, AY_q)
                for AY_q in AY] for AX_p in AX]
    S_naive = MPArray(np.array(S_naive))

    # ReducedSDP returns a full matrix (since it's dense), but SDP
    # returns a BlockMatrix since it preserves the block structure.
    if not reduced:
        S = S.as_matrix()

    assert np.allclose(S, S_naive)

def test_read_write_sdp(sdp):
    with tempfile.TemporaryDirectory() as d:
        workdir = pathlib.Path(d)
        path = workdir / "test_sdp"
        write_sdp(sdp, path)
        sdp2 = read_sdp(path)
        assert sdp == sdp2
