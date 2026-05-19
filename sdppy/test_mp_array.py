import numpy as np

from .mp_array import MPArray
from . import scipy_linalg_mp

import pytest

def test_mparray_concat():
    a = np.arange(3)
    a_mp = MPArray(a)

    b = np.concatenate((a_mp, a_mp))

    assert np.sum(b) == 2 * np.sum(a)

def test_mparray_sum():
    a = np.arange(15).reshape(3, 5)
    a_mp = MPArray(a)

    assert np.all(np.sum(a_mp, axis=1) == np.sum(a, axis=1))

def test_mparray_mul_float():
    a = np.arange(15).reshape(3, 5)
    a_mp = MPArray(a)

    b = 2.0 * a_mp
    assert np.sum(b) == 2 * np.sum(a)
    assert not b.cplx

def test_mparray_real_imag():
    a = np.arange(15).reshape(3, 5)
    a = a + 1j * (3 - a)

    a_mp = MPArray(a)

    assert np.all(a_mp.real == a.real)
    assert np.all(a_mp.imag == a.imag)

def test_mparray_eig():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2,3,4,4))
    a = a + 1j * rng.normal(size=(2,3,4,4))
    a_mp = MPArray(a)

    eigvals, eigvecs = np.linalg.eig(a)
    eigvals_mp, eigvecs_mp = np.linalg.eig(a_mp)

    def sort_eig(vals, vecs):
        order = np.argsort(vals.real + vals.imag, axis=-1)
        vals = np.take_along_axis(vals, order, axis=-1)
        vecs = np.take_along_axis(vecs, order[...,None,:], axis=-1)
        return vals, vecs
    eigvals, eigvecs = sort_eig(eigvals, eigvecs)
    eigvals_mp, eigvecs_mp = sort_eig(eigvals_mp, eigvecs_mp)

    assert np.allclose(MPArray(eigvals), eigvals_mp)
    assert np.allclose(np.abs(MPArray(eigvecs)), np.abs(eigvecs_mp))

def test_mparray_eigh():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2,3,4,4))
    a = a + 1j * rng.normal(size=(2,3,4,4))
    a = a + np.conj(np.swapaxes(a, -1, -2))
    a_mp = MPArray(a)

    eigvals, eigvecs = np.linalg.eigh(a)
    eigvals_mp, eigvecs_mp = np.linalg.eigh(a_mp)

    def sort_eig(vals, vecs):
        order = np.argsort(vals, axis=-1)
        vals = np.take_along_axis(vals, order, axis=-1)
        vecs = np.take_along_axis(vecs, order[...,None,:], axis=-1)
        return vals, vecs
    eigvals, eigvecs = sort_eig(eigvals, eigvecs)
    eigvals_mp, eigvecs_mp = sort_eig(eigvals_mp, eigvecs_mp)

    assert np.allclose(MPArray(eigvals), eigvals_mp)
    assert np.allclose(np.abs(MPArray(eigvecs)), np.abs(eigvecs_mp))

def test_mparray_svdvals():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4,4))
    a_mp = MPArray(a)

    svals = np.linalg.svdvals(a)
    svals_mp = np.linalg.svdvals(a_mp)

    assert np.allclose(MPArray(svals), svals_mp)

@pytest.mark.parametrize("batched", [False, True])
def test_mparray_solve(batched):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4,4))
    b = rng.normal(size=(4,4)) if batched else rng.normal(size=(4,))
    a_mp = MPArray(a)
    b_mp = MPArray(b)

    x = np.linalg.solve(a, b)
    x_mp = np.linalg.solve(a_mp, b_mp)

    assert np.allclose(MPArray(x), x_mp)

def test_mparray_lu():
    from mpmath import mp
    mp.dps = 200
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4,4))
    a_mp = MPArray(a)

    P, L, U = scipy_linalg_mp.lu(a_mp)

    a2_mp = P @ L @ U

    assert np.allclose(a2_mp, a_mp)

def test_mparray_lu_solve():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4,4))
    b = rng.normal(size=(4))
    a_mp = MPArray(a)
    b_mp = MPArray(b)

    a_LU = scipy_linalg_mp.lu_factor(a_mp)

    x = np.linalg.solve(a, b)
    x_mp = scipy_linalg_mp.lu_solve(a_LU, b_mp)

    assert np.allclose(MPArray(x), x_mp)

def test_mparray_cholesky():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4,4))
    a = a @ a.T
    a_mp = MPArray(a)

    L = np.linalg.cholesky(a)
    L_mp = np.linalg.cholesky(a_mp)

    assert np.allclose(MPArray(L), L_mp)

@pytest.mark.parametrize("mode", ['reduced', 'complete'])
def test_mparray_qr(mode):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6,4))
    a_mp = MPArray(a)

    q, r = np.linalg.qr(a, mode=mode)
    q_mp, r_mp = np.linalg.qr(a_mp, mode=mode)

    assert np.allclose(MPArray(q), q_mp)
    assert np.allclose(MPArray(r), r_mp)

@pytest.mark.parametrize("m", [4, 6, 8])
def test_mparray_lstsq(m):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6,m))
    b = rng.normal(size=(6,))
    a_mp = MPArray(a)
    b_mp = MPArray(b)

    x, resid, rank, s = np.linalg.lstsq(a, b)
    x_mp, resid_mp, rank_mp, s_mp = np.linalg.lstsq(a_mp, b_mp)

    assert rank == rank_mp
    assert np.allclose(MPArray(x), x_mp)
    assert np.allclose(MPArray(s), s_mp)
    if len(resid) > 0:
        assert np.allclose(MPArray(resid), resid_mp)
    else:
        assert len(resid_mp) == 0

def test_polyfit():
    from .util import chebyshev_nodes
    from .mp_utils import mp_allclose

    poly = np.polynomial.Polynomial(MPArray(np.array([1, 2, 3, 4])))
    degree = poly.degree()

    nodes = chebyshev_nodes(degree + 1)
    values = MPArray(np.array([poly(x_i) for x_i in nodes]))
    poly2 = np.polynomial.Polynomial(np.flip(np.polyfit(nodes, values, degree)))

    assert mp_allclose(poly.coef, poly2.coef)
