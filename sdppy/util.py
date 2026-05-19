import numpy as np

from mpmath import mp

def trace_product(a, b):
    from .sdp import BlockMatrix
    if isinstance(a, BlockMatrix):
        assert isinstance(b, BlockMatrix)
        return sum(trace_product(a_i, b_i)
                    for a_i, b_i in zip(a.blocks, b.blocks))
    # None in a BlockMatrix means 0
    if a is None or b is None:
        return 0
    return np.sum(a.T * b)

def cholesky_solve(L, b):
    from . import scipy_linalg_mp
    x = scipy_linalg_mp.solve_triangular(L, b, lower=True)
    x = scipy_linalg_mp.solve_triangular(L.T, x, lower=False)
    return x


def adjoint(x, axis1=-1, axis2=-2):
    return np.swapaxes(np.conj(x), axis1=axis1, axis2=axis2)

def is_scalar(x):
    if isinstance(x, (int, float, mp.mpf, mp.mpc)):
        return True
    return False

def chebyshev_nodes(n):
    from .mp_array import MPArray
    k = MPArray(np.arange(n))
    return np.flip(np.cos((mp.pi * (k + 0.5)) / n))

class CorrelatorFlattener:
    def __init__(self, Nt, r):
        self.Nt = Nt
        self.r = r

        corr_shape = (Nt,r,r)

        corr_index_map = np.zeros(corr_shape) + np.prod(corr_shape)
        k = 0
        for t in range(Nt):
            for a in range(r):
                for b in range(a+1):
                    corr_index_map[t, a, b] = k
                    k = k + 1

        corr_index_map = corr_index_map.reshape(-1)
        self.corr_indices = np.argsort(corr_index_map)[:k]
        self.corr_shape = corr_shape

    def flatten_corr(self, corr):
        assert corr.shape[-3:] == self.corr_shape

        Nt = self.Nt
        r = self.r

        corr = corr.reshape((*corr.shape[:-3], Nt*r*r))
        return corr[..., self.corr_indices]

    def unflatten_corr(self, corr_flat):
        from .mp_array import MPArray
        assert corr_flat.shape[-1] == len(self.corr_indices)

        Nt = self.Nt
        r = self.r

        batch_shape = corr_flat.shape[:-1]

        if isinstance(corr_flat, MPArray):
            corr = np.zeros((*batch_shape, Nt*r*r), dtype=object)
            corr = MPArray(corr, cplx=corr_flat.cplx)
        else:
            corr = np.zeros((*batch_shape, Nt*r*r), dtype=corr_flat.dtype)
        corr[..., self.corr_indices] = corr_flat
        corr = corr.reshape((*batch_shape, Nt, r, r))
        corr = corr + (1 - np.eye(r)) * np.swapaxes(corr, -1, -2)
        return corr

    def flatten_cov(self, cov):
        assert cov.shape[-6:] == self.corr_shape + self.corr_shape

        Nt = self.Nt
        r = self.r

        cov = cov.reshape((*cov.shape[:-6], Nt*r*r, Nt*r*r))
        cov = cov[..., self.corr_indices, :]
        cov = cov[..., :, self.corr_indices]
        return cov

    def unflatten_cov(self, cov_flat):
        from .mp_array import MPArray
        assert cov_flat.shape[-1] == len(self.corr_indices)
        assert cov_flat.shape[-2] == len(self.corr_indices)

        Nt = self.Nt
        r = self.r

        batch_shape = cov_flat.shape[:-2]

        if isinstance(cov_flat, MPArray):
            cov = np.zeros((*batch_shape, Nt*r*r, cov_flat.shape[-1]), dtype=object)
            cov = MPArray(cov, cplx=cov_flat.cplx)
        else:
            cov = np.zeros((*batch_shape, Nt*r*r, cov_flat.shape[-1]),
                           dtype=cov_flat.dtype)
        cov[..., self.corr_indices, :] = cov_flat
        cov = cov.reshape((*batch_shape, Nt, r, r, cov_flat.shape[-1]))
        # Fill in below diagonal
        cov = cov + (1 - np.eye(r))[...,None] * np.swapaxes(cov, -2, -3)
        return self.unflatten_corr(cov)

def covariance_real(cov):
    """
    Takes a covariance matrix 'cov' for a complex variable and returns
    the corresponding covariance matrix for the real and imaginary
    parts.

    This operation commutes with inverting, so either a covariance or
    inverse covariance matrix can be applied here.
    """
    cov_real = np.concatenate(
        [
            np.concatenate([cov.real, -cov.imag], axis=1),
            np.concatenate([cov.imag, cov.real], axis=1),
        ],
        axis=0,
    )

    return cov_real

def complex_from_components(xs):
    n = xs.shape[-1]
    assert n % 2 == 0
    xs = xs.reshape(*xs.shape[:-1], 2, n//2)
    ws = xs[...,0,:] + 1j * xs[...,1,:]
    return ws

def tree_allclose(a, b, rtol=1e-10, atol=1e-10):
    from .sdp import BlockMatrix

    if isinstance(a, BlockMatrix):
        assert isinstance(b, BlockMatrix)
        return tree_allclose(a.blocks, b.blocks, rtol=rtol, atol=atol)
    elif isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list))
        assert len(a) == len(b)
        return all(tree_allclose(a_i, b_i, rtol=rtol, atol=atol)
                   for a_i, b_i in zip(a, b))

    assert not isinstance(b, (tuple, list, BlockMatrix)), "Tree structure mismatch"

    return np.allclose(a, b, rtol=rtol, atol=atol)
