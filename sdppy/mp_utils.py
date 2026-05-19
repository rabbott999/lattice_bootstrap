import mpmath

from mpmath import mp

import numpy as np

def init_mpmath(digits):
    mp.dps = digits
    print(mp)

def mp_array(A):
    return np.array(list(map(mp.mpc, A.ravel())), dtype=object).reshape(A.shape)

def mp_real_array(A):
    return np.array(list(map(mp.mpf, A.ravel())), dtype=object).reshape(A.shape)

## Batching
# decorator to treat all but last n_dims as batch dimensions
# handles multiple returns, but only one input allowed
def batched_function(n_dims):
    def decorator(f):
        def batched_f(x, *args, **kwargs):
            batch = x.shape[:-n_dims]
            x = x.reshape(-1, *x.shape[-n_dims:]) # flatten batch dim
            fx = [f(b, *args, **kwargs) for b in x] # apply function to each batch element
            if isinstance(fx[0], tuple):
                # multiple returns
                assert all([isinstance(y, tuple) for y in fx]) # everything must be tuple
                lens = np.array(list(map(len,fx)))
                assert (lens == lens[0]).all(), lens # all same length
                stacked = []
                for y in zip(*fx): # iterate over each return
                    y = np.stack(y, axis=0)
                    y = y.reshape(*batch, *y.shape[1:]) # unflatten batch dims
                    stacked.append(y)
                return tuple(stacked)
            else:
                # scalar return
                fx = np.stack(fx, axis=0)
                fx = fx.reshape(*batch, *fx.shape[1:]) # unflatten batch dims
                return fx
        return batched_f
    return decorator

mp_conj = np.vectorize(lambda x: x.conjugate())
mp_log = np.vectorize(mp.log)
mp_exp = np.vectorize(mp.exp)
# etc
mp_sin = np.vectorize(mp.sin)
mp_cos = np.vectorize(mp.cos)
mp_arcsin = np.vectorize(mp.asin)
mp_arccos = np.vectorize(mp.acos)
mp_sqrt = np.vectorize(mp.sqrt)
mp_re = np.vectorize(mp.re)
mp_im = np.vectorize(mp.im)
mp_abs = np.vectorize(mp.fabs)
mp_arg = np.vectorize(mp.arg)
mp_sign = np.vectorize(mp.sign)
mp_isnan = np.vectorize(mp.isnan)
mp_isfinite = np.vectorize(mp.isfinite)
mp_complex = np.vectorize(mp.mpc)
mp_adjoint = lambda x: mp_conj(np.swapaxes(x, -1, -2))
mp_norm = lambda x: mp_sqrt(mp_re(np.sum(x * mp_conj(x))))

def mp_allclose(a, b, *, rtol=1e-8, atol=0):
    return np.all(mp_abs(a - b) <= (atol + rtol * mp_abs(b)))

def mp_logsumexp(a, axis=None, keepdims=False):
    a_max = np.max(a, axis=axis, keepdims=True)
    tmp = mp_exp(a - a_max)
    out = mp_log(np.sum(tmp, axis=axis, keepdims=keepdims))

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)

    out = out + a_max
    return out

def mp_inv(A):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    A = mp.matrix(A.tolist())
    Ainv = A**-1
    return np.array(Ainv.tolist())

@batched_function(2)
def mp_eig(A):
    from flint import acb_mat, ctx
    ctx.dps = mp.dps

    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0
    A = acb_mat(A.tolist())
    evals, evecs = A.eig(right=True, algorithm="approx")
    evals = np.array(evals)
    evecs = np.array(evecs.tolist())

    # UNIT-NORMALIZE
    evecs /= (evecs * evecs.conj()).sum(0)[None,:]**0.5

    return evals, evecs

@batched_function(2)
def mp_eigh(A):
    from flint import acb_mat, ctx
    ctx.dps = mp.dps

    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0
    A = acb_mat(A.tolist())
    evals, evecs = A.eig(right=True, algorithm="approx")
    evals = np.array([x.real for x in evals])
    evecs = np.array(evecs.tolist())

    # UNIT-NORMALIZE
    evecs /= (evecs * evecs.conj()).sum(0)[None,:]**0.5

    return evals, evecs

@batched_function(2)
def mp_eigvalsh(A):
    from flint import acb_mat, ctx
    ctx.dps = mp.dps

    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0

    A = acb_mat(A.tolist())
    evals = A.eig(algorithm="approx")
    evals = np.array([x.real for x in evals])

    return evals

def mp_expm(A, n):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    A = mp.matrix(A.tolist())
    An = A**n
    return np.array(An.tolist())

def mp_solve(A, b):
    assert len(A.shape) == 2

    if len(b.shape) == 2:
        # Note: the LU decomposition is automatically cached in
        # mp.lu_solve(), so it will be re-used here and we'll avoid
        # the inefficiency of recomputing the LU decomposition per column
        x = [mp_solve(A, b[:,k]) for k in range(b.shape[1])]
        return np.stack(x, axis=-1)

    assert len(b.shape) == 1

    A = mp.matrix(A.tolist())
    b = mp.matrix(b.tolist())

    x = mp.lu_solve(A, b)
    return np.array(x)

def mp_cholesky(A):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0

    tol = 100 * mp.eps**2

    A = mp.matrix(A.tolist())
    L = mp.cholesky(A, tol=tol)
    L = np.array(L.tolist())

    return L

def mp_qr(A, mode='full'):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0
    A = mp.matrix(A.tolist())
    q, r = mp.qr(A, mode=mode)
    q = np.array(q.tolist())
    r = np.array(r.tolist())

    return q, r

def mp_svd(A, compute_uv=False, full_matrices=False):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0
    A = mp.matrix(A.tolist())
    if compute_uv:
        U, S, V = mp.svd(A, compute_uv=True, full_matrices=full_matrices)
        U = np.array(U.tolist())
        S = np.array(S)
        V = np.array(V.tolist())

        return U, S, V
    else:
        S = mp.svd(A, compute_uv=False, full_matrices=full_matrices)
        S = np.array(S)
        return S

def mp_lu(A):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0
    A = mp.matrix(A.tolist())
    P, L, U = mp.lu(A)
    P = np.array(P.tolist())
    L = np.array(L.tolist())
    U = np.array(U.tolist())

    return P, L, U

def mp_lu_factor(A):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0
    A = mp.matrix(A.tolist())
    A, p = mp.LU_decomp(A)
    A = np.array(A.tolist())

    return A, p

def mp_lu_solve(A_LU, b):
    A, p = A_LU

    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0
    assert len(b.shape) == 1

    A = mp.matrix(A.tolist())
    b = mp.matrix(b.tolist())

    b = mp.L_solve(A, b, p)
    x = mp.U_solve(A, b)

    return np.array(x)

def mp_solve_lower(A, b):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0
    assert len(b.shape) == 1

    # For efficiency reasons, mp.L_solve assumes that L[i,i] = 1.0, so
    # we have to convert to that form before calling L_solve
    diag = np.diag(A)
    # Factor A = L D, with D diagonal
    # (LD)^{-1} = D^{-1} L^{-1}
    L = A / diag[None,:]

    L = mp.matrix(L.tolist())
    b = mp.matrix(b.tolist())

    x = mp.L_solve(L, b)
    x = np.array(x) / diag

    return np.array(x)

def mp_solve_upper(A, b):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    assert len(A) > 0
    assert len(b.shape) == 1

    A = mp.matrix(A.tolist())
    b = mp.matrix(b.tolist())

    x = mp.U_solve(A, b)

    return np.array(x)
