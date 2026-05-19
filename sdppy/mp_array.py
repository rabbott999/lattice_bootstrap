import flint
from mpmath import mp

import numpy as np
import scipy

from . import mp_utils

MPARRAY_HANDLED_FUNCTIONS = {}
MPARRAY_HANDLED_UFUNCS = {}
# Functions where calling
MPARRAY_PASSTHROUGH_FUNCTIONS = [
    np.concatenate,
    np.sort,
    np.argsort,
    np.take_along_axis,
    np.swapaxes,
    np.einsum,
    np.stack,
    np.diag,
    np.dot,
    np.min,
    np.max,
    np.flip,
    np.roll,
    np.kron,
    np.trace,
    np.outer,
    np.argmin,
    np.argmax,
    np.shape,
    np.clip,
    np.where,
    np.broadcast_to
]
MPARRAY_PASSTHROUGH_UFUNCS = [
    np.equal,
    np.not_equal,
    np.less,
    np.less_equal,
    np.greater,
    np.greater_equal,
    np.power,
]

def _unwrap_pytree(x):
    if isinstance(x, MPArray):
        return x.inner
    elif isinstance(x, tuple):
        return tuple(_unwrap_pytree(y) for y in x)
    elif isinstance(x, list):
        return [_unwrap_pytree(y) for y in x]
    return x

class MPArray:

    def __init__(self, arr, cplx=None):
        if isinstance(arr, MPArray):
            self.inner = arr.inner
            self.cplx = arr.cplx
            return

        if isinstance(arr, (mp.mpf, mp.mpc)):
            self.inner = np.array(arr, dtype=object)
            self.cplx = isinstance(arr, mp.mpc)
            return

        assert arr.dtype is not object, "Re-wrapping MPArray"

        if cplx is None:
            cplx = not np.isrealobj(arr)
            if all(isinstance(x, mp.mpf) for x in arr.ravel()):
                cplx = False
            elif any(isinstance(x, mp.mpc) for x in arr.ravel()):
                cplx = True
            elif any(isinstance(x, flint.acb) for x in arr.ravel()):
                cplx = True

        if arr.dtype in [np.int32, np.int64]:
            arr = arr.astype(np.float64)

        self.cplx = cplx

        if cplx:
            self.inner = mp_utils.mp_array(arr)
        else:
            self.inner = mp_utils.mp_real_array(arr)

    def __str__(self):
        return f"MPArray({str(self.inner)}, cplx={self.cplx})"

    def __repr__(self):
        return f"MPArray(arr={repr(self.inner)}, cplx={self.cplx})"

    def __len__(self):
        return len(self.inner)

    def __sub__(self, other):
        if isinstance(other, MPArray):
            return MPArray(self.inner - other.inner)
        return MPArray(self.inner - other)

    def __rsub__(self, other):
        return MPArray(other - self.inner)

    def __mul__(self, other):
        if isinstance(other, MPArray):
            return MPArray(self.inner * other.inner)
        return MPArray(self.inner * other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, MPArray):
            return MPArray(self.inner + other.inner)
        return MPArray(self.inner + other)

    def __matmul__(self, other):
        if isinstance(other, MPArray):
            return MPArray(self.inner @ other.inner)
        return MPArray(self.inner @ other)

    def __truediv__(self, other):
        if isinstance(other, MPArray):
            return MPArray(self.inner / other.inner)
        return MPArray(self.inner / other)

    def __rtruediv__(self, other):
        return MPArray(other / self.inner)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return MPArray(-self.inner, cplx=self.cplx)

    def __pow__(self, pow):
        return np.power(self, pow)

    def __lt__(self, other):
        return self.inner < other

    def __le__(self, other):
        return self.inner <= other

    def __gt__(self, other):
        return self.inner > other

    def __ge__(self, other):
        return self.inner >= other

    def __eq__(self, other):
        return self.inner == other

    def __ne__(self, other):
        return self.inner != other

    def __getitem__(self, key):
        result = self.inner[key]
        if isinstance(result, (mp.mpc, mp.mpf)):
            return result
        return MPArray(result, cplx=self.cplx)

    def __setitem__(self, key, value):
        if isinstance(value, MPArray):
            value = value.inner
        self.inner[key] = value

    @property
    def T(self):
        return MPArray(self.inner.T, cplx=self.cplx)

    def ravel(self):
        return MPArray(self.inner.ravel(), cplx=self.cplx)

    def reshape(self, *args):
        return MPArray(self.inner.reshape(*args), cplx=self.cplx)

    @property
    def real(self):
        return np.real(self)

    @property
    def imag(self):
        return np.imag(self)

    @property
    def shape(self):
        return self.inner.shape

    def astype(self, dtype):
        return self.inner.astype(dtype)

    ## Copied and modified from https://numpy.org/doc/stable/reference/arrays.classes.html
    def __array_function__(self, func, types, args, kwargs):
        if func in MPARRAY_PASSTHROUGH_FUNCTIONS:
            return self._handle_passthrough(func, args, kwargs)
        if func not in MPARRAY_HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MPArray objects
        # if not all(issubclass(t, MPArray) for t in types):
        #     return NotImplemented
        return MPARRAY_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in MPARRAY_PASSTHROUGH_UFUNCS:
            return self._handle_passthrough(ufunc, inputs, kwargs)
        if method != "__call__":
            return NotImplemented
        if ufunc not in MPARRAY_HANDLED_UFUNCS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MPArray objects
        # if not all(issubclass(t, MPArray) for t in types):
        #     return NotImplemented
        return MPARRAY_HANDLED_UFUNCS[ufunc](*inputs, **kwargs)

    def _handle_passthrough(self, func, args, kwargs):
        args = tuple(_unwrap_pytree(x) for x in args)
        result = func(*args, **kwargs)
        if hasattr(result, "dtype") and result.dtype == object:
            return MPArray(result)
        if isinstance(result, (mp.mpf, mp.mpc)):
            return MPArray(result)
        return result

def implements(numpy_function):
    """Register an __array_function__ implementation for MPArray objects."""
    def decorator(func):
        MPARRAY_HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

def implements_ufunc(numpy_function):
    """Register an __array_function__ implementation for MPArray objects."""
    def decorator(func):
        MPARRAY_HANDLED_UFUNCS[numpy_function] = func
        return func
    return decorator

@implements(np.sum)
def mp_sum(x, axis=None, dtype=None, out=None):
    assert out is None, "Output arrays not supported for MPArray"
    res = np.sum(x.inner, axis=axis, dtype=dtype)
    if hasattr(res, 'dtype'):
        return MPArray(res)
    return res

@implements(np.allclose)
def mp_allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    result = (np.less_equal(np.abs(x-y), atol + rtol * np.abs(y))
              & np.isfinite(y)
              | (x == y))
    return np.all(result)

@implements_ufunc(np.isfinite)
def mp_isfinite(x, *args, **kwargs):
    return np.vectorize(mp.isfinite)(x.inner)

@implements_ufunc(np.abs)
def mp_abs(x, *args, **kwargs):
    return MPArray(mp_utils.mp_abs(x.inner), cplx=False)

@implements_ufunc(np.sqrt)
def mp_sqrt(x, *args, **kwargs):
    return MPArray(mp_utils.mp_sqrt(x.inner))

@implements_ufunc(np.exp)
def mp_exp(x, *args, **kwargs):
    return MPArray(mp_utils.mp_exp(x.inner))

@implements_ufunc(np.log)
def mp_log(x, *args, **kwargs):
    return MPArray(mp_utils.mp_log(x.inner))

@implements_ufunc(np.conjugate)
def mp_conjugate(x, *args, **kwargs):
    return MPArray(mp_utils.mp_conj(x.inner))

@implements_ufunc(np.add)
def mp_add(x, y, *args, **kwargs):
    if isinstance(x, MPArray):
        x = x.inner
    if isinstance(y, MPArray):
        y = y.inner
    return MPArray(x + y)

@implements_ufunc(np.subtract)
def mp_sub(x, y, *args, **kwargs):
    if isinstance(x, MPArray):
        x = x.inner
    if isinstance(y, MPArray):
        y = y.inner
    return MPArray(x - y)

@implements_ufunc(np.multiply)
def mp_multiply(x, y, *args, **kwargs):
    if isinstance(x, MPArray):
        x = x.inner
    if isinstance(y, MPArray):
        y = y.inner
    return MPArray(x * y)

@implements_ufunc(np.sign)
def mp_sign(x):
    return MPArray(mp_utils.mp_sign(x.inner))

@implements(np.real)
def mp_real(x, *args, **kwargs):
    return MPArray(mp_utils.mp_re(x.inner), cplx=False)

@implements(np.imag)
def mp_imag(x, *args, **kwargs):
    return MPArray(mp_utils.mp_im(x.inner), cplx=False)

@implements(np.linalg.norm)
def mp_norm(x, *args, **kwargs):
    return mp_utils.mp_norm(x.inner)

@implements_ufunc(np.less_equal)
def mp_lessequal(x, y):
    if x.cplx or y.cplx:
        raise ValueError("No ordering relation defined for complex numbers")
    return x.inner <= y.inner

@implements(np.linalg.eig)
def mp_eig(a):
    eigvals, eigvecs = mp_utils.mp_eig(a.inner)
    return MPArray(eigvals), MPArray(eigvecs)

@implements(np.linalg.eigvals)
def mp_eigvals(a):
    eigvals, _ = mp_utils.mp_eig(a.inner)
    return MPArray(eigvals)

@implements(np.linalg.eigh)
def mp_eigh(a):
    eigvals, eigvecs = mp_utils.mp_eigh(a.inner)
    return MPArray(eigvals), MPArray(eigvecs)

@implements(np.linalg.solve)
def mp_solve(A, b):
    x = mp_utils.mp_solve(A.inner, b.inner)
    return MPArray(x)

@implements(np.linalg.eigvalsh)
def mp_eigvalsh(a):
    eigvals = mp_utils.mp_eigvalsh(a.inner)
    return MPArray(eigvals)

@implements(np.linalg.svdvals)
def mp_svdvals(a):
    svals = mp_utils.mp_svd(a.inner, compute_uv=False)
    return MPArray(svals)

@implements(np.linalg.inv)
def mp_inv(a):
    return MPArray(mp_utils.mp_inv(a.inner))

@implements(np.linalg.cholesky)
def mp_cholesky(a):
    L = mp_utils.mp_cholesky(a.inner)
    return MPArray(L)

@implements(np.linalg.qr)
def mp_qr(a, mode='reduced'):
    if mode == 'reduced':
        mode = 'skinny'
        assert a.shape[0] >= a.shape[1]
    elif mode == 'complete':
        mode = 'full'
    else:
        raise NotImplementedError(f"np.linalg.qr(MPArray, mode={mode})")
    q, r = mp_utils.mp_qr(a.inner, mode=mode)
    return MPArray(q), MPArray(r)

@implements(np.linalg.svd)
def mp_svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    assert not hermitian, "svd(hermitian=True) Not implemented"

    if compute_uv:
        u, s, vt = mp_utils.mp_svd(a.inner, full_matrices=full_matrices,
                                   compute_uv=True)
        return MPArray(u), MPArray(s), MPArray(vt)
    else:
        s = mp_utils.mp_svd(a.inner, full_matrices=full_matrices,
                            compute_uv=False)
        return MPArray(s)

@implements(np.linalg.lstsq)
def mp_lstsq(a, b, rcond=None):
    assert len(a.shape) == 2
    assert len(b.shape) == 1
    if rcond is None:
        rcond = mp.eps * max(*a.shape)

    u, s, vt = np.linalg.svd(a)
    rank = np.argmax(s < rcond * s[0])
    if np.all(s > rcond * s[0]):
        rank = len(s)

    _adjoint = lambda x: np.swapaxes(np.conj(x), -1, -2)

    sinv = np.concatenate((1/s[:rank], 0*s[rank:]))
    x = _adjoint(vt)[:,:rank] @ (sinv * (_adjoint(u) @ b)[:len(s)])

    if rank < a.shape[1] or a.shape[0] <= a.shape[1]:
        resid = np.array([])
    else:
        resid = np.linalg.norm(a @ x - b)**2
        resid = MPArray(np.array([resid]))

    return x, resid, rank, s

# Note: scipy does not implement __array_function__, so we can't
# override scipy functionality. Seems like this might be a deliberate
# choice (see https://github.com/scipy/scipy/issues/18286), but the
# roadmap mentions support, so I'm not sure if this will ever be
# implemented. For now, scipy functions are in scipy_linalg_mp.py
#@implements(scipy.linalg.lu)
def mp_lu(a, permute_l=False, overwrite_a=False,
          check_finite=True, p_indices=False):
    assert (not permute_l) and (not overwrite_a) \
        and check_finite and (not p_indices), \
        "Not implemented options for scipy.linalg.lu"
    P, L, U = mp_utils.mp_lu(a.inner)
    # Note: convention difference here P -> P.T
    # mpmath uses the convention A = P^T L U
    # while scipy chooses A = P L U, so we have to convert
    return MPArray(P.T), MPArray(L), MPArray(U)

# Note: see comment above w/r/t __array_function__
#@implements(scipy.linalg.lu_factor)
def mp_lu_factor(a, overwrite_a=False, check_finite=True):
    assert (not overwrite_a) and check_finite, \
        "Not implemented options for scipy.linalg.lu_factor"
    A, p = mp_utils.mp_lu_factor(a.inner)
    return MPArray(A), p

# Note: see comment above w/r/t __array_function__
#@implements(scipy.linalg.lu_factor)
def mp_lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    assert (not overwrite_b) and check_finite and trans == 0, \
        "Not implemented options for scipy.linalg.lu_factor"
    lu, piv = lu_and_piv
    x = mp_utils.mp_lu_solve((lu.inner, piv), b.inner)
    return MPArray(x)

@implements_ufunc(np.cos)
def mp_cos(a):
    cos_a = mp_utils.mp_cos(a.inner)
    return MPArray(cos_a)

@implements_ufunc(np.sin)
def mp_sin(a):
    sin_a = mp_utils.mp_sin(a.inner)
    return MPArray(sin_a)

@implements_ufunc(np.arcsin)
def mp_arcsin(a):
    arcsin_a = mp_utils.mp_arcsin(a.inner)
    return MPArray(arcsin_a)

@implements_ufunc(np.arccos)
def mp_arccos(a):
    arccos_a = mp_utils.mp_arccos(a.inner)
    return MPArray(arccos_a)

@implements(np.result_type)
def mp_result_type(*arrays_and_types):
    return np.dtype('object')

@implements(np.polyfit)
def mp_polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    assert rcond is None and not full and w is None and not cov, "Not implemented"
    vander = x[:,None]**np.arange(deg+1)
    coeff = np.linalg.lstsq(vander, y)[0]
    return np.flip(coeff)
