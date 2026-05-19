import numpy as np

from .mp_array import MPArray
from . import mp_array
from . import mp_utils

lu = mp_array.mp_lu
lu_factor = mp_array.mp_lu_factor
lu_solve = mp_array.mp_lu_solve

def solve_triangular(a, b, trans=0, lower=False,
                     unit_diagonal=False, overwrite_b=False,
                     check_finite=True):
    assert (not unit_diagonal) and (not overwrite_b) \
        and check_finite and trans == 0, \
        "Not implemented options for solve_triangular"

    assert len(a.shape) == 2
    if len(b.shape) == 2:
        x = [solve_triangular(a, b_i, lower=lower)
             for b_i in b.T]
        return np.stack(x, axis=1)

    a = a.inner
    b = b.inner

    if lower:
        x = mp_utils.mp_solve_lower(a, b)
    else:
        x = mp_utils.mp_solve_upper(a, b)

    return mp_array.MPArray(x)
