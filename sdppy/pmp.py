from typing import List, Optional

import os
import json
import numpy as np
import tempfile
import subprocess
import pathlib

from mpmath import mp

from dataclasses import dataclass

from .mp_array import MPArray
from .io_utils import mpf_array_str
from .util import is_scalar, CorrelatorFlattener
from . import polynomials

def stack_along(arrs, dim):
    if dim == 0:
        return arrs
    return [stack_along(arrs_i, dim-1) for arrs_i in zip(*arrs)]

# Add together polynomials, constant coefficient first
def add_poly(p: MPArray, q: MPArray):
    assert len(p.shape) == 1
    assert len(q.shape) == 1

    # Need to match degrees so addition below goes through
    if len(p) > len(q):
        padding = MPArray(np.zeros(len(p) - len(q)))
        q = np.concatenate((q, padding))
    elif len(p) < len(q):
        padding = MPArray(np.zeros(len(q) - len(p)))
        p = np.concatenate((p, padding))

    return p + q

@dataclass
class PolynomialMatrix:
    polynomials: List[List[MPArray]]

    def as_array(self):
        return mpf_array_str(self.polynomials)

    def as_numpy(self):
        max_degree = max(max(len(x) for x in row)
                         for row in self.polynomials)
        def uniformize(p):
            p_out = MPArray(np.zeros(max_degree))
            p_out[:len(p)] = p
            return p_out

        return np.stack([
            np.stack([uniformize(p) for p in row])
            for row in self.polynomials
        ])

    def degree(self):
        return max(max(len(p)-1 for p in row)
                   for row in self.polynomials)

    def eval(self, x):
        res = [[np.polynomial.Polynomial(p.inner)(x) for p in row]
               for row in self.polynomials]
        return MPArray(np.array(res, dtype=object))

    def __mul__(self, c):
        assert is_scalar(c)
        return PolynomialMatrix([[c * x for x in row]
                                 for row in self.polynomials])

    def __rmul__(self, c):
        return self * c

    def __add__(self, M):
        if isinstance(M, int) and M == 0:
            return self

        assert isinstance(M, PolynomialMatrix)
        assert len(self.polynomials) == len(M.polynomials)
        result = []
        for row_left, row_right in zip(self.polynomials, M.polynomials):
            assert len(row_left) == len(row_right)
            result.append([add_poly(x, y) for x, y in zip(row_left, row_right)])
        return PolynomialMatrix(result)

    def __radd__(self, x):
        return self + x

@dataclass
class PositiveMatrixWithPrefactorArray:
    polynomial_matrices: List[PolynomialMatrix]

    def to_dict(self):
        return {
            "polynomials" : stack_along([M.as_array()
                                         for M in self.polynomial_matrices], 2)
        }

    def as_numpy(self):
        matrices_numpy = [M.as_numpy() for M in self.polynomial_matrices]
        max_degree = max(a.shape[-1] for a in matrices_numpy)

        def pad(M):
            padding_size = list(M.shape)
            padding_size[-1] = max_degree - padding_size[-1]
            if padding_size[-1] == 0:
                return M

            padding = MPArray(np.zeros(padding_size))
            return np.concatenate((M, padding), axis=-1)

        return np.stack([pad(M) for M in matrices_numpy])

    def degree(self):
        return max(M.degree() for M in self.polynomial_matrices)

@dataclass
class PolynomialMatrixProgram:
    objective: MPArray
    normalization: Optional[MPArray]
    matrices: List[PositiveMatrixWithPrefactorArray]
    left_bound: Optional[mp.mpf]
    right_bound: Optional[mp.mpf]

    def to_dict(self):
        result = {
            "objective": mpf_array_str(self.objective),
            "PositiveMatrixWithPrefactorArray" : [M.to_dict() for M in self.matrices]
        }

        if self.normalization is not None:
            result["normalization"] = mpf_array_str(self.normalization)

        return result

def pmp_to_sdp(program: PolynomialMatrixProgram, *,
               precision: int,
               verbose=True):
    from .sdp import read_sdp

    run_script = os.environ["SDPPY_SDPB_EXE"]
    workdir = pathlib.Path(tempfile.TemporaryDirectory().name)

    infile = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False)
    sdpfile = workdir / "sdp"

    json.dump(program.to_dict(), infile)
    infile.close()

    subprocess.run([run_script, "pmp2sdp",
                    f"--precision={precision}",
                    "--outputFormat=json",
                    f"--input={infile.name}",
                    f"--output={sdpfile}"], capture_output=not verbose)

    return read_sdp(sdpfile)

def make_bilinear_bases(x, degree, *, poly_basis_type, left_bound, right_bound):
    m = degree // 2
    basis_m = polynomials.make_poly_basis(poly_basis_type, m).polynomials
    basis_m_1 = polynomials.make_poly_basis(poly_basis_type, m-1).polynomials
    # References to theorems are Dette & Studden, "Matrix
    # measures, moment spaces and Favard’s theorem for the
    # interval [0, 1] and [0, \infty)"
    if left_bound is None and right_bound is None:
        bilinear_basis = MPArray(np.array(
            [[q_i(x_k) for x_k in x] for q_i in basis_m],
            dtype=object))

        bilinear_bases = [bilinear_basis]
    elif left_bound is not None and right_bound is None:
        # Interval = [a, infty), Theorem 5.1
        bilinear_basis1 = MPArray(np.array(
            [[q_i(x_k) for x_k in x] for q_i in basis_m],
            dtype=object))
        bilinear_basis2 = MPArray(np.array(
            [[np.sqrt(x_k - left_bound) * q_i(x_k) for x_k in x]
                for q_i in (basis_m if degree % 2 == 1 else basis_m_1)],
            dtype=object))

        bilinear_bases = [bilinear_basis1, bilinear_basis2]
    elif left_bound is not None and right_bound is not None:
        # Interval = [a, b], Theorem 2.5
        if degree % 2 == 0: # n = 2m
            bilinear_basis1 = MPArray(np.array(
                [[q_i(x_k) for x_k in x] for q_i in basis_m],
                dtype=object))
            bilinear_basis2 = MPArray(np.array(
                [[np.sqrt((x_k - left_bound) * (right_bound - x_k)) * q_i(x_k)
                for x_k in x] for q_i in basis_m_1],
                dtype=object))
        else: # n = 2m + 1
            bilinear_basis1 = MPArray(np.array(
                [[np.sqrt(x_k - left_bound) * q_i(x_k) for x_k in x]
                for q_i in basis_m],
                dtype=object))
            bilinear_basis2 = MPArray(np.array(
                [[np.sqrt(right_bound - x_k) * q_i(x_k) for x_k in x]
                for q_i in basis_m],
                dtype=object))

        bilinear_bases = [bilinear_basis1, bilinear_basis2]
    else:
        raise NotImplementedError(f"{left_bound=} {right_bound=}")

    return bilinear_bases

def pmp_to_sdp2(program: PolynomialMatrixProgram, *,
                poly_basis_type="hermite", verbose=True):
    from .sdp import LinearMatrixEquality, SDP, compute_dense_indices
    from .util import chebyshev_nodes

    left_bound = program.left_bound
    right_bound = program.right_bound

    equalities = []
    for matrix_pencil in program.matrices:
        P0 = matrix_pencil.polynomial_matrices[0]
        inner_block_size = len(P0.polynomials)
        indices = compute_dense_indices(inner_block_size)

        degree = matrix_pencil.degree()

        x = chebyshev_nodes(degree+1)
        if left_bound is not None and right_bound is None:
            x = (x + 1.0) / 2 + left_bound
            assert np.all(x >= left_bound)
        if left_bound is not None and right_bound is not None:
            x = (x + 1.0) / 2 * (right_bound - left_bound) + left_bound
            assert np.all(x >= left_bound)
            assert np.all(x <= right_bound)

        B = []
        for Pi in matrix_pencil.polynomial_matrices[1:]:
            # All polynomial matrices should be the same size
            assert len(Pi.polynomials) == inner_block_size
            Bi = -np.stack([Pi.eval(x_k).ravel()[indices] for x_k in x],
                           axis=1).reshape(-1)
            B.append(Bi)

        B = np.stack(B, axis=1)
        C = None
        c = np.stack([P0.eval(x_k).ravel()[indices] for x_k in x],
                     axis=1).reshape(-1)

        bilinear_bases = make_bilinear_bases(x, degree,
                                             poly_basis_type=poly_basis_type,
                                             left_bound=left_bound,
                                             right_bound=right_bound)

        equalities.append(LinearMatrixEquality(B, c, C,
                                               bilinear_bases,
                                               inner_block_size,
                                               indices))

    return SDP(program.objective[1:], equalities)


def make_pmp(corr, numer: MPArray, denom: MPArray, maximize=False, *,
             left_bound=None, right_bound=None,
             poly_basis_type="hermite"):
    Nt, r, r2 = corr.shape
    assert r == r2, f"Invalid correlator shape: {corr.shape=}"

    all_mat = []

    # Kernel K(x) = numer(x)/denom(x) e_0 e_0^\dag
    # Later will multiply through by denom(x), so only numer(x)
    # appears here
    mat = [[MPArray(np.zeros(1)) for _ in range(r)] for _ in range(r)]
    mat[0][0] = -numer if maximize else numer
    mat = PolynomialMatrix(mat)
    all_mat.append(mat)

    flattener = CorrelatorFlattener(Nt, r)
    poly_basis = polynomials.make_poly_basis(poly_basis_type, Nt - 1)

    objective = flattener.flatten_corr(
        poly_basis.coefficients_from_monomial(corr))
    objective = np.concatenate(([0.0], objective), axis=0)
    objective = MPArray(np.array(objective))

    denom = np.polynomial.Polynomial(denom)

    for t in range(Nt):
        basis_poly = -poly_basis.polynomials[t] * denom
        basis_poly = MPArray(basis_poly.coef)
        for a in range(r):
            for b in range(a+1):
                mat = [[MPArray(np.zeros(1)) for _ in range(r)] for _ in range(r)]
                if a == b:
                    mat[a][b] = basis_poly
                else:
                    mat[a][b] = 0.5 * basis_poly
                    mat[b][a] = 0.5 * basis_poly
                mat = PolynomialMatrix(mat)
                all_mat.append(mat)

    positive_matrix = PositiveMatrixWithPrefactorArray(all_mat)

    pmp = PolynomialMatrixProgram(
        objective=objective,
        normalization=None,
        matrices=[positive_matrix],
        left_bound=left_bound,
        right_bound=right_bound
    )

    return pmp

# The bounds should be in the form: (left_bound, right_bound, kernel_support)
# Where kernel_support indicates whether the kernel should be zeroed
# out in that region
def make_pmp_sdp(corr, numer: MPArray, denom: MPArray, maximize=False, *,
             bounds, poly_basis_type="hermite"):
    sdp = None
    for bound in bounds:
        left_bound, right_bound, kernel_support = bound
        if kernel_support:
            pmp = make_pmp(corr, numer, denom,
                           maximize=maximize, poly_basis_type=poly_basis_type,
                           left_bound=left_bound, right_bound=right_bound)
        else:
            zero = MPArray(np.array([0.0]))
            one = MPArray(np.array([1.0]))
            pmp = make_pmp(corr, zero, one,
                           maximize=maximize, poly_basis_type=poly_basis_type,
                           left_bound=left_bound, right_bound=right_bound)

        sdp_i = pmp_to_sdp2(pmp, poly_basis_type=poly_basis_type)
        if sdp is None:
            sdp = sdp_i
        else:
            sdp.equalities.extend(sdp_i.equalities)

    sdp._update_constants()
    return sdp

def make_cauchy_pmp(corr, omega, epsilon, maximize=False, *,
                    left_bound=None, right_bound=None,
                    poly_basis_type="hermite"):
    from .kernels import cauchy_kernel
    return make_pmp(corr, *cauchy_kernel(omega, epsilon),
                    maximize=maximize,
                    left_bound=left_bound,
                    right_bound=right_bound,
                    poly_basis_type=poly_basis_type)

def make_noise_LME(objective, cov_flat, *, maximize, sigma0):
    from .sdp import LinearMatrixEquality

    ## Constructing final linear matrix equality to impose chi^2 bound
    Ncorr, = objective.shape
    Gamma0 = MPArray(np.zeros((Ncorr + 1, Ncorr + 1)))
    Gamma0[:Ncorr,:Ncorr] = cov_flat
    Gamma0[:Ncorr,-1] = -objective
    Gamma0[-1,:Ncorr] = -objective
    Gamma0[-1,-1] = sigma0**2

    inner_block_size=Ncorr + 1
    c = MPArray(np.zeros(Ncorr))
    bilinear_bases = [MPArray(np.ones((1,1)))]
    B = MPArray(-np.eye(Ncorr, Ncorr)/2)

    # Compute indices
    M = np.zeros((inner_block_size, inner_block_size))
    M[:Ncorr,-1] = 1
    indices = np.flatnonzero(M)

    eq = LinearMatrixEquality(B, c, [-Gamma0], bilinear_bases, inner_block_size=Ncorr+1, indices=indices)

    return eq

def make_noisy_pmp_sdp(corr, cov, numer, denom, *, maximize=False,
                       bounds, sigma0=mp.mpf("1.0"), poly_basis_type="hermite"):
    Nt, r, r2 = corr.shape
    assert r == r2, f"Invalid correlator shape: {corr.shape=}"
    poly_basis = polynomials.make_poly_basis(poly_basis_type, Nt - 1)

    flattener = CorrelatorFlattener(Nt, r)
    objective = flattener.flatten_corr(
        poly_basis.coefficients_from_monomial(corr))
    objective = MPArray(np.array(objective))

    cov_flat = flattener.flatten_cov(poly_basis.covariance_from_monomial(cov))

    eq = make_noise_LME(objective, cov_flat, maximize=maximize, sigma0=sigma0)

    sdp = make_pmp_sdp(corr, numer, denom,
                       maximize=maximize, poly_basis_type=poly_basis_type,
                       bounds=bounds)

    sdp.equalities.append(eq)
    sdp.objective_vector = 0.0 * sdp.objective_vector
    sdp._update_constants()

    return sdp

def make_pick_pmp(corr, omega, numer: MPArray, denom: MPArray,
                  maximize=False, *,
                  left_bound=None, right_bound=None):
    Nt, r, r2 = corr.shape
    assert r == r2, f"Invalid correlator shape: {corr.shape=}"
    assert omega.shape == (Nt,)

    all_mat = []

    # 1/(E^2 + omega_l^2) for all l/t
    cauchy_denoms = [
        np.polynomial.Polynomial(np.array([omega[t]**2, 0.0, 1.0]))
        for t in range(Nt)
    ]
    E = np.polynomial.Polynomial([0, 1.0])

    def _excluded_denom_prod(t_exclude=None):
        result = np.polynomial.Polynomial([1.0])
        for t, p in enumerate(cauchy_denoms):
            if t == t_exclude:
                continue
            result = result  * p
        return result

    # Kernel K(x) = numer(x)/denom(x) e_0 e_0^\dag
    # Later will multiply through by denom(x), so only numer(x)
    # appears here. Also multiply through by \prod_l (E^2 + \omega_l^2)
    mat = [[MPArray(np.zeros(1)) for _ in range(r)] for _ in range(r)]
    mat00 = np.polynomial.Polynomial(numer) * _excluded_denom_prod()
    mat[0][0] = MPArray(mat00.coef)
    if maximize:
        mat[0][0] = -mat[0][0]
    mat = PolynomialMatrix(mat)
    all_mat.append(mat)

    flattener = CorrelatorFlattener(Nt, r)
    objective = flattener.flatten_corr(corr)
    objective = MPArray(np.array(objective))
    objective = np.concatenate(([0.0], objective.real, objective.imag), axis=0)

    denom = np.polynomial.Polynomial(denom)

    # PMPs are constructed using MPArray, not Polynomial, so we need
    # to convert
    _strip = lambda p: MPArray(p.coef)

    real_mat = []
    imag_mat = []

    for t in range(Nt):
        denom_prod = -_excluded_denom_prod(t) * denom
        for a in range(r):
            for b in range(a+1):
                mat = [[MPArray(np.zeros(1)) for _ in range(r)] for _ in range(r)]
                if a == b:
                    mat[a][b] = _strip(denom_prod * E)
                else:
                    mat[a][b] = _strip(0.5 * denom_prod * E)
                    mat[b][a] = _strip(0.5 * denom_prod * E)
                mat = PolynomialMatrix(mat)
                real_mat.append(mat)

                assert omega[t] != 0.0

                mat = [[MPArray(np.zeros(1)) for _ in range(r)] for _ in range(r)]
                if a == b:
                    mat[a][b] = _strip(denom_prod * omega[t])
                else:
                    mat[a][b] = _strip(0.5 * denom_prod * omega[t])
                    mat[b][a] = _strip(0.5 * denom_prod * omega[t])
                mat = PolynomialMatrix(mat)
                imag_mat.append(mat)

    all_mat = all_mat + real_mat + imag_mat

    positive_matrix = PositiveMatrixWithPrefactorArray(all_mat)
    objective = MPArray(np.array(objective))

    pmp = PolynomialMatrixProgram(
        objective=objective,
        normalization=None,
        matrices=[positive_matrix],
        left_bound=left_bound,
        right_bound=right_bound
    )

    return pmp

# The bounds should be in the form: (left_bound, right_bound, kernel_support)
# Where kernel_support indicates whether the kernel should be zeroed
# out in that region
def make_pick_pmp_sdp(corr, omega, numer: MPArray, denom: MPArray, *,
                      maximize=False, bounds):
    sdp = None
    for bound in bounds:
        left_bound, right_bound, kernel_support = bound
        if kernel_support:
            pmp = make_pick_pmp(corr, omega, numer, denom,
                                maximize=maximize,
                                left_bound=left_bound,
                                right_bound=right_bound)
        else:
            zero = MPArray(np.array([0.0]))
            one = MPArray(np.array([1.0]))
            pmp = make_pick_pmp(corr, omega, zero, one,
                                maximize=maximize,
                                left_bound=left_bound,
                                right_bound=right_bound)

        sdp_i = pmp_to_sdp2(pmp)
        if sdp is None:
            sdp = sdp_i
        else:
            sdp.equalities.extend(sdp_i.equalities)
    sdp._update_constants()
    return sdp

def make_noisy_pick_pmp_sdp(corr, omega, cov, numer, denom, *, maximize=False,
                            bounds, sigma0):
    from .util import covariance_real

    Nt, r, r2 = corr.shape
    assert r == r2, f"Invalid correlator shape: {corr.shape=}"

    flattener = CorrelatorFlattener(Nt, r)
    objective = flattener.flatten_corr(corr)
    objective = MPArray(np.array(objective))
    objective = np.concatenate((objective.real, objective.imag), axis=0)

    cov_flat = flattener.flatten_cov(cov)
    cov_flat = covariance_real(cov_flat)

    eq = make_noise_LME(objective, cov_flat, maximize=maximize, sigma0=sigma0)

    sdp = make_pick_pmp_sdp(corr, omega, numer, denom,
                            maximize=maximize, bounds=bounds)

    sdp.equalities.append(eq)
    sdp.objective_vector = 0.0 * sdp.objective_vector
    sdp._update_constants()

    return sdp

# If a PMP has odd degree, and requires support over the full real
# line, then the only way to satisfy p(x) >= 0 for all x is to force
# the leading coefficient of p(x) to vanish. This causes complications
# in the SDP derived from such a PMP, in particular since there ends
# up being a nontrivial linear combination of the matrices A_p that
# vanishes. This can be handled at the PMP level by eliminating one of
# the variables with a nontrivial leading coefficient. For instance,
# suppose that the PMP constraint has the form
#
#     y_1 M_1(x) + y_2 M_2(x) \geq 0  \forall x \in \R
#
# with matrices
#
#     M_1(x) = a_0 + a_1 x + ... + a_n x^n
#     M_2(x) = b_0 + b_1 x + ... + b_n x^n
#
# where n is odd, and neither a_n nor b_n vanish. Then for large |x|,
# the coefficient of x^n dominates, with the leading behavior described by
#
#     (a_n y_1 + b_n y_2) x^n >= 0
#
# since n is odd, this can only be satisfied at x \to \pm \infty if
# the coefficient vanishes, i.e. if
#
#     a_n y_1 + b_n y_2 = 0
#
# This can be used to remove either y_1 or y_2 as a variable,
# resulting in an equivalent PMP with one fewer variable.
# This function serves to perform that reduction, returning an
# equivalent PMP that has an even degree.
def reduce_unbounded_odd_pmp(pmp: PolynomialMatrixProgram):
    from .sdp import compute_dense_indices
    # Reduction only applies to PMPs on full real line
    assert pmp.left_bound is None
    assert pmp.right_bound is None

    # Might be able to relax these, but this simplifies things for now.
    assert len(pmp.matrices) == 1
    assert pmp.normalization is None

    degree = pmp.matrices[0].degree()
    assert degree % 2 == 1, \
        f"reduce_unbounded_odd_pmp with even degree ({degree=})"

    positive_poly = pmp.matrices[0].as_numpy()
    assert positive_poly.shape[-1] == degree + 1

    flat_indices = compute_dense_indices(positive_poly.shape[1])

    drop_indices = np.argmax(np.abs(positive_poly[..., -1]), axis=0)
    drop_indices = drop_indices.reshape(-1)[flat_indices]

    new_positive_poly = []
    objective = []
    for i in range(positive_poly.shape[0]):
        if i in drop_indices:
            continue
        poly_i = positive_poly[i]
        objective_i = pmp.objective[i]

        # Eliminate leading coefficient by linearly combining poly_i
        # with positive_poly[drop_j] for drop_j in drop_indices
        leading_coeffs = poly_i[:,:,-1].reshape(-1)[flat_indices]
        for j, drop_j in enumerate(drop_indices):
            # Note: there's an assumption here that the polynomials
            # we're dropping only have a single nonzero entry at the
            # highest degree, e.g. [[1,1/2],[1/2,1]] * x^n would be
            # prohibited.
            drop_leading_coeff = positive_poly[drop_j,...,-1].reshape(-1)[flat_indices][j]
            coeff_drop_j = -leading_coeffs[j] / drop_leading_coeff
            poly_i = poly_i + coeff_drop_j * positive_poly[drop_j]
            objective_i = objective_i + coeff_drop_j * pmp.objective[drop_j]

        assert np.allclose(poly_i[:,:,-1], 0.0)
        poly_i = poly_i[:,:,:-1]
        poly_i = [list(row) for row in poly_i]
        poly_i = PolynomialMatrix(poly_i)

        new_positive_poly.append(poly_i)
        objective.append(objective_i)

    new_positive_poly = PositiveMatrixWithPrefactorArray(new_positive_poly)
    objective = np.stack(objective)
    return PolynomialMatrixProgram(
        objective=objective,
        normalization=None,
        matrices=[new_positive_poly],
        left_bound=None,
        right_bound=None
    )

# Evaluates the polynomial corresponding to the dual variable Y. For a
# PMP on the full real line, this has the form
#
#    p(x) = \Tr(Y Q(x))
#
# where Q(x) is the bilinear basis. For PMPs on other intervals, the
# form is similar but there are two matrices Y_1 and Y_2, and p(x) is
# the sum of two different trace terms.
def eval_sum_of_squares_poly(pmp, Y, z, *, poly_basis_type, inexact):
    result = 0
    k = 0
    for matrix_pencil in pmp.matrices:
        degree = matrix_pencil.degree()
        P0 = matrix_pencil.polynomial_matrices[0]
        inner_block_size = len(P0.polynomials)

        basis = make_bilinear_bases([z], degree,
                                    poly_basis_type=poly_basis_type,
                                    left_bound=pmp.left_bound,
                                    right_bound=pmp.right_bound)
        ident_r = np.eye(inner_block_size, inner_block_size)
        for Q_z in basis:
            Q_z = np.kron(Q_z, ident_r)
            result = result + np.einsum("ba,bc,cd -> ad", np.conj(Q_z), Y.blocks[k], Q_z)
            k = k + 1
    if inexact:
        assert k == len(Y.blocks) - 1
    else:
        assert k == len(Y.blocks)
    return result

# Evaluates the polynomial corresponding to the dual variable Y. For a
# PMP on the full real line, this has the form
#
#    p(x) = M_0(x) + \sum_{j>1} y_j M_j(x)
#
# where Q(x) is the bilinear basis, and M_j(x) are the polynomial
# matrices that define the PMP.
def eval_coefficient_poly(pmp, y, z):
    result = 0
    y_aug = np.concat(([1.0], y))
    for matrix_pencil in pmp.matrices:
        pencil = matrix_pencil.as_numpy()
        z_n = z**np.arange(pencil.shape[-1])
        result = result + np.einsum("i,i...j,j->...", y_aug, pencil,z_n)
    return result
