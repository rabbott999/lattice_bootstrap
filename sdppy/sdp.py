from typing import List, Optional

import os
import re
import numpy as np
import json
from mpmath import mp
import itertools
import pathlib

from .mp_array import MPArray
from .io_utils import mpf_array_str, file_exists
from .util import adjoint, cholesky_solve, trace_product
from . import scipy_linalg_mp

from dataclasses import dataclass

def _parse_list(arr):
    if isinstance(arr, list):
        return [_parse_list(x) for x in arr]
    return mp.mpf(arr)

def _parse_array(arr):
    return MPArray(np.array(_parse_list(arr)))

# Compute indices to flatten symmetric matrix into a list
def compute_dense_indices(block_size):
    M = np.zeros((block_size, block_size))
    for r, s in itertools.product(range(block_size),
                                    range(block_size)):
        if r <= s:
            M[r, s] = 1
    indices = np.flatnonzero(M)
    return indices

@dataclass
class LinearMatrixEquality:
    B: MPArray
    c: MPArray
    C: Optional[MPArray]
    bilinear_bases: List[MPArray]
    # inner_block_size represents the size of the individual matrices
    # in the PMP; SDPB uses the notation "m"
    inner_block_size: int
    # Stores the values of (r, s) that enter into the equality
    # If M is an $m \times m$ block, then M.ravel()[indices] will
    # extract the relevant data.
    # For SDPB, indices always extracts every upper/lower diagonal
    # element, but we'll have cases where we only want a subset of the
    # equalities.
    indices: np.ndarray

    def __eq__(self, other):
        if self.inner_block_size != other.inner_block_size:
            return False
        if np.any(self.indices != other.indices):
            return False
        if np.any(self.B != other.B):
            return False
        if np.any(self.c != other.c):
            return False

        if self.C is None:
            if other.C is not None:
                return False
        else:
            if other.C is None:
                return False
            if np.any(self.C != other.C):
                return False

        if len(self.bilinear_bases) != len(other.bilinear_bases):
            return False
        for x, y in zip(self.bilinear_bases, other.bilinear_bases):
            if np.any(x != y):
                return False

        return True

    @staticmethod
    def from_json(fp, inner_block_size):
        data = json.load(fp)
        B = _parse_array(data["B"])
        c = _parse_array(data["c"])
        if "C" in data:
            C = _parse_array(data["C"])
        else:
            C = None
        if "bilinear_bases" in data:
            bilinear_bases = [_parse_array(x) for x in data["bilinear_bases"]]
        else:
            assert "bilinear_bases_even" in data
            assert "bilinear_bases_odd" in data
            bilinear_bases = [
                _parse_array(data["bilinear_bases_even"]),
                _parse_array(data["bilinear_bases_odd"])
            ]

        if "indices" in data:
            indices = np.array(data["indices"]).astype(int)
        else:
            indices = compute_dense_indices(inner_block_size)

        return LinearMatrixEquality(B, c, C, bilinear_bases,
                                    inner_block_size, indices)

    def to_json_dict(self):
        result = {
            "B": mpf_array_str(self.B),
            "c": mpf_array_str(self.c),
            "bilinear_bases": mpf_array_str(self.bilinear_bases),
            "indices": [int(x) for x in self.indices],
        }

        if self.C is not None:
            result.update(C=mpf_array_str(self.C))

        return result

@dataclass
class BlockMatrix:
    blocks: List[MPArray]

    def as_matrix(self):
        size = sum(M_b.shape[0] for M_b in self.blocks)
        M_full = MPArray(np.zeros((size, size)))
        i = 0
        for M_b in self.blocks:
            block_size = M_b.shape[0]
            M_full[i:i+block_size,i:i+block_size] = M_b
            i = i + block_size
        return M_full

    def __add__(self, other):
        assert isinstance(other, BlockMatrix)
        assert len(self.blocks) == len(other.blocks)
        def _add(a, b):
            # None = 0
            if a is None:
                return b
            if b is None:
                return a
            return a + b
        return BlockMatrix([_add(a, b) for a, b in zip(self.blocks, other.blocks)])

    def __sub__(self, other):
        def _sub(a, b):
            # None = 0
            if a is None:
                return -b
            if b is None:
                return a
            return a - b
        assert isinstance(other, BlockMatrix)
        assert len(self.blocks) == len(other.blocks)
        return BlockMatrix([_sub(a, b) for a, b in zip(self.blocks, other.blocks)])

    def __matmul__(self, other):
        def _matmul(a, b):
            # None = 0
            if a is None or b is None:
                return None
            return a @ b
        assert isinstance(other, BlockMatrix)
        assert len(self.blocks) == len(other.blocks)
        return BlockMatrix([_matmul(a, b) for a, b in zip(self.blocks, other.blocks)])

    def __mul__(self, other):
        assert isinstance(other, (mp.mpf, float, np.float64,
                                  np.float32))
        def _scale(M):
            if M is None:
                return None
            return other * M
        return BlockMatrix([_scale(block) for block in self.blocks])

    def __rmul__(self, other):
        return self * other

    def trace(self):
        return sum(np.trace(M) for M in self.blocks
                   if M is not None)[()]

    def inverse(self):
        return BlockMatrix([np.linalg.inv(block)
                            for block in self.blocks])

    @property
    def T(self):
        return BlockMatrix([block.T for block in self.blocks])

    def cholesky(self):
        return BlockMatrix([np.linalg.cholesky(block)
                            for block in self.blocks])

    def astype(self, dtype):
        return BlockMatrix([block.astype(dtype)
                            for block in self.blocks])

# Given a tensor of shape (n, n), reshapes to (n/k, k, n/k, k), where
# k = inner_block_size.
# Useful for making the tensor product structure of a matrix manifest
def _unravel_block(M, inner_block_size):
    block_size = M.shape[-1]
    assert block_size % inner_block_size == 0
    outer_block_size = block_size // inner_block_size
    shape = (outer_block_size, inner_block_size,
             outer_block_size, inner_block_size)
    return M.reshape(shape)

class SDP:
    objective_vector: MPArray

    equalities: List[LinearMatrixEquality]

    def __init__(self, objective_vector, equalities):
        self.objective_vector = objective_vector
        self.equalities = equalities

        self._update_constants()


    def _update_constants(self):
        self.B = np.concatenate([eq.B for eq in self.equalities], axis=0)
        self.c = np.concatenate([eq.c for eq in self.equalities], axis=0)
        self.b = self.objective_vector
        C = []
        for eq in self.equalities:
            if eq.C is not None:
                C.extend(eq.C)
            else:
                C.extend([None for _ in eq.bilinear_bases])
        self.C = BlockMatrix(C)

    def num_constraints(self):
        result = 0
        for eq in self.equalities:
            num_point = eq.bilinear_bases[0].shape[1]
            result = result + num_point * len(eq.indices)
        return result

    def __eq__(self, other):
        if np.any(self.objective_vector != other.objective_vector):
            return False
        if len(self.equalities) != len(other.equalities):
            return False
        for eq1, eq2 in zip(self.equalities, other.equalities):
            if eq1 != eq2:
                return False
        return True

    def initialize(self, Omega_P, Omega_D):
        X, Y = [], []
        for eq in self.equalities:
            for bilinear_basis in eq.bilinear_bases:
                block_size = bilinear_basis.shape[0] * eq.inner_block_size
                X_b = Omega_P * MPArray(np.eye(block_size, block_size))
                Y_b = Omega_D * MPArray(np.eye(block_size, block_size))
                X.append(X_b)
                Y.append(Y_b)

        X = BlockMatrix(X)
        Y = BlockMatrix(Y)

        x = MPArray(np.zeros(self.num_constraints()))
        y = MPArray(np.zeros(self.objective_vector.shape))

        return (x, X, y, Y)

    def primal_objective(self, q):
        x, X, y, Y = q
        return np.dot(x, self.c)[()]

    def dual_objective(self, q):
        x, X, y, Y = q
        return np.dot(y, self.b)[()] + trace_product(self.C, Y)

    # Computes Tr(A_p M), returned as a vector over p
    def tr_A_vec(self, M):
        b = 0
        result = []
        for equality in self.equalities:
            tr_p = []
            for bilinear_basis in equality.bilinear_bases:
                M_b = M.blocks[b]
                M_b = _unravel_block(M_b, equality.inner_block_size)

                traces = np.einsum("mrns,mk,nk -> rsk", M_b, bilinear_basis, bilinear_basis)
                traces = traces.reshape(-1, traces.shape[-1])[equality.indices]
                tr_p.append(traces.reshape(-1))

                b = b + 1
            result.append(sum(tr_p))

        assert b == len(M.blocks)

        return np.concatenate(result)

    # Computes \sum_i c_i A_i, where c_i = coeffs
    def combine_A(self, coeffs):
        k = 0
        blocks = []
        for equality in self.equalities:
            for bilinear_basis in equality.bilinear_bases:
                num_point = bilinear_basis.shape[1]
                m = equality.inner_block_size
                nindex = len(equality.indices)
                ncoeff = nindex * num_point

                coeff_block = coeffs[k:k+ncoeff].reshape(nindex, num_point)
                coeff_matrix = MPArray(np.zeros((m * m, num_point)))
                coeff_matrix[equality.indices] = coeff_block
                coeff_matrix = coeff_matrix.reshape((m, m, num_point))
                coeff_matrix = (coeff_matrix + np.swapaxes(coeff_matrix, 0, 1)) / 2

                blocks_b = np.einsum("rsk,mk,nk -> mrns", coeff_matrix,
                                    bilinear_basis, bilinear_basis)
                delta = bilinear_basis.shape[0]
                blocks.append(blocks_b.reshape(m * delta, m * delta))

            k = k + len(equality.indices) * num_point

        return BlockMatrix(blocks)


    def compute_residues(self, q):
        x, X, y, Y = q

        ## Compute residues (Eq. 2.25)
        P = self.combine_A(x) - X - self.C
        p = self.b - adjoint(self.B) @ x
        d = self.c - self.tr_A_vec(Y) - self.B @ y

        return P, p, d

    def compute_schur_complement(self, X_inv: BlockMatrix, Y: BlockMatrix):
        b = 0
        S = []
        for eq in self.equalities:
            S_b = []

            m = eq.inner_block_size
            Nk = eq.bilinear_bases[0].shape[1]
            Nindex = len(eq.indices)

            # When running with noisy data, we often have large blocks
            # that have relatively sparse indices, since the size of
            # the block is O(N_t) with O(N_t) indices, but the number
            # of possible indices is O(N_t^2). For moderate N_t, this
            # leads to very large intermediaries if we naively compute
            # the Schur complement for a dense set of indices and then
            # reduce (which is what was originally done here). What we
            # do instead here is add the pulling out of the relevant
            # indices to the einsum below, which allows the einsum
            # optimization to avoid large intermediaries.
            indices_mat = np.zeros((Nindex, m * m))
            indices_mat[np.arange(Nindex), eq.indices] = 1.0
            indices_mat = indices_mat.reshape((Nindex, m, m))

            for bases_b in eq.bilinear_bases:
                X_inv_b = _unravel_block(X_inv.blocks[b], eq.inner_block_size)
                Y_b = _unravel_block(Y.blocks[b], eq.inner_block_size)

                # Eqs. 2.49-2.50
                # Note: indices have been slightly rearranged to simplify
                # the code below
                U_b = np.einsum("msnr,mk,nl -> skrl", X_inv_b, bases_b, bases_b)
                V_b = np.einsum("msnr,mk,nl -> rlsk", Y_b, bases_b, bases_b)

                b = b + 1

                # Eq. 2.48
                S_b.append(np.einsum("skRK,rkSK,xrs,yRS -> xkyK",
                                     U_b, V_b, indices_mat, indices_mat,
                                     optimize=True))
                # Swap r <-> s
                S_b.append(np.einsum("rkRK,skSK,xrs,yRS -> xkyK",
                                     U_b, V_b, indices_mat, indices_mat,
                                     optimize=True))
                # Swap R <-> S
                S_b.append(np.einsum("skSK,rkRK,xrs,yRS -> xkyK",
                                     U_b, V_b, indices_mat, indices_mat,
                                     optimize=True))
                # Swap r <-> s and R <-> S
                S_b.append(np.einsum("rkSK,skRK,xrs,yRS -> xkyK",
                                     U_b, V_b, indices_mat, indices_mat,
                                     optimize=True))
            S_b = (1/4) * sum(S_b)
            S.append(S_b.reshape(Nindex * Nk, Nindex * Nk))

        assert b == len(X_inv.blocks)

        return BlockMatrix(S)

    def factor_schur_complement(self, X_inv, Y):
        S = self.compute_schur_complement(X_inv, Y)

        L = S.cholesky().as_matrix()
        LinvB = scipy_linalg_mp.solve_triangular(L, self.B, lower=True)
        # Note: QR decompsition on A <-> cholesky on A.T @ A
        L_Q = np.linalg.qr(LinvB)[1].T
        # Equivalently, could have:
        # Q = LinvB.T @ LinvB
        # L_Q = np.linalg.cholesky(Q)
        # But QR is more numerically stable

        return L, L_Q, LinvB

    def solve_schur_complement(self, X_inv, Y, residues, R, S_factor):
        P, p, d = residues
        L, L_Q, LinvB = S_factor

        # Below Eq. 2.28
        Z = X_inv @ (P @ Y - R)
        # Eq. 2.26
        schur_top_rhs = -d - self.tr_A_vec(Z)

        # Eq. 2.44
        a, b = schur_top_rhs, p
        # First term (lower)
        # Note that order matters here (a is intentionally overwitten)
        a = scipy_linalg_mp.solve_triangular(L, a, lower=True)
        b = b - LinvB.T @ a

        # Middle term (Q)
        b = cholesky_solve(L_Q, b)

        # Last term (upper)
        a = scipy_linalg_mp.solve_triangular(L.T, a + LinvB @ b, lower=False)

        dx, dy = a, b

        return dx, dy

    # Explicitly instantiates the matrices A_p as a list of BlockMatrix's.
    def make_matrices(self):
        A = []
        num_blocks = sum(len(eq.bilinear_bases)
                         for eq in self.equalities)
        b = 0
        for eq in self.equalities:
            E = []
            for index in eq.indices:
                m = eq.inner_block_size
                Ers = MPArray(np.zeros((m, m)))
                Ers = Ers.reshape(-1)
                Ers[index] = mp.mpf("1.0")
                Ers = Ers.reshape((m, m))
                Ers = (Ers + Ers.T) / 2

                E.append(Ers)

            # This loop is a bit awkward since we want to sum over the
            # outermost for loop, but not the inner for loops. We can't
            # rearrange the for loops since the size of the innermost loop
            # (k) depends on the iteration in the outer loop
            A_eq = None
            for idx, bilinear_basis in enumerate(eq.bilinear_bases):
                A_basis = []
                outer_prods = np.einsum("ik,jk -> kij", bilinear_basis,
                                        bilinear_basis)
                for Ers in E:
                    for k in range(bilinear_basis.shape[1]):
                        A_p = [None for _ in range(num_blocks)]
                        A_p[b] = np.kron(outer_prods[k], Ers)
                        A_basis.append(BlockMatrix(A_p))
                if idx == 0:
                    A_eq = A_basis
                else:
                    A_eq = [A_eq_i + A_basis_i
                            for A_eq_i, A_basis_i in zip(A_eq, A_basis)]
                b = b + 1
            A.extend(A_eq)

        assert b == num_blocks

        return A


class ReducedSDP:

    def __init__(self, sdp: SDP):
        self.sdp = sdp

        B = np.concatenate([eq.B for eq in sdp.equalities], axis=0)
        c = np.concatenate([eq.c for eq in sdp.equalities], axis=0)
        b = sdp.objective_vector

        U, S, Vdag = np.linalg.svd(B, full_matrices=True, compute_uv=True)

        k = len(S)
        U_parallel = U[:,:k]
        U_perp = U[:,k:]
        self.num_constrained = k

        btilde = (Vdag @ b) / S

        self.objective_primal_const = (c @ U_parallel @ btilde)[()]
        self.objective_primal = c @ U_perp
        self.C = sdp.C - sdp.combine_A(U_parallel @ btilde)

        self.c = self.objective_primal

        self.btilde = btilde
        self.U_parallel = U_parallel
        self.U_perp = U_perp

    def initialize(self, Omega_P, Omega_D):
        x, X, y, Y = self.sdp.initialize(Omega_P, Omega_D)
        y = np.array([0.0])
        x = x[self.num_constrained:]
        return x, X, y, Y

    def primal_objective(self, q):
        x, X, y, Y = q
        return np.dot(x, self.c)[()] + self.objective_primal_const

    def dual_objective(self, q):
        x, X, y, Y = q
        return trace_product(self.C, Y) + self.objective_primal_const

    # Computes Tr(A_p M), returned as a vector over p
    def tr_A_vec(self, M):
        return self.U_perp.T @ self.sdp.tr_A_vec(M)

    # Computes \sum_i c_i A_i, where c_i = coeffs
    def combine_A(self, coeffs):
        return self.sdp.combine_A(self.U_perp @ coeffs)

    def compute_residues(self, q):
        x, X, y, Y = q
        assert np.all(y == 0.0)

        ## Compute residues (Eq. 2.25)
        P = self.combine_A(x) - X - self.C
        p = np.array([0.0])
        d = self.c - self.tr_A_vec(Y)

        return P, p, d

    def compute_schur_complement(self, X_inv: BlockMatrix, Y: BlockMatrix):
        S = self.sdp.compute_schur_complement(X_inv, Y)
        return self.U_perp.T @ S.as_matrix() @ self.U_perp

    def factor_schur_complement(self, X_inv, Y):
        S = self.compute_schur_complement(X_inv, Y)
        L = np.linalg.cholesky(S)
        return L

    def solve_schur_complement(self, X_inv, Y, residues, R, S_factor):
        P, p, d = residues

        L = S_factor

        # Below Eq. 2.28
        Z = X_inv @ (P @ Y - R)
        # Eq. 2.26
        schur_top_rhs = -d - self.tr_A_vec(Z)

        dx = cholesky_solve(L, schur_top_rhs)
        dy = np.array([0.0])
        return dx, dy

    # Explicitly instantiates the matrices A_p as a list of BlockMatrix's.
    def make_matrices(self):
        A = self.sdp.make_matrices()

        result = []
        for q in range(self.U_perp.shape[1]):
            Atilde_q = BlockMatrix([None for _ in A[0].blocks])
            for p, A_p in enumerate(A):
                Atilde_q = Atilde_q + self.U_perp[p,q] * A_p
            result.append(Atilde_q)

        return result

    def recover_free_variables(self, q):
        x, X, y, Y = q

        x = self.U_perp @ x + self.U_parallel @ self.btilde
        y, _, _, _ = np.linalg.lstsq(self.sdp.B, self.sdp.c - self.sdp.tr_A_vec(Y))

        return x, X, y, Y

def read_sdp(path):
    filenames = list(os.listdir(path))
    block_nums = []
    for fname in filenames:
        if m := re.match(r"block_info_([0-9]+).json", fname):
            block_nums.append(int(m.groups()[0]))

    # Parsing blocks
    equalities = []
    for block_num in sorted(block_nums):
        with open(path / f"block_info_{block_num}.json") as f:
            block_info = json.load(f)
        inner_block_size = int(block_info["dim"])

        with open(path / f"block_data_{block_num}.json") as f:
            eq = LinearMatrixEquality.from_json(f, inner_block_size)
        equalities.append(eq)

    # # Normalization
    normalization_path = path / "normalization.json"
    if file_exists(normalization_path):
        with open(normalization_path) as f:
            normalization = json.load(f)
        normalization = _parse_array(normalization["normalization"])
        assert normalization[0] == 1.0
        assert np.all(normalization[1:] == 0.0)

    # Objective
    with open(path / "objectives.json") as f:
        objective = json.load(f)
    objective_b = _parse_array(objective["b"])

    return SDP(objective_b, equalities)

def write_sdp(sdp: SDP, path):
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True)

    for idx, eq in enumerate(sdp.equalities):
        block_info = {
            "dim" : eq.inner_block_size,
            # Not relevant for later reading
            "num_points": None,
            "num_bilinear_bases": len(eq.bilinear_bases)
        }

        block_data = eq.to_json_dict()

        with open(path / f"block_info_{idx}.json", 'w') as f:
            json.dump(block_info, f)
        with open(path / f"block_data_{idx}.json", 'w') as f:
            json.dump(block_data, f)

    objective = { "b": mpf_array_str(sdp.objective_vector) }
    with open(path / "objectives.json", 'w') as f:
        json.dump(objective, f)
