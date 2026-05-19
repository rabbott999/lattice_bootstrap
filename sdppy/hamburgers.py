from dataclasses import dataclass

import abc
import numpy as np

from .mp_array import MPArray

def _adjoint(x):
    return np.conj(np.swapaxes(x, -1, -2))

def matrix_sqrt(M):
    eigvals, eigvecs = np.linalg.eigh(M)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ _adjoint(eigvecs)

def matrix_operator_norm(M):
    svals = np.linalg.svdvals(M)
    return np.max(svals)

@dataclass
class MatrixBall:
    center: MPArray
    left_radius: MPArray
    right_radius: MPArray

    def projection(self, v_left, v_right=None):
        if v_right is None:
            v_right = v_left

        center = np.dot(np.conj(v_left), self.center @ v_right)
        left_norm = np.linalg.norm(self.left_radius @ v_left)
        right_norm = np.linalg.norm(self.right_radius @ v_right)
        radius = left_norm * right_norm

        return center, radius

    def norm(self):
        left_norm = matrix_operator_norm(self.left_radius)
        right_norm = matrix_operator_norm(self.right_radius)
        return left_norm * right_norm


class JTheoryProblem:
    def __init__(self):
        pass

    @abc.abstractmethod
    def _b(self, z):
        pass

    @abc.abstractmethod
    def _c(self, z):
        pass

    def _bc_stacked(self, z):
        return np.concatenate([self._b(z), -self._c(z)], axis=1)

    def _A(self, z):
        ident_2r = np.eye(2 * self.r, 2 * self.r)
        return ident_2r - 1j * z * (self.J @ (_adjoint(self._bc_stacked(np.conj(z))) @ self.rhs))

    def get_ball(self, z):
        Ainv = np.linalg.inv(self._A(z))
        W = _adjoint(Ainv) @ self.J @ Ainv
        W = W.reshape(2, self.r, 2, self.r)
        R = -W[0, :, 0, :]
        S = W[0, :, 1, :]
        T = -W[1, :, 1, :]

        Rinv = np.linalg.inv(R)
        C = Rinv @ S
        rho_d = _adjoint(S) @ Rinv @ S - T
        rho_g = Rinv

        return MatrixBall(C, matrix_sqrt(rho_g), matrix_sqrt(rho_d))


class MomentProblem(JTheoryProblem):

    def __init__(self, corr):
        m = (corr.shape[0] + 1) // 2
        if len(corr.shape) == 1:
            r = 1
        else:
            r = corr.shape[-1]

        S = MPArray(np.zeros((m, r, m, r), dtype=np.complex128))
        for i in range(m):
            for j in range(m):
                S[i, :, j, :] = corr[i + j]

        S = S.reshape((m * r, m * r))
        Sinv = np.linalg.inv(S)

        ident_r = np.eye(r, r)
        J2 = MPArray(np.zeros((2, r, 2, r), dtype=np.complex128))
        J2[0,:,1,:] = 1j * ident_r
        J2[1,:,0,:] = -1j * ident_r
        J2 = J2.reshape(2*r, 2*r)

        self.m = m
        self.r = r
        self.S = S
        self.Sinv = Sinv
        self.corr = corr
        self.J = J2

        rhs = Sinv @ self._bc_stacked(0.0)
        self.rhs = rhs

    def _b(self, z):
        ident = MPArray(np.eye(self.r, self.r))
        return np.concatenate([z**i * ident for i in range(self.m)], axis=0)

    def _c(self, z):
        x = MPArray(np.zeros((self.r, self.r), dtype=np.complex128))
        result = [x]
        for i in range(self.m-1):
            x = z * x + self.corr[i]
            result.append(x)
        return -np.concatenate(result, axis=0)

class NevanlinnnaPickProblem(JTheoryProblem):

    def __init__(self, z, w):
        m = w.shape[0]
        if len(w.shape) == 1:
            r = 1
        else:
            r = w.shape[-1]

        assert z.shape == (m,)

        ident_r = np.eye(r, r)
        S = MPArray(np.zeros((m, r, m, r), dtype=np.complex128))
        for i in range(m):
            for j in range(m):
                prefactor = 1 / (z[i] - np.conj(z[j]))
                S[i, :, j, :] = prefactor * (w[i] - _adjoint(w[j]))

        S = S.reshape((m * r, m * r))
        Sinv = np.linalg.inv(S)

        J2 = MPArray(np.zeros((2, r, 2, r), dtype=np.complex128))
        J2[0,:,1,:] = 1j * ident_r
        J2[1,:,0,:] = -1j * ident_r
        J2 = J2.reshape(2*r, 2*r)

        self.m = m
        self.r = r
        self.Sinv = Sinv
        self.J = J2
        self.z = z
        self.w = w

        rhs = Sinv @ self._bc_stacked(0.0)
        self.rhs = rhs

    def _b(self, z):
        ident = MPArray(np.eye(self.r, self.r))
        return np.concatenate([1/(z - self.z[i]) * ident for i in range(self.m)], axis=0)

    def _c(self, z):
        return np.concatenate([1/(z - self.z[i]) * self.w[i] for i in range(self.m)], axis=0)
