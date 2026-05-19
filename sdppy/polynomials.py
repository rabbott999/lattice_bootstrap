import numpy as np
from numpy.polynomial import Polynomial

from mpmath import mp

# Computes p(x - dx) as a polynomial
def shift_poly(p, dx):
    line = Polynomial(np.array([-dx, 1.0], dtype=object))
    return p(line)

def monomial_basis(max_degree):
    return [Polynomial(np.array([0.0 for _ in range(n)] + [1.0]))
            for n in range(max_degree+1)]

## Note: These are the probabilist's Hermite polynomials (see
## https://en.wikipedia.org/wiki/Hermite_polynomials). The difference
## is a factor of 2^n, which makes the probabilist's polynomials
## smaller.
def hermite_polynomials(max_degree):
    x = Polynomial([0.0, 1.0])
    one = Polynomial([1.0])

    if max_degree == 0:
        return [one]

    poly = [one, x]

    for n in range(1,max_degree):
        poly_np1 = x * poly[n] - n * poly[n - 1]
        poly.append(poly_np1)
    return poly

def _chebyshev_polynomials(max_degree):
    x = Polynomial([0.0, 1.0])
    one = Polynomial([1.0])

    if max_degree == 0:
        return [one]

    Tn_minus_one = one
    Tn = x

    T = [one, x]
    for n in range(2,max_degree+1):
        Tn_plus_one = 2 * x * Tn - Tn_minus_one
        T.append(Tn_plus_one)
        Tn_minus_one = Tn
        Tn = Tn_plus_one
    return T

def chebyshev_polynomials(max_degree, x0, x1):
    x = Polynomial([0.0, 1.0])
    one = Polynomial([1.0])

    arg = (x - x0 * one) / (x1 - x0) # Lives in [0, 1]
    arg = 2 * arg - 1 # Lives in [-1, 1]
    return [T(arg) for T in _chebyshev_polynomials(max_degree)]

def _make_poly_basis(basis_name, max_degree):
    if basis_name == "monomial":
        return monomial_basis(max_degree)
    elif basis_name == "hermite":
        return hermite_polynomials(max_degree)
    elif basis_name[0] == "chebyshev":
        x0, x1 = basis_name[1], basis_name[2]
        return chebyshev_polynomials(max_degree, x0, x1)
    else:
        raise ValueError(f"Unkown polynomial basis '{basis_name}'")

# Creates a matrix for changing basis from the monomial basis to the
# target basis
def change_of_basis_matrix(poly_basis):
    from .mp_array import MPArray

    n = len(poly_basis)
    result = MPArray(np.zeros((n, n)))

    for i, p in enumerate(poly_basis):
        for j, a in enumerate(p.coef):
            result[i, j] = a

    return result

def evaluation_matrix(poly_basis, nodes):
    from .mp_array import MPArray

    n = len(nodes)
    result = MPArray(np.zeros((n, len(poly_basis))))

    for i, p in enumerate(poly_basis):
        for j, node_j in enumerate(nodes):
            result[j, i] = p(node_j)

    return result

class PolynomialBasis:

    def __init__(self, poly_basis):
        self.polynomials = poly_basis
        self.change_of_basis = change_of_basis_matrix(poly_basis)
        self.change_of_basis_inverse = np.linalg.inv(self.change_of_basis)

    def covariance_from_monomial(self, cov):
        L = self.change_of_basis
        return np.einsum("xs,sabtcd,ty -> xabycd", L, cov, L.T)

    def coefficients_from_monomial(self, coeffs):
        L = self.change_of_basis
        return np.einsum("st,tab -> sab", L, coeffs)

    def coefficients_to_monomial(self, coeffs):
        Linv = self.change_of_basis_inverse
        return np.einsum("st,tab -> sab", Linv, coeffs)

    def values_to_coefficients(self, nodes, values):
        assert len(nodes) == len(values)
        assert len(nodes) == len(self.polynomials)

        M = evaluation_matrix(self.polynomials, nodes)
        return np.linalg.solve(M, values)

    def coefficients_to_values(self, nodes, coefficients):
        assert len(nodes) == len(self.polynomials)

        M = evaluation_matrix(self.polynomials, nodes)
        return M @ coefficients

def make_poly_basis(basis_name, max_degree):
    poly_basis = _make_poly_basis(basis_name, max_degree)
    return PolynomialBasis(poly_basis)
