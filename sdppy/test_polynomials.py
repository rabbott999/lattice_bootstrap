import pytest
import numpy as np
from numpy.polynomial import Polynomial

from .polynomials import make_poly_basis

def test_chebyshev():
    from .polynomials import chebyshev_polynomials

    # See https://en.wikipedia.org/wiki/Chebyshev_polynomials#Examples
    T_wikipedia = [
        Polynomial([1.0]),
        Polynomial([0.0, 1.0]),
        Polynomial([-1.0, 0.0, 2.0]),
        Polynomial([0.0, -3.0, 0.0, 4.0]),
        Polynomial([1.0, 0.0, -8.0, 0.0, 8.0]),
        Polynomial([0.0, 5.0, 0.0, -20.0, 0.0, 16.0]),
        Polynomial([-1.0, 0.0, 18.0, 0.0, -48.0, 0.0, 32.0]),
        Polynomial([0.0, -7.0, 0.0, 56.0, 0.0, -112.0, 0.0, 64.0]),
        Polynomial([1.0, 0.0, -32.0, 0.0, 160.0, 0.0, -256.0, 0.0, 128.0]),
        Polynomial([0.0, 9.0, 0.0, -120.0, 0.0, 432.0, 0.0, -576.0, 0.0, 256.0]),
    ]

    T = chebyshev_polynomials(9, -1, 1)

    assert len(T) == len(T_wikipedia)

    for p, q in zip(T, T_wikipedia):
        assert p == q

all_poly_basis_types = ["monomial", "hermite", ("chebyshev", 0, 1)]

@pytest.mark.parametrize("poly_basis_type", all_poly_basis_types)
def test_poly_degree(poly_basis_type):
    for d in range(4):
        basis = make_poly_basis(poly_basis_type, d).polynomials
        assert len(basis) == d + 1

@pytest.mark.parametrize("poly_basis_type", all_poly_basis_types)
def test_poly_interp(poly_basis_type):
    poly_basis_type = "hermite"
    max_degree = 4
    poly_basis = make_poly_basis(poly_basis_type, max_degree)

    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=(max_degree+1,))
    nodes = rng.uniform(0, 1, size=(max_degree+1,))

    vals = poly_basis.coefficients_to_values(nodes, coeffs)
    coeffs2 = poly_basis.values_to_coefficients(nodes, vals)

    vals_manual = [sum(c_i * p_i(x_j)
                       for c_i, p_i in zip(coeffs, poly_basis.polynomials))
                   for x_j in nodes]

    assert np.allclose(coeffs, coeffs2)
    assert np.allclose(vals, vals_manual)
