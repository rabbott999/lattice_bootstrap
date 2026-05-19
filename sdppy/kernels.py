import numpy as np
import sympy

from .mp_array import MPArray
from mpmath import mp

# Given K(x) = n(x)/d(x), computes K(x - dx) by shifting the numerator
# and denominator
def shift_kernel(numer, denom, shift):
    from numpy.polynomial import Polynomial
    from .polynomials import shift_poly
    numer_p = shift_poly(Polynomial(numer), shift)
    denom_p = shift_poly(Polynomial(denom), shift)
    return MPArray(numer_p.coef), MPArray(denom_p.coef)

def tau_variables():
    E, m_tau = sympy.symbols("E m_tau")
    return E, m_tau

def tau_kernel_L():
    E, m_tau = tau_variables()
    x = E / m_tau
    return (1 - x**2)**2 / x

def tau_kernel_moment(m_tau_lat, degree):
    from mpmath import pi
    E, m_tau = tau_variables()
    lam = sympy.symbols("lambda")
    K = tau_kernel_L()
    K = (2 * pi * K).subs(E, -sympy.log(lam))
    K = sympy.simplify((1 - lam) * sympy.series(K, lam, x0=1, n=degree).removeO())
    K = K.subs(m_tau, m_tau_lat)
    K = sympy.Poly(K, lam)

    coeffs = list(reversed(K.all_coeffs()))
    numer = MPArray(np.array(coeffs))
    # denom = 1 - lambda
    denom = MPArray(np.array([1.0, -1.0]))
    return numer, denom

def tau_kernel_pick(m_tau_lat, degree):
    from mpmath import pi
    E, m_tau = tau_variables()
    K = 2 * pi * E * tau_kernel_L().subs(m_tau, m_tau_lat)
    K = sympy.Poly(K, E)
    coeffs = list(reversed(K.all_coeffs()))
    numer = MPArray(np.array(coeffs))
    denom = MPArray(np.array([0.0, 1.0]))
    return numer, denom

def cauchy_kernel(omega, epsilon):
    numer = MPArray(np.array([1.0*epsilon], dtype=object))
    denom = MPArray(np.array([omega**2 + epsilon**2, -2 * omega, 1.0],
                             dtype=object))
    return numer, denom

def _hvp_kernel_large(s, m_mu_lat):
    beta_mu = np.sqrt(1 - 4 * m_mu_lat**2 / s)
    x = (1 - beta_mu) / (1 + beta_mu)

    term1 = x**2/2 * (2 - x**2)
    prefactor = (1 + x**2) * (1 + x)**2 / x**2
    term2 = prefactor * (np.log(1 + x) - x + x**2/2)
    term3 = (1 + x) / (1 - x) * x**2 * np.log(x)
    return term1 + term2 + term3

def _hvp_kernel_small(s, m_mu_lat):
    tau = s/(4*m_mu_lat**2)
    term1 = 1/2 - 4 * tau - 4 * tau * (1 - 2*tau) * np.log(4 * tau)
    term2 = -2 * (1 - 8*tau + 8*tau**2) * np.sqrt(tau/(1-tau)) * np.arccos(np.sqrt(tau))
    return term1 + term2

def hvp_kernel(s, m_mu_lat):
    mask = s > 4 * m_mu_lat**2
    s_safe_large = np.where(mask, s, 5 * m_mu_lat**2)
    s_safe_small = np.where(mask, m_mu_lat**2, s)

    K_large = _hvp_kernel_large(s_safe_large, m_mu_lat)
    K_small = _hvp_kernel_small(s_safe_small, m_mu_lat)
    return np.where(mask, K_large, K_small)

def hvp_kernel_moment(m_mu_lat, Nt):
    def _hvp_kernel_moment_numer(lam):
        sqrt_s = -np.log(lam)
        s = sqrt_s**2
        if lam != 1.0:
            return hvp_kernel(s, m_mu_lat) * 2 * (1 - lam) / sqrt_s
        return hvp_kernel(s, m_mu_lat) * 2

    def f(x):
        return _hvp_kernel_moment_numer(MPArray(np.array(x))).real[()]

    f_approx = mp.chebyfit(f, [0, 1], Nt)

    numer = MPArray(np.array(list(reversed(f_approx))))
    denom = MPArray(np.array([1.0, -1.0]))

    return numer, denom

def make_kernel(kernel_type, Nt, *, mass_scale, m_gap):
    lam_gap = np.exp(-m_gap)
    if kernel_type == "tau_moment":
        lam_tau = np.exp(-mass_scale)
        numer, denom = tau_kernel_moment(mass_scale, Nt)
        bounds = [
            (0.0, lam_tau, False),
            (lam_tau, lam_gap, True)
        ]
    elif kernel_type == "tau_pick":
        m_tau = mass_scale
        numer, denom = tau_kernel_pick(mass_scale, Nt)
        bounds = [
            (m_gap, m_tau, True),
            (m_tau, None, False)
        ]
    elif kernel_type == "hvp_moment":
        numer, denom = hvp_kernel_moment(mass_scale, Nt)
        bounds = [
            (0, lam_gap, True)
        ]
    elif kernel_type == "cauchy_moment":
        numer, denom = cauchy_kernel(omega=0.0, epsilon=mass_scale)
        bounds = [
            (0, 1.0, True)
        ]
    else:
        raise ValueError(f"Unknown kernel type: '{kernel_type}'")

    return numer, denom, bounds

if __name__ == '__main__':
    from mpmath import mp

    m_tau_lat = mp.mpf("0.8")
    print(tau_kernel_moment(m_tau_lat, 3))

