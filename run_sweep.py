import sys
import os
import time

# %%
import numpy as np
from mpmath import mp

import matplotlib.pyplot as plt

import pickle
from multiprocessing import Pool

import dataclasses
from dataclasses import dataclass

import sdppy
from sdppy.mp_array import MPArray
from sdppy.pmp import make_pmp, pmp_to_sdp2, make_noisy_pmp_sdp, make_noisy_pick_pmp_sdp
from sdppy.sdp import ReducedSDP
from sdppy.kernels import make_kernel, shift_kernel
from sdppy.generate import generate_test_corr, generate_test_cov0, generate_test_pick_corr
from sdppy.util import CorrelatorFlattener

from sdppy.solve_sdp import solve_sdp, default_params
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Nproc", type=int, default=48)
parser.add_argument("--digits", type=int, default=150)
parser.add_argument("--maxiter", type=int, default=1000)
parser.add_argument("--Nstate", type=int, default=16)
parser.add_argument("--Nt", type=int, default=16)
parser.add_argument("--mass-scale", type=float, required=True)
parser.add_argument("--m-gap", type=float, default=0.05)
parser.add_argument("--cov-scale", type=float, default=1.0)
parser.add_argument("--x-min", type=float, default=-5)
parser.add_argument("--x-max", type=float, default=0)
parser.add_argument("--tolerance", type=float, default=1e-30)
parser.add_argument("--num-x", type=int, default=200)
parser.add_argument("--scalar", type=bool, action=argparse.BooleanOptionalAction)
parser.add_argument("--unreduced", type=bool, action=argparse.BooleanOptionalAction)
parser.add_argument("--add-noise", type=bool, action=argparse.BooleanOptionalAction)
parser.add_argument("--kernel-type", type=str, required=True)
parser.add_argument("--sweep-type", type=str, required=True)
parser.add_argument("--output", type=str, required=True,
                    help="File for writing output")

args = parser.parse_args()
if args.digits > 0:
    mp.dps = args.digits
print(mp)

## Parameters
Nstate = args.Nstate
Nt = args.Nt
if args.kernel_type == "tau_pick":
    corr, Z_exact, energies, omega = generate_test_pick_corr(Nstate, Nt)
else:
    corr, Z_exact, energies = generate_test_corr(Nstate, Nt)
    omega = None
if args.scalar:
    corr = corr[:,:1,:1]
r = corr.shape[-1]

print(f"{Z_exact.shape=}")
print(f"{energies.shape=}")
print(f"{corr.shape=}")

# %%
dt_cov = mp.mpf("1.3")
shrinkage = mp.mpf("0.5")

Ncorr = Nt * r * (r + 1) // 2
alpha0 = args.cov_scale

cov = generate_test_cov0(corr, dt_cov, shrinkage)

if args.add_noise:
    flattener = CorrelatorFlattener(Nt, r)
    cov_flat = flattener.flatten_cov(cov)
    corr_flat = flattener.flatten_corr(corr)

    rng = np.random.default_rng(0)
    if cov_flat.cplx:
        cov_real = sdppy.util.covariance_real(cov_flat)
        dcorr_flat = rng.multivariate_normal(np.zeros(cov_real.shape[0]), cov_real)
        dcorr_flat = sdppy.util.complex_from_components(dcorr_flat)
    else:
        dcorr_flat = rng.multivariate_normal(np.zeros(corr_flat.shape), cov_flat)

    dcorr = flattener.unflatten_corr(dcorr_flat)
else:
    dcorr = np.zeros(corr.shape)

x_plot = MPArray(np.linspace(args.x_min, args.x_max, num=args.num_x))

if args.kernel_type == "tau_pick":
    poly_basis_type = "monomial"
else:
    poly_basis_type = ("chebyshev", 0, 1)

numer, denom, bounds = make_kernel(args.kernel_type, Nt,
                                   mass_scale=args.mass_scale,
                                   m_gap=args.m_gap)

print(f"{numer=}")
print(f"{denom=}")
print(f"{bounds=}")

tol = mp.mpf(args.tolerance)
sdp_solve_params = dataclasses.replace(default_params,
                                       maxiter=args.maxiter,
                                       maxStalls=30,
                                       dualityGapThreshold=tol,
                                       primalErrorThreshold=tol,
                                       dualErrorThreshold=tol)

def _solve_sdp(alpha_i, q0=None, prefix=None, *, maximize, shift):
    numer_i, denom_i = shift_kernel(numer, denom, shift)
    sdp = make_noisy_pmp_sdp(corr + alpha_i * dcorr, cov, numer_i, denom_i,
                             maximize=maximize,
                             bounds=bounds,
                             sigma0=mp.sqrt(2 * Ncorr) * alpha_i,
                             poly_basis_type=poly_basis_type)

    if not args.unreduced:
        sdp = ReducedSDP(sdp)

    q, history = solve_sdp(sdp, sdp_solve_params, verbose=True,
                           q0=q0, prefix=prefix)
    history["solution"] = q

    return history

def _solve_pick_sdp(alpha_i, prefix=None, *, maximize, shift):
    numer_i, denom_i = shift_kernel(numer, denom, shift)
    sdp = make_noisy_pick_pmp_sdp(corr, omega, cov, numer_i, denom_i,
                                  maximize=maximize,
                                  bounds=bounds,
                                  sigma0=mp.sqrt(2 * Ncorr) * alpha_i)

    if not args.unreduced:
        sdp = ReducedSDP(sdp)

    q, history = solve_sdp(sdp, sdp_solve_params,
                           verbose=True, prefix=prefix)
    history["solution"] = q

    return history

start = time.perf_counter()

def _process(i):
    idx = i // 2
    maximize = i % 2 == 1
    if args.sweep_type == "variance":
        alpha_i = 10**x_plot[idx] * alpha0
        shift = 0.0
    elif args.sweep_type == "center":
        alpha_i = alpha0
        shift = x_plot[idx]
    prefix = f"[{i=:04} max={int(maximize)} {float(x_plot[idx]):.2f}] "
    try:
        if args.kernel_type == "tau_pick":
            histories = _solve_pick_sdp(alpha_i, prefix=prefix, maximize=maximize,
                                        shift=shift)
        else:
            histories = _solve_sdp(alpha_i, prefix=prefix, maximize=maximize,
                                   shift=shift)
    except Exception as e:
        print(f"{prefix} Error: {e}")
        raise e
    return histories

Nproc = args.Nproc
print(f"Running SDP with {Nproc} processes")

with Pool(Nproc) as p:
    histories = list(p.map(_process, range(2*x_plot.shape[0])))
end = time.perf_counter()

print(f"Total optimization time: {end - start} s")

out_filename = args.output
with open(out_filename, 'wb') as f:
    pickle.dump((args, histories), f)
