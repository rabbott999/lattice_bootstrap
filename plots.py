import numpy as np
from mpmath import mp

import pathlib

import matplotlib.pyplot as plt

from sdppy.mp_array import MPArray
from sdppy.generate import generate_test_corr, generate_test_pick_corr
from sdppy.kernels import make_kernel, shift_kernel

import pickle

mp.dps = 200

## Parameters

data_dir = pathlib.Path("data")

default_figsize = (8, 6)

plt.rcParams['figure.figsize'] = default_figsize
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['font.size'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amssymb} \usepackage{amsmath}"
plt.rcParams['savefig.bbox'] = 'tight'

def savefig(filename):
    plot_dir = pathlib.Path("paper_plots")
    plt.savefig(plot_dir / filename)
    plt.clf()
    plt.figure(figsize=default_figsize)

## Extracting data
def extract_bounds(histories):
    bounds_min = []
    bounds_max = []
    for (hist_min, hist_max) in zip(histories[0::2], histories[1::2]):
        bounds_min.append(hist_min["dual_objective"][-1])
        bounds_max.append(-hist_max["dual_objective"][-1])

    bounds_min = MPArray(np.array(bounds_min))
    bounds_max = MPArray(np.array(bounds_max))
    return bounds_min, bounds_max

class SweepData:

    def __init__(self, filename):
        with open(filename, 'rb') as f:
            args, histories = pickle.load(f)

        numer, denom, bounds = make_kernel(args.kernel_type, args.Nt,
                                            mass_scale=args.mass_scale,
                                            m_gap=args.m_gap)

        from numpy.polynomial import Polynomial
        def eval_kernel(lam):
            return Polynomial(numer)(lam) / Polynomial(denom)(lam)

        if args.kernel_type in ["tau_pick"]:
            corr, Z_exact, energies, omega = generate_test_pick_corr(args.Nstate, Nt=args.Nt)
        else:
            corr, Z_exact, energies = generate_test_corr(args.Nstate, Nt=args.Nt)

        if args.scalar:
            corr = corr[:,:1,:1]

        if args.kernel_type == "tau_moment":
            # Need to include theta function explicitly
            exact = np.sum(np.abs(Z_exact[:,0])**2 * eval_kernel(np.exp(-energies)) * (energies < args.mass_scale))
        elif args.kernel_type == "tau_pick":
            exact = np.sum(np.abs(Z_exact[:,0])**2 * eval_kernel(energies) * (energies < args.mass_scale))
        elif args.kernel_type == "hvp_moment":
            exact = np.sum(np.abs(Z_exact[:,0])**2 * eval_kernel(np.exp(-energies)))
        else:
            assert args.kernel_type in ["cauchy_moment"]
            lambdas = np.exp(-energies)
            E_plot = np.linspace(args.x_min, args.x_max, num=args.num_x)
            exact = []
            for E_i in E_plot:
                numer_i, denom_i = shift_kernel(numer, denom, E_i)
                rho_i = np.sum(np.abs(Z_exact[:,0])**2 * Polynomial(numer_i)(lambdas) / Polynomial(denom_i)(lambdas))
                exact.append(rho_i)
            exact = np.stack(exact)

        bounds_min, bounds_max = extract_bounds(histories)
        x = np.linspace(args.x_min, args.x_max, num=args.num_x)

        self.args = args
        self.corr = corr
        self.histories = histories
        self.numer = numer
        self.denom = denom
        self.exact = exact
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.x = x


## Plotting functions
def plot_variance_sweep(sweep, shift=1.0, *, ax=None, **kwargs):
    x = 10**sweep.x

    mid = (sweep.bounds_min + sweep.bounds_max) / 2
    mid = mid.astype(np.float64)
    err = (sweep.bounds_max - sweep.bounds_min) / 2
    err = err.astype(np.float64)

    if ax is None:
        ax = plt.gca()
    ax.errorbar(x*shift, mid, yerr=err, capsize=5, fmt='', ls='none', **kwargs)

## Tau lifetime
def tau_lifetime_bound_noisy():
    data_tau_moment_noisy = SweepData(data_dir / "tau_moment_Nt20_noisy.pickle")
    data_tau_pick_noisy = SweepData(data_dir / "tau_pick_Nt10_noisy.pickle")
    fig, axs = plt.subplots(2,1,figsize=(6,8),sharex=True)

    plot_variance_sweep(data_tau_moment_noisy, 1.0, ax=axs[0], label="Moment Problem")
    plot_variance_sweep(data_tau_pick_noisy, 1.0, ax=axs[1], label="Nevanlinna--Pick")

    axs[0].axhline(data_tau_moment_noisy.exact, color="grey", linestyle="dashed", label="Exact")
    axs[1].axhline(data_tau_pick_noisy.exact, color="grey", linestyle="dashed", label="Exact")

    axs[0].set_ylabel("$R_L$")
    axs[1].set_ylabel("$R_L$")

    axs[0].set_xscale("log")
    axs[1].set_xscale("log")

    axs[1].set_xlabel(r"$\alpha$")

    axs[0].legend(loc="best")
    axs[1].legend(loc="best")

    fig.subplots_adjust(wspace=0, hspace=0.05, left=0.15, right=0.95,
                        bottom=0.15, top=0.95)

if __name__ == '__main__':
    tau_lifetime_bound_noisy()
    savefig("tau_decay_noisy.pdf")

## HVP
def hvp_scalar_block_noisy():
    data_hvp_block_noisy = SweepData(data_dir / "hvp_moment_Nt20_noisy.pickle")
    data_hvp_scalar_noisy = SweepData(data_dir / "hvp_moment_scalar_Nt20_noisy.pickle")

    plot_variance_sweep(data_hvp_block_noisy, 1.0, label="Matrix ($r=2$)")
    plot_variance_sweep(data_hvp_scalar_noisy, 1.2, label="Scalar ($r=1$)")

    plt.axhline(data_hvp_block_noisy.exact, color="grey",
                linestyle="dashed", label="Exact")

    plt.ylabel(r"$a_\mu^\text{HVP}$")
    plt.xlabel(r"$\alpha$")
    plt.xscale("log")

    plt.legend(loc="best")

if __name__ == '__main__':
    hvp_scalar_block_noisy()
    savefig("hvp_noisy.pdf")

## Cauchy Kernel

def extract_mid_err(sweep):
    mid = (sweep.bounds_min + sweep.bounds_max) / 2
    mid = mid.astype(np.float64)
    err = (sweep.bounds_max - sweep.bounds_min) / 2
    err = err.astype(np.float64)

    return mid, err

def cauchy_reconstruct_noisy():
    data_cauchy_block_noisy = SweepData(data_dir / "cauchy_moment_Nt20_noisy.pickle")
    data_cauchy_scalar_noisy = SweepData(data_dir / "cauchy_moment_scalar_Nt20_noisy.pickle")

    fig, axs = plt.subplots(2,1, height_ratios=[3,1], sharex=True, figsize=(6,8))

    x_plot = -np.log(data_cauchy_block_noisy.x)

    exact = data_cauchy_scalar_noisy.exact.astype(np.float64)

    mid_block, err_block = extract_mid_err(data_cauchy_block_noisy)
    axs[0].fill_between(x_plot, mid_block-err_block, mid_block+err_block, alpha=0.2, label="Matrix ($r=2$)")
    axs[1].plot(x_plot, (mid_block - exact) / err_block)

    mid_scalar, err_scalar = extract_mid_err(data_cauchy_scalar_noisy)
    axs[0].fill_between(x_plot, mid_scalar-err_scalar, mid_scalar+err_scalar, alpha=0.2, label="Scalar ($r=1$)")
    axs[1].plot(x_plot, (mid_scalar - exact) / err_scalar)


    axs[0].plot(x_plot, exact, color="grey", linestyle="dashed", label="Exact")
    axs[1].axhline(1, color="grey")
    axs[1].axhline(-1, color="grey")

    axs[0].set_ylabel(r"$\tilde{\rho}_\epsilon(E)$")
    axs[1].set_ylabel(r"$\frac{\text{center} - \text{exact}}{\text{radius}}$")
    axs[1].set_xlabel("$E$")

    axs[0].legend(loc="best")

    axs[0].set_ylim(0, 44)

    fig.subplots_adjust(wspace=0, hspace=0.05, left=0.15, right=0.95,
                        bottom=0.15, top=0.95)

if __name__ == '__main__':
    cauchy_reconstruct_noisy()
    savefig("cauchy_noisy.pdf")
