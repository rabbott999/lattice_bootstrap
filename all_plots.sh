#!/bin/bash -l

NT=20 ADD_NOISE=1 ./run_cauchy_moment.sh
NT=20 ADD_NOISE=1 SCALAR=1 ./run_cauchy_moment.sh
NT=20 ADD_NOISE=1 ./run_hvp_moment.sh
NT=20 ADD_NOISE=1 SCALAR=1 ./run_hvp_moment.sh
NT=20 ADD_NOISE=1 ./run_tau_moment.sh
NT=20 ADD_NOISE=1 SCALAR=1 ./run_tau_moment.sh
NT=10 ADD_NOISE=1 ./run_tau_pick.sh
NT=10 ADD_NOISE=1 SCALAR=1 ./run_tau_pick.sh

mkdir -p paper_plots
uv run plots.py
