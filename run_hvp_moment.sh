#!/bin/bash -l

set -eu

DATA=$(readlink -f ./data)
mkdir -p ${DATA}

NSTATE=${NSTATE:-96}
NT=${NT:-4}
MMU=0.2

echo "NSTATE: ${NSTATE}"
echo "NT: ${NT}"

NPROC=$(nproc)

PARAMS="--digits 150"
PARAMS+=" --maxiter 10000"
PARAMS+=" --tolerance 1e-15"
PARAMS+=" --Nproc ${NPROC}"
PARAMS+=" --Nstate ${NSTATE}"
PARAMS+=" --Nt ${NT}"
PARAMS+=" --mass-scale ${MMU}"
PARAMS+=" --x-min -5"
PARAMS+=" --x-max -2"
PARAMS+=" --num-x 12"
PARAMS+=" --sweep-type variance"
PARAMS+=" --kernel-type hvp_moment"
if [[ -v SCALAR ]]; then
    if [[ -v ADD_NOISE ]]; then
	echo "Running noisy scalar case"
	PARAMS+=" --add-noise"
	PARAMS+=" --output ${DATA}/hvp_moment_scalar_Nt${NT}_noisy.pickle"
    else
	echo "Running exact scalar case"
	PARAMS+=" --output ${DATA}/hvp_moment_scalar_Nt${NT}.pickle"
    fi
    PARAMS+=" --scalar"
else
    if [[ -v ADD_NOISE ]]; then
	echo "Running noisy block case"
	PARAMS+=" --add-noise"
	PARAMS+=" --output ${DATA}/hvp_moment_Nt${NT}_noisy.pickle"
    else
	echo "Running exact block case"
	PARAMS+=" --output ${DATA}/hvp_moment_Nt${NT}.pickle"
    fi
fi

set -x

uv run run_sweep.py ${PARAMS}
