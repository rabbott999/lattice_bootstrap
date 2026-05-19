#!/bin/bash -l

set -eu

DATA=$(readlink -f ./data)
mkdir -p ${DATA}

NSTATE=${NSTATE:-96}
NT=${NT:-4}
EPS_SMEAR=0.10

echo "NSTATE: ${NSTATE}"
echo "NT: ${NT}"

NPROC=$(nproc)

PARAMS="--digits 150"
PARAMS+=" --maxiter 10000"
PARAMS+=" --tolerance 1e-15"
PARAMS+=" --Nproc ${NPROC}"
PARAMS+=" --Nstate ${NSTATE}"
PARAMS+=" --Nt ${NT}"
PARAMS+=" --cov-scale 1e-4"
PARAMS+=" --mass-scale ${EPS_SMEAR}"
PARAMS+=" --x-min 1.0"
PARAMS+=" --x-max 0.6"
PARAMS+=" --num-x 30"
PARAMS+=" --sweep-type center"
PARAMS+=" --kernel-type cauchy_moment"
if [[ -v SCALAR ]]; then
    if [[ -v ADD_NOISE ]]; then
	echo "Running noisy scalar case"
	PARAMS+=" --add-noise"
	PARAMS+=" --output ${DATA}/cauchy_moment_scalar_Nt${NT}_noisy.pickle"
    else
	echo "Running exact scalar case"
	PARAMS+=" --output ${DATA}/cauchy_moment_scalar_Nt${NT}.pickle"
    fi
    PARAMS+=" --scalar"
else
    if [[ -v ADD_NOISE ]]; then
	echo "Running noisy block case"
	PARAMS+=" --add-noise"
	PARAMS+=" --output ${DATA}/cauchy_moment_Nt${NT}_noisy.pickle"
    else
	echo "Running exact block case"
	PARAMS+=" --output ${DATA}/cauchy_moment_Nt${NT}.pickle"
    fi
fi

set -x

uv run python -u run_sweep.py ${PARAMS}
