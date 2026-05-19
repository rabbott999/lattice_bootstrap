#!/bin/bash -l

set -eu

DATA=$(readlink -f ./data)
mkdir -p ${DATA}

NSTATE=${NSTATE:-96}
NT=${NT:-4}
MTAU=0.35

echo "NSTATE: ${NSTATE}"
echo "NT: ${NT}"

NPROC=$(nproc)

PARAMS="--digits 150"
PARAMS+=" --maxiter 10000"
PARAMS+=" --tolerance 1e-15"
PARAMS+=" --Nproc ${NPROC}"
PARAMS+=" --Nstate ${NSTATE}"
PARAMS+=" --Nt ${NT}"
PARAMS+=" --mass-scale ${MTAU}"
PARAMS+=" --x-min -5"
PARAMS+=" --x-max -2"
PARAMS+=" --num-x 12"
PARAMS+=" --sweep-type variance"
PARAMS+=" --kernel-type tau_pick"
if [[ -v SCALAR ]]; then
    if [[ -v ADD_NOISE ]]; then
	echo "Running noisy scalar case"
	PARAMS+=" --add-noise"
	PARAMS+=" --output ${DATA}/tau_pick_scalar_Nt${NT}_noisy.pickle"
    else
	echo "Running exact scalar case"
	PARAMS+=" --output ${DATA}/tau_pick_scalar_Nt${NT}.pickle"
    fi
    PARAMS+=" --scalar"
else
    if [[ -v ADD_NOISE ]]; then
	echo "Running noisy block case"
	PARAMS+=" --add-noise"
	PARAMS+=" --output ${DATA}/tau_pick_Nt${NT}_noisy.pickle"
    else
	echo "Running exact block case"
	PARAMS+=" --output ${DATA}/tau_pick_Nt${NT}.pickle"
    fi
fi

set -x

uv run python -u run_sweep.py ${PARAMS}
