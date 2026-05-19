import os
import sys
import pathlib

filepath = pathlib.Path(os.path.abspath(__file__))
root = filepath.parent.parent
sys.path.insert(0, str(root))

from mpmath import mp
mp.dps = 200

import json

import sdppy
from sdppy.solve_sdp import solve_sdp, default_params
from sdppy.test_solve_sdp import _read_test_sdp

from sdppy.io_utils import mpf_array_str

test_cases = [
    "test_sdp",
    "hamburger_sdp",
    "stieltjes_sdp",
    "noisy_stieltjes_sdp"
]

for name in test_cases:
    print(f"Generating tests for '{name}'")
    sdp = _read_test_sdp(name)
    q, history = solve_sdp(sdp, params=default_params)

    results = {
        "primal_objective" : mpf_array_str(history["primal_objective"][-1]),
        "dual_objective" : mpf_array_str(history["dual_objective"][-1])
    }

    save_loc = root / "test_data" / name / "results.json"

    with open(save_loc, 'w') as f:
        json.dump(results, f)
