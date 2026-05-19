import numpy as np
from dataclasses import dataclass
from mpmath import mp

from .mp_array import MPArray

from .sdp import SDP, BlockMatrix

# All notation is from https://arxiv.org/pdf/1502.02033

@dataclass
class SDPSolveParameters:
    # \Omega_P, \Omega_D
    initialMatrixScalePrimal: mp.mpf
    initialMatrixScaleDual: mp.mpf

    # \beta_feasible, \beta_infeasible
    feasibleCenterinParameter: mp.mpf
    infeasibleCenterinParameter: mp.mpf

    stepLengthReduction: mp.mpf

    # Detect a stall if step_size < step_min
    stepMin: mp.mpf
    # If stalling, try to force mu -> stallIncrement * mu for stallSteps steps
    stallIncrement: mp.mpf
    maxStalls: int
    stallSteps: int

    # If lock_step=True, then force the primal and dual step sizes to
    # match by taking the minimum.
    lock_step: bool

    dualityGapThreshold: mp.mpf
    primalErrorThreshold: mp.mpf
    dualErrorThreshold: mp.mpf

    maxiter: int

default_params = SDPSolveParameters(mp.mpf("1e20"), mp.mpf("1e20"),
                                    mp.mpf("0.1"), mp.mpf("0.3"),
                                    mp.mpf("0.7"),
                                    stepMin=mp.mpf("1e-3"),
                                    stallIncrement=mp.mpf("1.5"),
                                    maxStalls=1,
                                    stallSteps=10,
                                    lock_step=True,
                                    dualityGapThreshold=mp.mpf("1e-30"),
                                    primalErrorThreshold=mp.mpf("1e-30"),
                                    dualErrorThreshold=mp.mpf("1e-30"),
                                    maxiter=1000)

def _transpose(A):
    return np.swapaxes(A, axis1=-1, axis2=-2)

def _identity_like(X):
    blocks = []
    for X_b in X.blocks:
        blocks.append(np.eye(*X_b.shape))
    return BlockMatrix(blocks)

def _identity_shift(q, alpha):
    x, X, y, Y = q
    X = X + alpha * _identity_like(X)
    Y = Y + alpha * _identity_like(Y)
    return x, X, y, Y

# Computes the maximum value of alpha such that X + alpha * dX is
# still positive-definite
def max_stepsize(X: BlockMatrix, dX: BlockMatrix):
    result = None
    for X_b, dX_b in zip(X.blocks, dX.blocks):
        L = np.linalg.cholesky(X_b)
        Linv = np.linalg.inv(L)
        eigvals = np.linalg.eigvalsh(Linv @ dX_b @ _transpose(Linv))
        alpha = np.min(eigvals)[()]
        if result is None:
            result = alpha
        else:
            result = min(result, alpha)
    result = -1/result
    if result < 0:
        return np.inf
    return result

def initialize_free_variables(sdp, X: BlockMatrix, Y: BlockMatrix):
    B = np.concatenate([eq.B for eq in sdp.equalities], axis=0)
    b = sdp.objective_vector
    c = np.concatenate([eq.c for eq in sdp.equalities], axis=0)

    ident_X = _identity_like(X)
    ident_Y = _identity_like(Y)
    g = sdp.compute_schur_complement(ident_X, ident_Y)

    x = np.linalg.solve(g.as_matrix(), sdp.tr_A_vec(X))
    dx, _, _, _ = np.linalg.lstsq(B.T, b - B.T @ x)
    x = x + dx

    y, _, _, _ = np.linalg.lstsq(B, c - sdp.tr_A_vec(Y))

    return x, y

def newton_step(sdp: SDP, solve_params: SDPSolveParameters,
                x: MPArray, X: BlockMatrix, y: MPArray, Y: BlockMatrix,
                *, step_type="mehrota", mu_target=None):
    assert len(X.blocks) == len(Y.blocks)

    XY = X @ Y
    ident = BlockMatrix([np.eye(X_i.shape[-1]) for X_i in X.blocks])
    X_inv = X.inverse()
    nrow = sum(X_i.shape[0] for X_i in X.blocks)
    mu = XY.trace() / nrow

    q = (x, X, y, Y)

    residues = sdp.compute_residues(q)
    P, p, d = residues

    ## Compute schur complement
    S_factor = sdp.factor_schur_complement(X_inv, Y)

    def _solve_schur(R):
        dx, dy = sdp.solve_schur_complement(X_inv, Y, residues, R, S_factor)
        assert dx.shape == x.shape
        assert dy.shape == y.shape

        # Eq. 2.27
        dX = P + sdp.combine_A(dx)
        # Eq. 2.28
        dY = X_inv @ (R - dX @ Y)

        dY = 0.5 * (dY + dY.T)

        return dx, dX, dy, dY

    if step_type == 'mehrota':
        mu_p = mu * solve_params.infeasibleCenterinParameter
        R_p = mu_p * ident - XY
        _, dX_p, _, dY_p = _solve_schur(R_p)

        ## Mehrota corrector step
        # Eqs. 2.37-2.38 and above
        r = ((X + dX_p) @ (Y + dY_p)).trace() / nrow / mu
        beta = r**2 if r < 1.0 else r
        beta_c = np.maximum(solve_params.infeasibleCenterinParameter,
                            beta)
        mu_c = mu * beta_c

        R_c = mu_c * ident - XY - dX_p @ dY_p
        dx, dX, dy, dY = _solve_schur(R_c)
    elif step_type == "fixed_mu":
        assert mu_target is not None
        R = mu_target * ident - XY
        dx, dX, dy, dY = _solve_schur(R)

    ## Compute step sizes
    alpha_dX = max_stepsize(X, dX)
    alpha_dY = max_stepsize(Y, dY)

    # Eq. 2.39
    gamma = solve_params.stepLengthReduction
    alpha_primal = min(gamma * alpha_dX, 1.0)
    alpha_dual = min(gamma * alpha_dY, 1.0)
    if solve_params.lock_step:
        alpha = min(alpha_primal, alpha_dual)
        alpha_primal = alpha
        alpha_dual = alpha

    ## Rescale updates to step size
    dX = alpha_primal * dX
    dx = alpha_primal * dx
    dY = alpha_dual * dY
    dy = alpha_dual * dy

    P_err = np.max([np.max(np.abs(P_b))
                    for P_b in P.blocks])[()]
    p_err = np.linalg.norm(p, ord=np.inf)[()]
    primal_error = max(P_err, p_err)
    dual_error = np.linalg.norm(d, ord=np.inf)[()]
    primal_objective = sdp.primal_objective(q)
    dual_objective = sdp.dual_objective(q)
    duality_gap = (primal_objective - dual_objective) / max(1, abs(primal_objective) + abs(dual_objective))

    info = {
        "primal_error" : primal_error,
        "dual_error" : dual_error,
        "primal_objective" : primal_objective,
        "dual_objective" : dual_objective,
        "mu" : mu,
        "step_primal": alpha_primal,
        "step_dual": alpha_dual,
        "duality_gap": duality_gap,
    }

    return (dx, dX, dy, dY), info

def solve_sdp(sdp: SDP, params: SDPSolveParameters,
              verbose=False, prefix=None, q0=None):
    stall_count = 0
    if prefix is None:
        prefix = ""

    q = sdp.initialize(params.initialMatrixScalePrimal,
                       params.initialMatrixScaleDual)
    if q0 is not None:
        q = q0
    history = {}

    headers = [
        ('mu', 11),
        ('P_obj', 11),
        ('D_obj', 11),
        ('P_err', 11),
        ('D_err', 11),
        ('P_step', 11),
        ('D_step', 11),
        ('gap', 11),
    ]
    line = "it   "
    for header, width in headers:
        header = header.ljust(width)
        line = line + header
    if verbose:
        print(line)
        print("-" * len(line))

    solution_found = False
    num_fixed_mu_steps = 0
    mu_target = None
    for i in range(params.maxiter):
        if num_fixed_mu_steps > 0:
            step_type = "fixed_mu"
            num_fixed_mu_steps = num_fixed_mu_steps - 1
        else:
            step_type = "mehrota"
        dq, info = newton_step(sdp, params, *q,
                               step_type=step_type,
                               mu_target=mu_target)
        q = tuple(q_i + dq_i for q_i, dq_i in zip(q, dq))

        if i == 0:
            for key in info.keys():
                history[key] = []
        for key, value in info.items():
            history[key].append(value)

        metrics = [
            ('mu', 3, 11),
            ('primal_objective', 3, 11),
            ('dual_objective', 3, 11),
            ('primal_error', 3, 11),
            ('dual_error', 3, 11),
            ('step_primal', 3, 11),
            ('step_dual', 3, 11),
            ('duality_gap', 3, 11),
        ]

        line = f"{i:<4} "
        for name, ndigit, width in metrics:
            value = float(info[name])
            s = "{value:." + str(ndigit) + "e}"
            s = s.format(value=value)
            s = s.ljust(width)
            line += s.format(value=value)
        if verbose:
            print(prefix + line)

        cond1 = info['duality_gap'] < params.dualErrorThreshold
        cond2 = info['primal_error'] < params.primalErrorThreshold
        cond3 = info['dual_error'] < params.dualErrorThreshold

        if cond1 and cond2 and cond3:
            solution_found = True
            break

        if cond2 and cond3:
            step_type = "mehrota"

        stalling_P = info['step_primal'] < params.stepMin
        stalling_D = info['step_dual'] < params.stepMin
        if (stalling_P or stalling_D) and num_fixed_mu_steps == 0:
            stall_count = stall_count + 1
            if stall_count > params.maxStalls:
                break
            mu_target = params.stallIncrement * info['mu']
            num_fixed_mu_steps = params.stallSteps
            if verbose:
                print(prefix + "WARNING: stall detected")

    if verbose and solution_found:
        print(prefix + "Solution found!")
        print(prefix + "primal_objective =", str(info["primal_objective"]))
        print(prefix + "dual_objective =", str(info["dual_objective"]))
        print(prefix + "duality_gap =", str(info["duality_gap"]))
        print(prefix + "primal_error =", str(info["primal_error"]))
        print(prefix + "dual_error =", str(info["dual_error"]))
    elif verbose:
        raise RuntimeError("Solution not found")

    history = {key: np.array(value, dtype=object)
               for key, value in history.items()}

    return q, history
