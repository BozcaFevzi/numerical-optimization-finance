import numpy as np
from scipy.optimize import minimize, Bounds
import nlopt


# ==============================================================================
# DONNÉES DU PROJET
# ==============================================================================

# Vecteur des rendements espérés E(R)
ER = np.array([
    3.66, 1.33, 2.51, 2.24, 2.62, 2.03, 1.91, 1.39, 2.61, 3.58, 5.23, 8.36
])

# Matrice de Covariance
Cov = np.array([
    [73.73, 40.79, 39.46, 38.01, 49.40, 58.51, 46.08, 42.26, 12.44, -1.22, -2.31, 131.17],
    [40.79, 39.54, 32.15, 34.43, 34.25, 36.88, 33.11, 26.50, 19.06, -3.13, 0.53, 136.78],
    [39.46, 32.15, 54.98, 31.62, 33.80, 41.20, 26.48, 29.76, 25.34, -2.90, -3.62, 151.82],
    [38.01, 34.43, 31.62, 39.61, 35.73, 33.64, 38.12, 26.00, 13.23, -0.08, -5.61, 73.96],
    [49.40, 34.25, 33.80, 35.73, 68.10, 36.88, 37.02, 31.64, 3.33, 14.42, 1.43, -0.12],
    [58.51, 36.88, 41.20, 33.64, 36.88, 66.93, 40.89, 39.08, 22.07, -13.71, -14.96, 210.26],
    [46.08, 33.11, 26.48, 38.12, 37.02, 40.89, 75.40, 38.58, 19.37, 19.01, 30.95, 52.63],
    [42.26, 26.50, 29.76, 26.00, 31.64, 39.08, 38.58, 47.35, 17.02, -3.53, -8.28, 111.37],
    [12.44, 19.06, 25.34, 13.23, 3.33, 22.07, 19.37, 17.02, 110.87, -90.65, -50.97, 360.36],
    [-1.22, -3.13, -2.90, -0.08, 14.42, -13.71, 19.01, -3.53, -90.65, 242.03, 196.18, -507.15],
    [-2.31, 0.53, -3.62, -5.61, 1.43, -14.96, 30.95, -8.28, -50.97, 196.18, 428.23, -504.38],
    [131.17, 136.78, 151.82, 73.96, -0.12, 210.26, 52.63, 111.37, 360.36, -507.15, -504.38, 2863.07]
])

PHI = 5.0
N = 12
DIM = N - 1

# ==============================================================================
# (Question 1)
# Construction de J_tilde(x) = 0.5 * x'Hx + c'x + const
# ==============================================================================

H = np.zeros((DIM, DIM))
c = np.zeros(DIM)
Cov_nn = Cov[DIM, DIM]
En = ER[DIM]

# Construction de H et c
for i in range(DIM):
    c[i] = -PHI * (ER[i] - En) + 2 * Cov[i, DIM] - 2 * Cov_nn
    for j in range(DIM):
        val = Cov[i, j] - Cov[i, DIM] - Cov[j, DIM] + Cov_nn
        H[i, j] = 2 * val

J_const = -PHI * En + Cov_nn

def func_J(x):
    return 0.5 * x @ H @ x + c @ x + J_const

def grad_J(x):
    return H @ x + c


# ==============================================================================
# ALGORITHMES DU COURS
# ==============================================================================

def algorithm_3_zoom(alpha_lo, alpha_hi, phi, phi_prime, phi0, phi_prime0, c1, c2):
    while True:
        alpha_j = 0.5 * (alpha_lo + alpha_hi)
        phi_j = phi(alpha_j)
        if (phi_j > phi0 + c1 * alpha_j * phi_prime0) or (phi_j >= phi(alpha_lo)):
            alpha_hi = alpha_j
        else:
            phi_prime_j = phi_prime(alpha_j)
            if abs(phi_prime_j) <= -c2 * phi_prime0:
                return alpha_j
            if phi_prime_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j


def algorithm_2_linesearch(x_k, p_k, func, grad, c1=0.0001, c2=0.9):
    # Variables
    alpha_max = 2.0
    alpha_precedent = 0.0
    alpha_i = 1.0

    phi0 = func(x_k)
    phi_prime0 = np.dot(grad(x_k), p_k)

    def phi(alpha):
        return func(x_k + alpha * p_k)

    def phi_prime(alpha):
        return np.dot(grad(x_k + alpha * p_k), p_k)

    phi_prev = phi0
    i = 1
    max_iter = 100
    while i < max_iter:
        phi_i = phi(alpha_i)
        if (phi_i > phi0 + c1 * alpha_i * phi_prime0) or ((i > 1) and (phi_i >= phi_prev)):
            return algorithm_3_zoom(alpha_precedent, alpha_i, phi, phi_prime, phi0, phi_prime0, c1, c2)

        phi_prime_i = phi_prime(alpha_i)
        if abs(phi_prime_i) <= -c2 * phi_prime0:
            return alpha_i

        if phi_prime_i >= 0:
            return algorithm_3_zoom(alpha_i, alpha_precedent, phi, phi_prime, phi0, phi_prime0, c1, c2)

        alpha_precedent = alpha_i
        phi_prev = phi_i
        alpha_i = min(2 * alpha_i, alpha_max)
        i += 1
    return alpha_i


# ==============================================================================
# ALGORITHMES D'OPTIMISATION (Questions 2, 3 et 4)
# ==============================================================================

def question_2_steepest_descent():
    print("\n--- Question 2 : Steepest Descent ---")
    x = np.zeros(DIM)
    tol = 0.000001
    max_iter = 5000

    for k in range(max_iter):
        gk = grad_J(x)
        if np.linalg.norm(gk) < tol:
            print(f"Convergence atteinte à l'itération {k}")
            break

        pk = -gk
        alpha_k = algorithm_2_linesearch(x, pk, func_J, grad_J)
        x = x + alpha_k * pk

    print(f"J_opt = {func_J(x):.6f}")
    return x


def question_3_newton():
    print("\n--- Question 3 : Newton Method ---")
    x = np.zeros(DIM)
    tol = 0.000001
    L = np.linalg.cholesky(H)
    gk = grad_J(x)
    k = 0
    while np.linalg.norm(gk) > tol:
        y = np.linalg.solve(L, -gk)
        pk = np.linalg.solve(L.T, y)
        alpha_k = algorithm_2_linesearch(x, pk, func_J, grad_J)
        x = x + alpha_k * pk
        gk = grad_J(x)
        k += 1
        if k > 100:
            break

    print(f"Convergence en {k} itération(s).")
    print(f"J_opt = {func_J(x):.6f}")
    return x


def question_4_cg():
    print("\n--- Question 4 : Linear Conjugate Gradient (Alg 7) ---")
    x = np.zeros(DIM)
    r = H @ x + c  # gradient g = Hx + c
    p = -r
    k = 0

    while np.linalg.norm(r) > 0.000001 and k < DIM + 5:
        Ap = H @ p
        denom = np.dot(p, Ap)
        if denom <= 0:
            break
        alpha_k = np.dot(r, r) / denom

        x = x + alpha_k * p
        r_new = r + alpha_k * Ap
        beta_next = np.dot(r_new, r_new) / np.dot(r, r)

        p = -r_new + beta_next * p
        r = r_new
        k += 1

    print(f"Convergence en {k} itérations.")
    print(f"J_opt = {func_J(x):.6f}")
    return x


# ==============================================================================
# QUESTION 5 : OPTIMIZATION LIBRARIE (SciPy + NLopt)
# ==============================================================================

def question_5_scipy_unconstrained():
    print("\n--- Question 5a : SciPy - Unconstrained (BFGS) ---")
    x0 = np.zeros(DIM)
    result = minimize(
        func_J, 
        x0,
        method='BFGS',
        jac=grad_J,
        options={'disp': False, 'maxiter': 5000}
    )
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"J_opt = {result.fun:.6f}")
    print(f"Gradient norm = {np.linalg.norm(result.jac):.6e}")
    return result.x, result.fun


def question_5_scipy_constrained():
    print("\n--- Question 5b : SciPy - Constrained (SLSQP) ---")
    x0 = np.ones(DIM) / N

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    bounds = Bounds(0.0, 1.0)

    result = minimize(
        func_J, 
        x0,
        method='SLSQP',
        jac=grad_J,
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 10**(-10), 'maxiter': 5000}
    )
    print(f"J_opt = {result.fun:.6f}")
    print(f"Sum of weights = {np.sum(result.x):.6f}")
    print(f"Min weight = {np.min(result.x):.6f}")
    return result.x, result.fun


def question_5_nlopt_unconstrained():
    print("\n--- Question 5c : NLopt - Unconstrained (LBFGS) ---")
    opt = nlopt.opt(nlopt.LD_LBFGS, DIM)

    def f(x, grad):
        if grad.size > 0:
            grad[:] = grad_J(x)
        return func_J(x)

    opt.set_min_objective(f)
    opt.set_ftol_rel(10**(-10))
    opt.set_maxeval(5000)

    x0 = np.zeros(DIM)
    x_opt = opt.optimize(x0)
    minf = opt.last_optimum_value()

    print(f"Result code: {opt.last_optimize_result()}")
    print(f"J_opt = {minf:.6f}")
    print(f"Evaluations: {opt.get_numevals()}")
    print(f"Gradient norm = {np.linalg.norm(grad_J(x_opt)):.6e}")
    return x_opt, minf


# ==============================================================================
# QUESTION 6 : CONTRAINT (UNIQUEMENT AVEC SCIPY)
# ==============================================================================

def question_6_scipy():
    print("\n--- Question 6 : Résolution contrainte (SciPy SLSQP) ---")
    x0 = np.ones(DIM) / N

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    bounds = Bounds(0.0, 1.0)

    result = minimize(
        func_J, 
        x0,
        method='SLSQP',
        jac=grad_J,
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 10**(-10), 'maxiter': 5000}
    )
    print(f"J_opt = {result.fun:.6f}")
    print(f"Sum of weights = {np.sum(result.x):.6f}")
    print(f"Min weight = {np.min(result.x):.6f}")
    return result.x, result.fun


# ==============================================================================
# TABLEAU COMPARATIF DES RÉSULTATS
# ==============================================================================

def comparison_table():

    results = {}

    # Question 2: Steepest Descent
    x2 = question_2_steepest_descent()
    results["Q2: Steepest Descent"] = func_J(x2)

    # Question 3: Newton
    x3 = question_3_newton()
    results["Q3: Newton"] = func_J(x3)

    # Question 4: CG
    x4 = question_4_cg()
    results["Q4: Conjugate Gradient"] = func_J(x4)

    # Question 5a: SciPy (BFGS)
    x_scipy_unc, f_scipy_unc = question_5_scipy_unconstrained()
    results["Q5a: SciPy (BFGS)"] = f_scipy_unc

    # Question 5b: SciPy (SLSQP)
    x_scipy_con, f_scipy_con = question_5_scipy_constrained()
    results["Q5b: SciPy (SLSQP)"] = f_scipy_con

    # Question 5c: NLopt (LBFGS)
    x_nlopt_unc, f_nlopt_unc = question_5_nlopt_unconstrained()
    results["Q5c: NLopt (LBFGS)"] = f_nlopt_unc

    # Affichage du tableau
    print("\n" + "="*80)
    print(f"{'Algorithm':<35} {'J_opt':>18}")
    print("="*80)

    for algo, value in results.items():
        print(f"{algo:<35} {value:>18.6f}")

    print("="*80)

    if len(results) > 0:
        min_val = min(results.values())
        max_val = max(results.values())
        print(f"\nMeilleur résultat : {min_val:.6f}")
        print(f"Pire résultat    : {max_val:.6f}")
        print(f"Écart maximal    : {max_val - min_val:.6e}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    comparison_table()
    question_6_scipy()
