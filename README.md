# Quantitative Portfolio Optimization: Numerical Methods

This repository contains the implementation of various numerical optimization algorithms applied to a portfolio allocation problem. This project was developed as part of the "Numerical Methods for Optimization" course in the M2 MMMEF (Modélisation et Méthodes Mathématiques en Economie et Finance) program at Université Paris 1 Panthéon-Sorbonne.

## 📌 Project Overview

The objective is to determine the optimal asset allocation for a portfolio of 12 assets by minimizing an objective function $J(x)$ that balances risk minimization (variance) and expected return maximization. 

The problem is solved in two phases:
1.  **Unconstrained Optimization**: Reformulating the problem by substituting the equality constraint and solving it using custom-built descent methods.
2.  **Constrained Optimization**: Incorporating budget and no-short-selling (positivity) constraints using industry-standard optimization libraries.

## 🧮 Mathematical Formulation

The initial problem seeks to minimize the following objective function:

$$J(x)=\frac{1}{2}Var(R)-\phi E(R)$$

Subject to the constraints:
*   $\sum_{i=1}^{N}x_{i}=1$ (Budget constraint)
*   $x_{i}\ge0$ (No-short-selling constraint)

Where $N=12$ assets and the risk aversion parameter $\phi=5$. 
By substituting $x_{N}=1-\sum_{i=1}^{N-1}x_{i}$, the unconstrained reformulated objective function becomes strictly convex, defined by a positive-definite Hessian matrix $H$ derived from the asset covariance matrix.

## 💻 Implemented Methods

The project compares custom implementations of foundational optimization algorithms against robust library solvers:

### Custom Implementations (From scratch)
*   **Steepest Descent**: Includes a custom Line Search satisfying Wolfe conditions. 
*   **Newton's Method**: Utilizes exact second-order approximation since the objective function is quadratic.
*   **Linear Conjugate Gradient**: Solves the equivalent linear system $Hx=-c$ without requiring Hessian storage.

### Library Solvers
*   **SciPy**: `BFGS` for unconstrained and `SLSQP` (Sequential Least Squares Programming) for constrained optimization.
*   **NLopt**: `L-BFGS` for unconstrained optimization.

## 📊 Key Findings

*   **Algorithm Efficiency**: For the unconstrained quadratic problem, Newton's method is the most efficient, converging in a single iteration ($J(x^*) \approx 3.79$). The Conjugate Gradient method also performs exceptionally well, converging in 14 iterations.
*   **Conditioning Issues**: The Steepest Descent method struggled to converge even after 5000 iterations due to the high correlation between assets, which leads to a poorly conditioned Hessian matrix.
*   **Impact of Constraints**: Introducing the no-short-selling constraint (preventing negative weights) significantly increases the cost function to $J(x^*) \approx 8.53$ (solved via SciPy's SLSQP). This highlights the financial "cost" of long-only restrictions, forcing the portfolio onto a less favorable efficient frontier.

## 👨‍💻 Author

Fevzi BOZCA
Master 2 MMMEF - Université Paris 1 Panthéon-Sorbonne
