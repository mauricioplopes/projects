# Gradient Descent Optimization – Linear Algebra MO431A

This repository contains the implementation of gradient descent techniques applied to a two-dimensional nonlinear optimization problem, as part of the coursework for MO431A – Linear Algebra, offered at UNICAMP (State University of Campinas), Institute of Computing.

[Code](MO431A-Tarefa2.ipynb)

## Objective

Minimize the following two-variable nonlinear function using gradient descent:

$f(x_1, x_2) = (1 - x_1)^2 + 100(x_2 - x_1^2)^2$


This is the well-known **Rosenbrock function**, a classic test case for optimization algorithms due to its narrow, curved valley toward the global minimum.

## Gradient Computation

The gradients used for explicit gradient descent are:


$frac{d}{dx_1} f(x_1, x_2) = 2(200x_1^3 - 200x_1x_2 + x_1 - 1)$



$frac{d}{dx_2} f(x_1, x_2) = 200(x_2 - x_1^2)$


## Implementation Details

- Gradient Descent from scratch with:
  - Different learning rates
  - Convergence criteria
  - Tracking of iteration paths
- Visualization:
  - 3D surface plots of the Rosenbrock function
  - Contour plots with gradient descent steps
- Analysis of:
  - Learning rate impact
  - Convergence behavior

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

You can install the dependencies using pip:

```bash
pip install numpy matplotlib
