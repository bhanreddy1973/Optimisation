# Title: Optimizing the Rosenbrock Function with BFGS and Armijo Line Search

# Overview:
# This script implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization method to minimize the Rosenbrock function.
# The Armijo line search method dynamically adjusts the step size in each iteration, ensuring controlled, stable convergence.
#------------------------------------------------------------------------------------------------------- 

# Key Components:
# - Imports:
#   Uses numpy for array manipulations and matplotlib for visualizing the optimization path.
# - Rosenbrock Function:
#   Defines the Rosenbrock function, f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2, known for its challenging optimization landscape.
#------------------------------------------------------------------------------------------------------- 

# - Gradient Calculation:
#   Implements a function to compute the gradient of the Rosenbrock function, used to inform the direction of each optimization step.
# - Armijo Line Search:
#   Adapts the step size (alpha) in each iteration to satisfy the Armijo condition, balancing progress with stability.
#   * Parameters: alpha (initial step size), beta (step reduction factor), and c (condition scaling factor).
#------------------------------------------------------------------------------------------------------- 

# - BFGS Optimization Function:
#   Implements the BFGS algorithm, which updates an approximate inverse Hessian matrix to improve convergence.
#   * Starts from an initial point (x0) and iteratively applies BFGS updates to move towards the function minimum.
#   * Tracks each position to visualize the optimization path.
#   * Termination: Stops when the gradient magnitude falls below epsilon or max_iter is reached.
#------------------------------------------------------------------------------------------------------- 

# - Visualization:
#   Creates a contour plot of the Rosenbrock function's landscape, overlaid with the path taken by the BFGS optimization.
#   * Uses green to mark the starting point, red for the path, and blue for the converged point.
#   * Provides a color bar for function values and labels for easy interpretation.

# Execution:
# The code executes the optimization and displays a plot of the function’s contour with the BFGS path marked, illustrating the progress toward the minimum.

#------------------------------------------------------------------------------------------------------- 

# Code starts below:


import numpy as np
import matplotlib.pyplot as plt

def rosenbrock_function(point):
    x, y = point
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(point):
    x, y = point
    df_dx = -2 * (1 - x) - 400 * x * (y - x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

def armijo_line_search(f, grad, x, p, alpha=1.0, beta=0.5, c=1e-4):
    f_current = f(x)
    while f(x + alpha * p) > f_current + c * alpha * grad.dot(p):
        alpha *= beta
    return alpha

def bfgs_optimization(f, grad, x0, epsilon=1e-5, max_iter=1000):
    n = len(x0)
    x = x0
    B = np.eye(n)
    positions = [x.copy()]
    
    for i in range(max_iter):
        g = grad(x)
        
        if np.linalg.norm(g) < epsilon:
            break
            
        try:
            p = -np.linalg.solve(B, g)
        except np.linalg.LinAlgError:
            p = -g
            
        alpha = armijo_line_search(f, g, x, p)
        
        x_new = x + alpha * p
        
        s = x_new - x
        y = grad(x_new) - g
        
        if y.dot(s) > 0:
            rho = 1.0 / y.dot(s)
            B = B - np.outer(B.dot(s), s).dot(B) / (s.dot(B).dot(s)) + np.outer(y, y) / y.dot(s)
            
        x = x_new
        positions.append(x.copy())
        
        print(f'Iteration {i+1}: x = {x[0]:.6f}, y = {x[1]:.6f}, f(x,y) = {f(x):.6f}')
        
    return np.array(positions)

x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(lambda x, y: rosenbrock_function([x, y]))(X, Y)

initial_point = np.array([-2.0, 3.0])
positions = bfgs_optimization(rosenbrock_function, rosenbrock_gradient, initial_point)

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.logspace(-3, 3, 30), cmap='plasma')
plt.colorbar(label="Function Value (f)")

plt.plot(positions[:, 0], positions[:, 1], 'ro-', label="Optimization Path", linewidth=1.5, markersize=3)
plt.scatter(positions[-1, 0], positions[-1, 1], color='blue', s=100, label="Converged Point")
plt.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label="Starting Point")

plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("BFGS Optimization Path with Armijo Line Search")
plt.grid(True)
plt.show()