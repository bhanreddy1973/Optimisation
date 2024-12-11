# Title: Conjugate Gradient Optimization with Backtracking Line Search on the Rosenbrock Function

# Overview:
# This script uses the nonlinear Conjugate Gradient (CG) method to optimize the Rosenbrock function. A backtracking line 
# search with Armijo condition is implemented to dynamically control the step size for each iteration.

# Key Components:
# - Imports:
#   * numpy: Array operations
#   * matplotlib.pyplot: For plotting the optimization path on the function's contour map
#------------------------------------------------------------------------------------------------------- 
# - Rosenbrock Function:
#   Defines the Rosenbrock function, a classic optimization test problem known for its narrow, curved valley and sharp 
#   minimum, making it challenging to minimize.
# - Gradient Computation:
#   Provides the gradient of the Rosenbrock function, essential for guiding the optimization direction.
#------------------------------------------------------------------------------------------------------- 

# - Backtracking Line Search:
#   Dynamically adjusts the step size (alpha) by ensuring each update meets the Armijo condition:
#   * Parameters: `rho` controls the step size reduction rate, `c` sets the sufficient decrease criterion.

#------------------------------------------------------------------------------------------------------- 

# - Conjugate Gradient Optimization (CG):
#   Uses the Polak-Ribière-Polyak formula for the nonlinear conjugate gradient update:
#   * Resets the conjugate direction periodically for stability, particularly when limited progress is detected.
#------------------------------------------------------------------------------------------------------- 

# - Visualization:
#   * Plots the Rosenbrock function's contours with the optimization path overlaid.
#   * Markers indicate the starting point (green), optimization path (red), and final converged point (blue).
#   * Provides a color bar for function values and labeled axes.
#------------------------------------------------------------------------------------------------------- 

# Execution:
# After defining the Rosenbrock function, gradient, line search, and CG algorithm, the code performs optimization 
# starting from a user-defined point. It visualizes the function landscape and CG path, illustrating progress 
# towards the function's minimum.

#------------------------------------------------------------------------------------------------------- 

# Code below:


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

def backtracking_line_search(f, grad, x, p, alpha=1.0, rho=0.5, c=1e-4):
    """
    Backtracking line search with Armijo condition.
    
    Parameters:
    -----------
    f : callable
        Objective function
    grad : callable
        Gradient function
    x : ndarray
        Current point
    p : ndarray
        Search direction
    alpha : float
        Initial step size
    rho : float
        Step size reduction factor
    c : float
        Sufficient decrease parameter
    """
    f_x = f(x)
    grad_x = grad(x)
    slope = grad_x.dot(p)
    
    while f(x + alpha * p) > f_x + c * alpha * slope:
        alpha *= rho
        
    return alpha

def conjugate_gradient(f, grad, x0, epsilon=1e-5, max_iter=1000):
    """
    Nonlinear Conjugate Gradient optimization with Polak-Ribière-Polyak formula.
    
    Parameters:
    -----------
    f : callable
        Objective function
    grad : callable
        Gradient function
    x0 : ndarray
        Initial point
    epsilon : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
    """
    x = x0
    positions = [x.copy()]
    
    g = grad(x)
    d = -g  # Initial search direction
    
    for i in range(max_iter):
        g_norm = np.linalg.norm(g)
        if g_norm < epsilon:
            break
            
        # Perform line search
        alpha = backtracking_line_search(f, grad, x, d)
        
        # Update position
        x_new = x + alpha * d
        
        # Compute new gradient
        g_new = grad(x_new)
        
        # Polak-Ribière-Polyak beta
        beta = max(0, g_new.dot(g_new - g) / (g.dot(g) + 1e-10))
        
        # Update search direction
        d = -g_new + beta * d
        
        # Update for next iteration
        x = x_new
        g = g_new
        
        positions.append(x.copy())
        print(f'Iteration {i+1}: x = {x[0]:.6f}, y = {x[1]:.6f}, f(x,y) = {f(x):.6f}')
        
        # Reset conjugate direction if not making good progress
        if i % 2 == 0 and i > 0:  # Reset every 2 iterations
            if f(x) > f(positions[-2]):
                d = -g
                print("Resetting conjugate direction")
    
    return np.array(positions)

# Set up the visualization
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(lambda x, y: rosenbrock_function([x, y]))(X, Y)

# Run Conjugate Gradient optimization
initial_point = np.array([-2.0, 3.0])
positions = conjugate_gradient(rosenbrock_function, rosenbrock_gradient, initial_point)

# Create the visualization
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.logspace(-3, 3, 30), cmap='plasma')
plt.colorbar(label="Function Value (f)")

# Plot optimization path
plt.plot(positions[:, 0], positions[:, 1], 'ro-', label="Optimization Path", linewidth=1.5, markersize=3)
plt.scatter(positions[-1, 0], positions[-1, 1], color='blue', s=100, label="Converged Point")
plt.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label="Starting Point")

plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Conjugate Gradient Optimization Path")
plt.grid(True)
plt.show()