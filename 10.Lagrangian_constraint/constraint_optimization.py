# Topic: Augmented Lagrangian Optimization with Constraints
# Description: 
# This Python code uses the Augmented Lagrangian method to perform constrained optimization on a simple function with both 
# equality and inequality constraints. The code includes functions for the objective function, constraints, and their gradients, 
# as well as a visualization of the optimization path using Matplotlib. An animation shows the optimization path as it converges 
# to a solution that satisfies the constraints.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Function Definitions:

# Objective function
def objective_function(x):
    # This function computes the value of the objective function to minimize.
    return (x[0] - 2)**2 + (x[1] - 1)**2

# Gradient of the objective function
def objective_gradient(x):
    # This function returns the gradient of the objective function.
    return np.array([2*(x[0] - 2), 2*(x[1] - 1)])

# Equality constraint
def equality_constraint(x):
    # This function defines the equality constraint: x1^2 + x2^2 = 1.
    return x[0]**2 + x[1]**2 - 1

# Gradient of the equality constraint
def equality_constraint_gradient(x):
    # This function returns the gradient of the equality constraint.
    return np.array([2*x[0], 2*x[1]])

# Inequality constraint
def inequality_constraint(x):
    # This function defines the inequality constraint: -(x1 + x2) <= 0.
    return -(x[0] + x[1])

# Gradient of the inequality constraint
def inequality_constraint_gradient(x):
    # This function returns the gradient of the inequality constraint.
    return np.array([-1, -1])

# Augmented Lagrangian function
def augmented_lagrangian(x, lambda_eq, lambda_ineq, rho_eq, rho_ineq):
    # This function calculates the Augmented Lagrangian value with equality and inequality constraints.
    obj = objective_function(x)
    eq_constr = equality_constraint(x)
    ineq_constr = inequality_constraint(x)
    eq_term = lambda_eq * eq_constr + (rho_eq / 2) * eq_constr**2
    ineq_term = lambda_ineq * max(0, ineq_constr) + (rho_ineq / 2) * max(0, ineq_constr)**2
    return obj + eq_term + ineq_term

# Gradient of the Augmented Lagrangian function
def augmented_lagrangian_gradient(x, lambda_eq, lambda_ineq, rho_eq, rho_ineq):
    # This function computes the gradient of the Augmented Lagrangian function.
    grad_obj = objective_gradient(x)
    eq_constr = equality_constraint(x)
    grad_eq = equality_constraint_gradient(x)
    ineq_constr = inequality_constraint(x)
    grad_ineq = inequality_constraint_gradient(x)
    eq_term = (lambda_eq + rho_eq * eq_constr) * grad_eq
    ineq_term = (lambda_ineq + rho_ineq * max(0, ineq_constr)) * grad_ineq
    return grad_obj + eq_term + ineq_term

# Backtracking line search for step size selection
def backtracking_line_search(x, direction, lambda_eq, lambda_ineq, rho_eq, rho_ineq):
    # This function performs a backtracking line search to find an appropriate step size along the descent direction.
    alpha = 1.0
    beta = 0.8
    c = 1e-4
    initial_value = augmented_lagrangian(x, lambda_eq, lambda_ineq, rho_eq, rho_ineq)
    initial_grad = augmented_lagrangian_gradient(x, lambda_eq, lambda_ineq, rho_eq, rho_ineq)
    initial_slope = np.dot(initial_grad, direction)
    while True:
        new_x = x + alpha * direction
        new_value = augmented_lagrangian(new_x, lambda_eq, lambda_ineq, rho_eq, rho_ineq)
        if new_value <= initial_value + c * alpha * initial_slope:
            return alpha
        alpha *= beta
        if alpha < 1e-10:
            return 1e-10

# Minimization using Augmented Lagrangian
def minimize_augmented_lagrangian(x, lambda_eq, lambda_ineq, rho_eq, rho_ineq, max_iter=200, epsilon=1e-8):
    # This function minimizes the Augmented Lagrangian function and tracks the optimization path.
    positions = [x.copy()]
    for _ in range(max_iter):
        grad = augmented_lagrangian_gradient(x, lambda_eq, lambda_ineq, rho_eq, rho_ineq)
        if np.linalg.norm(grad) < epsilon:
            break
        direction = -grad
        alpha = backtracking_line_search(x, direction, lambda_eq, lambda_ineq, rho_eq, rho_ineq)
        x = x + alpha * direction
        positions.append(x.copy())
    return x, np.array(positions)

# Main augmented Lagrangian method
def augmented_lagrangian_method(x0, max_outer_iter=30, max_inner_iter=200, tolerance=1e-9):
    # This function applies the Augmented Lagrangian method to optimize while satisfying the constraints.
    x = x0
    lambda_eq = 0.0
    lambda_ineq = 0.0
    rho_eq = 10.0
    rho_ineq = 10.0
    all_positions = [x.copy()]
    gamma_eq = 10.0
    gamma_ineq = 10.0
    for outer_iter in range(max_outer_iter):
        x, positions = minimize_augmented_lagrangian(x, lambda_eq, lambda_ineq, rho_eq, rho_ineq, max_inner_iter)
        all_positions.extend(positions)
        eq_viol = abs(equality_constraint(x))
        ineq_viol = max(0, inequality_constraint(x))
        print(f"Iteration {outer_iter}: x = {x}, eq_viol = {eq_viol:.2e}, ineq_viol = {ineq_viol:.2e}")
        if eq_viol < tolerance and ineq_viol < tolerance and np.linalg.norm(augmented_lagrangian_gradient(x, lambda_eq, lambda_ineq, rho_eq, rho_ineq)) < tolerance:
            break
        lambda_eq = lambda_eq + rho_eq * equality_constraint(x)
        lambda_ineq = max(0, lambda_ineq + rho_ineq * inequality_constraint(x))
        if eq_viol > tolerance:
            rho_eq = min(gamma_eq * rho_eq, 1e6)
        if ineq_viol > tolerance:
            rho_ineq = min(gamma_ineq * rho_ineq, 1e6)
    return np.array(all_positions)

# Visualization setup for contour and animation
# This part creates a contour plot of the objective function and animates the optimization path.
fig, ax = plt.subplots(figsize=(12, 10))
x1 = np.linspace(-3, 3, 200)
x2 = np.linspace(-3, 3, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)
for i in range(len(x1)):
    for j in range(len(x2)):
        Z[i, j] = objective_function(np.array([X1[i, j], X2[i, j]]))
contours = ax.contour(X1, X2, Z, levels=20, cmap='viridis')
plt.colorbar(contours, label='Objective Value')

# Plot constraints and initialize animation
theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
ax.plot(circle_x, circle_y, 'r--', label='Equality Constraint')
ax.plot([-2, 3], [2, -3], 'g--', label='Inequality Constraint')
ax.fill_between([-2, 3], [-2, -3], [3, 3], alpha=0.1, color='g')
initial_point = np.array([0.0, 0.0])
positions = augmented_lagrangian_method(initial_point)
path_line, = ax.plot([], [], 'ro-', label='Optimization Path', linewidth=1.5, markersize=3)
current_point, = ax.plot([], [], 'ko', markersize=10, label='Current Point')

# Animation functions
def init():
    # Initialization function for animation.
    path_line.set_data([], [])
    current_point.set_data([], [])
    return path_line, current_point

def animate(frame):
    # Update function for each frame in the animation.
    path_line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
    current_point.set_data([positions[frame, 0]], [positions[frame, 1]])
    return path_line, current_point

# Run the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(positions), interval=100, blit=True, repeat=True)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Augmented Lagrangian Optimization with Constraints')
ax.grid(True)
ax.legend()
ax.set_aspect('equal')
plt.show()
