
# Title: Newton's Method Optimization on Rosenbrock Function with Damped Hessian

# This code applies Newton's method with a damped Hessian to minimize the Rosenbrock function.
# The optimization utilizes the Hessian matrix to calculate step directions, and damping is introduced
# to prevent overstepping and improve stability during convergence.

#------------------------------------------------------------------------------------------------------- 

# Key Components:
# - Rosenbrock Function:
#   The Rosenbrock function is a widely used test function in optimization. It is defined as:
#   f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2.
#------------------------------------------------------------------------------------------------------- 

#   It has a narrow, curved valley containing the global minimum at (1, 1).
# - Gradient Calculation:
#   The gradient of the Rosenbrock function is computed, which provides the direction of steepest ascent/descent.
#------------------------------------------------------------------------------------------------------- 

# - Hessian Matrix:
#   The Hessian matrix is used to approximate the curvature of the function. It contains second-order partial derivatives.
#   Damping is applied to the Hessian matrix to control large steps during the optimization process.

#------------------------------------------------------------------------------------------------------- 
# - Newton's Method:
#   In each iteration, the current position is updated by solving the linear system H * delta = -grad, where H is the 
#   damped Hessian matrix and grad is the gradient. The step delta is used to update the position.

#------------------------------------------------------------------------------------------------------- 
# - Visualization:
#   A contour plot of the Rosenbrock function is created, and the optimization path is shown. The final converged 
#   point is marked in blue, and the path taken by the optimization process is shown in red.

#------------------------------------------------------------------------------------------------------- 
# Final Output:
# - The plot visualizes the trajectory of Newton's method as it converges towards the minimum of the Rosenbrock function.
# - The optimization path is drawn in red, with the final converged point marked in blue.

#------------------------------------------------------------------------------------------------------- 
#code : 

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


def rosenbrock_hessian_damped(point, a=1, b=100, damping_factor=0.1):
    x, y = point
    hess_xx = 2 - 4 * b * y + 12 * b * x**2
    hess_xy = -4 * b * x
    hess_yy = 2 * b

    hessian = np.array([[hess_xx, hess_xy],
                        [hess_xy, hess_yy]])

    hessian_damped = hessian + damping_factor * np.eye(2)
    return hessian_damped


x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(lambda x, y: rosenbrock_function([x, y]))(X, Y)

current_pos = np.array([-2, 3])
positions = [current_pos.copy()]

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(-3, 3, 30), cmap='plasma')
plt.colorbar(label="Function Value (f)")

epsilon = 1e-5
while np.linalg.norm(rosenbrock_gradient(current_pos)) > epsilon:
    grad = rosenbrock_gradient(current_pos)
    hess = rosenbrock_hessian_damped(current_pos)

    delta = np.linalg.solve(hess, -grad)

    print(
        f'x = {current_pos[0]}, y = {current_pos[1]}, z = {rosenbrock_function(current_pos)}')
    current_pos = current_pos + delta
    positions.append(current_pos.copy())

positions = np.array(positions)
plt.plot(positions[:, 0], positions[:, 1], 'ro-', label="Optimization Path")
plt.scatter(positions[-1, 0], positions[-1, 1],
            color='blue', label="Converged Point")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Newton's Method Optimization Path")
plt.show()
