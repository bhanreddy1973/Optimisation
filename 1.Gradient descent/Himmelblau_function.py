# # optimization algorithm to minimize the Himmelblau function using gradient descent with an adaptive learning rate.

# Himmelblau Function
# The Himmelblau function is a well-known test function in optimization, defined as:
# f(x,y)=(x^2+y−11)^2+(x+y^2−7)^2
 
# This function has multiple local minima and a global minimum, making it useful for testing optimization algorithms. The global minimum occurs at the points (3, 2), (-2.805, 3.131), (-3.779, -3.283), and (3.584, -1.848), where the function value is zero.
# Key Components of the Code

# Function Definition:
# Himmelblau_Function(point): Computes the value of the Himmelblau function at a given point (x, y).
# Himmelblau_Function_gradient(point): Computes the gradient (partial derivatives) of the Himmelblau function with respect to x and y. This gradient indicates the direction of steepest ascent.

#-------------------------------------------------------------------------------------------------------
# Learning Rate Function:
# learning_rate_function(learning_rate, point): Evaluates the Himmelblau function at a new point obtained by moving in the direction of the gradient scaled by a proposed learning rate. This function helps in finding an appropriate step size for gradient descent.
# Grid Setup for Visualization:
# The code creates a grid using numpy's meshgrid function to evaluate the Himmelblau function over a specified range. This grid is used for plotting the surface of the function.

#-------------------------------------------------------------------------------------------------------
# Initialization:
# The starting position current_pos is initialized at (-2, 0), which is chosen arbitrarily and can be adjusted to see how it affects convergence.

#-------------------------------------------------------------------------------------------------------
# Gradient Descent Loop:
# The loop runs for a specified number of iterations (100 in this case). In each iteration:
# The gradient at the current position is calculated.
# An adaptive learning rate is determined using backtracking line search to ensure sufficient decrease in function value.
# The current position is updated by moving against the gradient scaled by the learning rate.
# The updated position and its corresponding function value are printed for monitoring convergence.
# The 3D surface plot of the Himmelblau function is updated to visualize the optimization process.

#-------------------------------------------------------------------------------------------------------
# Visualization:
# A 3D plot is created using matplotlib, where:
# The surface of the Himmelblau function is displayed.
# The current position in the optimization process is marked with a magenta dot.
# The plot updates dynamically to show how the algorithm approaches one of the minima.


import numpy as np
import matplotlib.pyplot as plt

def Himmelblau_Function(point):
    x, y = point
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def Himmelblau_Function_gradient(point):
    x, y = point
    df_dx = 2 * (x**2 + y - 11) * 2 * x + 2 * (x + y**2 - 7)
    df_dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * 2 * y
    return np.array([df_dx, df_dy])

def learning_rate_function(learning_rate, point):
    return Himmelblau_Function(point - learning_rate * Himmelblau_Function_gradient(point))

x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)
Z = Himmelblau_Function([X, Y])

current_pos = np.array([-2, 0])

fig = plt.figure()
ax = plt.subplot(projection="3d", computed_zorder=False)

for _ in range(100):
    alpha = 1
    beta = 0.4
    c1 = 0.0001

    gradient = Himmelblau_Function_gradient(current_pos)

    while learning_rate_function(alpha, current_pos) > \
            Himmelblau_Function(current_pos) + c1 * alpha * np.dot(gradient, gradient):
        alpha *= beta

    current_pos = current_pos - alpha * gradient

    z = Himmelblau_Function(current_pos)
    print(f"Current position: {current_pos[0]:.6f}, {current_pos[1]:.6f}, {z:.6f}")

    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.scatter(current_pos[0], current_pos[1], z,
               color='magenta', s=50, zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Himmelblau Function Optimization')
    plt.pause(0.001)

plt.show()
