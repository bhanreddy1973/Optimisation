# Title: Optimization of the Rosenbrock Function Using Adagrad with Momentum

# This code demonstrates optimizing the Rosenbrock function using an Adagrad algorithm with momentum.
# The Rosenbrock function, characterized by its narrow valley and steep gradients, is challenging for optimization.
# Adding momentum to Adagrad allows for improved convergence by accounting for past gradients, helping the optimizer move more efficiently through regions with varying gradients.
#------------------------------------------------------------------------------------------------------- 
 
# Key Code Components:
# - Imports:
#   numpy is imported for numerical operations, and matplotlib for visualization.
# - Rosenbrock Function:
#   Defines the Rosenbrock function as (1 - x)^2 + 100 * (y - x^2)^2, which has a global minimum at (1, 1).
#------------------------------------------------------------------------------------------------------- 

# - Gradient Calculation:
#   Calculates the gradient of the Rosenbrock function, which is used to direct each step of the optimizer.
# - Contour Plot Setup:
#   Sets up a contour plot in the range x = [-2, 2] and y = [-1, 3] for visualizing the Rosenbrock function's surface.
#------------------------------------------------------------------------------------------------------- 

# - Initialization:
#   Sets up initial conditions:
#   * current_pos: starting at [0.0, 0.0]
#   * gradient_vector and momentum_vector: to store accumulated gradients and momentum respectively.
#   * learning rate, epsilon, and beta: parameters controlling the optimization rate, stability, and momentum factor.
#------------------------------------------------------------------------------------------------------- 

# - Optimization Loop:
#   The loop runs for 500 iterations, performing the following steps in each:
#   * Calculates the gradient at the current position.
#   * Updates gradient_vector using the exponentially weighted average of past squared gradients.
#   * Updates momentum_vector with the exponentially weighted average of past gradients.
#   * Adjusts the current position using Adagrad’s adaptive step size and the momentum adjustment for efficient movement.
#   * Logs the position and function value at each step.
#------------------------------------------------------------------------------------------------------- 

# - Visualization:
#   The function’s contour plot is updated in each iteration to illustrate the optimization path in real-time.

# Final Output:
# - Displays the optimization path, showing how Adagrad with momentum converges toward the global minimum of the Rosenbrock function.


#------------------------------------------------------------------------------------------------------- 

# Code begins below:


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


x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock_function([X, Y])

current_pos = np.array([0.0, 0.0])
gradient_vector = np.zeros(2)
momentum_vector = np.zeros(2)
learning_rate = 0.01
epsilon = 1e-8
beta = 0.90

for _ in range(500):

    gradient = rosenbrock_gradient(current_pos)

    gradient_vector = beta * gradient_vector + (1 - beta) * gradient ** 2

    momentum_vector = beta * momentum_vector + (1 - beta) * gradient

    current_pos = current_pos - \
        (learning_rate / np.sqrt(gradient_vector + epsilon)) * momentum_vector

    z = rosenbrock_function([current_pos[0], current_pos[1]])

    print(
        f"Current position: {current_pos[0]:.6f}, {current_pos[1]:.6f}, {z:.6f}")

    plt.contour(X, Y, Z, levels=np.logspace(-2, 5, 15), cmap='plasma')
    plt.scatter(current_pos[0], current_pos[1], color='magenta', zorder=5)
    plt.grid(True)
    plt.pause(0.1)
    plt.clf()

plt.show()
