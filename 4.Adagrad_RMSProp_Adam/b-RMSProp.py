# Title: Minimizing the Rosenbrock Function Using RMSProp

# This code optimizes the Rosenbrock function using RMSProp, an adaptive learning rate method that includes a decay factor.
# RMSProp improves on Adagrad by preventing the learning rate from decaying too quickly, leading to more stable convergence.


#------------------------------------------------------------------------------------------------------- 

# Key Code Components:

# - Imports:
#   Uses numpy for numerical calculations and matplotlib for real-time visualization.
# - Rosenbrock Function:
#   Defines the Rosenbrock function as (1 - x)^2 + 100 * (y - x^2)^2, which has its global minimum at (1, 1).
#------------------------------------------------------------------------------------------------------- 

# - Gradient Calculation:
#   Implements a function to calculate the gradient of the Rosenbrock function, which guides each update step.
# - Contour Plot Setup:
#   Uses meshgrid to create contour lines for a visual representation of the Rosenbrock function’s landscape, covering the range x = [-2, 2] and y = [-1, 3].
#------------------------------------------------------------------------------------------------------- 

# - Initialization:
#   * Sets `current_pos` as the starting position at [0.0, 0.0].
#   * Initializes `gradient_vector` for RMSProp’s running average of squared gradients.
#   * Defines `learning_rate`, `epsilon` (for stability), and `beta` (the decay rate controlling the influence of past gradients).

#------------------------------------------------------------------------------------------------------- 
# - Optimization Loop:
#   Runs for 500 iterations, performing the following operations each time:
#   * Computes the gradient of the Rosenbrock function at `current_pos`.
#   * Updates `gradient_vector` using an exponentially weighted average with decay `beta`.
#   * Adjusts `current_pos` by scaling the gradient step according to the adaptive learning rate derived from `gradient_vector`.
#   * Prints the current position and Rosenbrock function value to monitor progress.
#------------------------------------------------------------------------------------------------------- 

# - Visualization:
#   Updates a contour plot at each step, marking the current position in magenta to illustrate the optimization path.

# Final Output:
# - Shows the optimization path as RMSProp converges towards the Rosenbrock function’s minimum, allowing visual insight into the optimization process.

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
learning_rate = 0.01
epsilon = 1e-8
beta = 0.90

for _ in range(500):

    gradient = rosenbrock_gradient(current_pos)

    gradient_vector = beta * gradient_vector + (1 - beta) * gradient ** 2

    current_pos = current_pos - \
        (learning_rate / np.sqrt(gradient_vector + epsilon)) * gradient

    z = rosenbrock_function([current_pos[0], current_pos[1]])

    print(
        f"Current position: {current_pos[0]:.6f}, {current_pos[1]:.6f}, {z:.6f}")

    plt.contour(X, Y, Z, levels=np.logspace(-2, 5, 15), cmap='plasma')
    plt.scatter(current_pos[0], current_pos[1], color='magenta', zorder=5)
    plt.grid(True)
    plt.pause(0.1)
    plt.clf()

plt.show()
