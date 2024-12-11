# Title: Optimization of the Rosenbrock Function with Adagrad Algorithm

# This code demonstrates the use of the Adagrad optimization algorithm to minimize the Rosenbrock function.
# The Rosenbrock function, a common test in optimization, is known for its narrow, curved valley, making it challenging for gradient-based optimizers.
# Adagrad is used here to adaptively scale the learning rate for each parameter, improving convergence in such complex landscapes.

#------------------------------------------------------------------------------------------------------- 

# Key Components:
# - Imports:
#   Imports numpy for numerical operations and matplotlib for visualization.

#------------------------------------------------------------------------------------------------------- 

# - Rosenbrock Function:
#   Defines the Rosenbrock function as (1 - x)^2 + 100 * (y - x^2)^2, which has a global minimum at (x, y) = (1, 1).
# - Gradient Computation:
#   Calculates the gradient of the Rosenbrock function to guide the optimizer in each step.

#------------------------------------------------------------------------------------------------------- 

# - Contour Plot Setup:
#   Uses meshgrid to plot contour lines for visualizing the function's shape in the range of x = [-2, 2] and y = [-1, 3].
# - Initialization:
#   Sets the initial position to [0.0, 0.0], initializes the gradient accumulation vector (for Adagrad), and defines learning rate and epsilon for stability.

#------------------------------------------------------------------------------------------------------- 

# - Optimization Loop:
#   Runs 500 iterations, updating the position with Adagrad's learning rate adjustment:
#   * Computes the gradient at the current position.
#   * Accumulates squared gradients to adjust the learning rate.
#   * Updates the position based on the adjusted gradient step.
#   * Calculates the Rosenbrock function value at each new position and logs it.

#------------------------------------------------------------------------------------------------------- 
# - Visualization:
#   In each iteration, displays a contour plot and marks the current position, updating in real-time to illustrate convergence.

# Final Output:
# - The algorithm shows the optimization path of the Adagrad algorithm as it converges toward the minimum of the Rosenbrock function.

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

for _ in range(500):

    gradient = rosenbrock_gradient(current_pos)

    gradient_vector += gradient ** 2

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
