# optimization algorithm to minimize the Rosenbrock function using a combination of gradient descent and the Golden Section Search method for determining the optimal learning rate

# Rosenbrock Function
# The Rosenbrock function is a well-known test function for optimization algorithms, defined as:
# f(x,y)=(1−x)^2+100(y−x^2)^2

#-------------------------------------------------------------------------------------------------------
# It has a global minimum at the point (1, 1), where the function value is zero. The function is characterized by a narrow, curved valley, making it challenging for optimization algorithms to converge efficiently.
# Key Components of the Code
# Function and Gradient Definition:
# rosenbrock_function(point): Computes the value of the Rosenbrock function at a given point.
# rosenbrock_gradient(point): Computes the gradient (partial derivatives) of the Rosenbrock function, which indicates the direction of steepest ascent.
# Learning Rate Function:
# learning_rate_function(learning_rate, point): Evaluates the Rosenbrock function at a new point obtained by moving in the direction of the gradient scaled by a proposed learning rate. This function helps in finding an appropriate step size for gradient descent.
# Golden Section Search:
# golden_section_search(f, a, b, tol, point): Implements the Golden Section Search algorithm to find an optimal learning rate within a specified interval [a, b]. This method iteratively narrows down the interval based on function evaluations at two points determined by the golden ratio.

#-------------------------------------------------------------------------------------------------------
# Optimization Loop:
# The loop runs for a fixed number of iterations (200 in this case). In each iteration:
# The gradient at the current position is computed.
# The optimal learning rate is determined using the Golden Section Search.
# The current position is updated by moving against the gradient scaled by the learning rate.
# The new position is stored for later visualization.

#-------------------------------------------------------------------------------------------------------
# Visualization:
# A contour plot of the Rosenbrock function is created using matplotlib. The optimization path taken by the algorithm is highlighted on this plot, showing how it approaches the minimum.
# The contour levels represent different values of the Rosenbrock function, and points along the optimization path are marked in magenta with a cyan line connecting them.

import numpy as np
import matplotlib.pyplot as plt
import math

def rosenbrock_function(point):
    x, y = point
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(point):
    x, y = point
    df_dx = -2 * (1 - x) - 400 * x * (y - x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

def learning_rate_function(learning_rate, point):
    return rosenbrock_function(point - learning_rate * rosenbrock_gradient(point))

def golden_section_search(f, a, b, tol=1e-5, point=np.array([0, 0])):
    golden_ratio = (math.sqrt(5) - 1) / 2

    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    while abs(b - a) > tol:
        if f(c, point) < f(d, point):
            b = d
        else:
            a = c

        c = b - golden_ratio * (b - a)
        d = a + golden_ratio * (b - a)

    return (b + a) / 2

# Set up grid for contour plot
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock_function([X, Y])

current_pos = np.array([0.0, 0.0])
positions = [current_pos.copy()]  # Store initial position

# Optimization loop
for _ in range(200):
    gradient = rosenbrock_gradient(current_pos)

    learning_rate = golden_section_search(
        learning_rate_function, 0, 1, tol=1e-5, point=current_pos)

    current_pos = current_pos - learning_rate * gradient
    positions.append(current_pos.copy())  # Store new position

# Convert positions to a numpy array for easy indexing
positions = np.array(positions)

# Create contour plot and highlight points
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 30), cmap='plasma')
plt.scatter(positions[:, 0], positions[:, 1], color='magenta', zorder=5)
plt.plot(positions[:, 0], positions[:, 1], color='cyan', linewidth=2)  # Path taken
plt.title('Rosenbrock Function Contour with Optimization Path')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.colorbar(label='Function Value')
plt.show()