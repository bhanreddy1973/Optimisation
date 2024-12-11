#optimization of the Rosenbrock function using gradient descent in a three-dimensional space
# This code has done in two approaches Simple Quadratic function and rosenbroak

#-------------------------------------------------------------------------------------------------------
# Rosenbrock Function

# The Rosenbrock function is a classic test problem for optimization algorithms, defined as:
# f(x,y)=(1−x)^2+100(y−x^2)^2
# This function has a global minimum at the point (1, 1), where the function value is zero. It is known for its narrow, curved valley that makes it challenging for optimization algorithms to converge efficiently.
# Key Components of the Code
#-------------------------------------------------------------------------------------------------------
# Function Definition:
# rosenbrock(x, y): Computes the value of the Rosenbrock function at given coordinates 
# (x,y).
# f_derivative(x, y): Computes the partial derivatives (gradient) of the Rosenbrock function with respect to 
# x and y.
# y. This gradient indicates the direction of steepest ascent.
# Grid Setup for Visualization:
# The code creates a grid using numpy's meshgrid function to evaluate the Rosenbrock function over a specified range. This grid is used for plotting the surface of the function.
# Initialization:
# The starting position currentpos is initialized at (-1, 1), which is close to the global minimum. The initial function value at this position is also calculated.

#-------------------------------------------------------------------------------------------------------
# Gradient Descent Loop:
# The loop runs for a specified number of iterations (3500 in this case). In each iteration:
# The gradient at the current position is calculated.
# The new position is updated by moving against the gradient scaled by a fixed learning rate (0.002).
# The updated position and its corresponding function value are printed for monitoring convergence.
# The 3D surface plot of the Rosenbrock function is updated to visualize the optimization process.
#-------------------------------------------------------------------------------------------------------
# Visualization:
# A 3D plot is created using matplotlib, where:
# The surface of the Rosenbrock function is displayed.
# The current position in the optimization process is marked with a magenta dot.
# The plot updates dynamically to show how the algorithm approaches the minimum.

###-------------------------------------------------------------------------------------------------------------------------------
### 1st Approach
# # # Simple Quadratic Function(x^2)

import numpy as np
import matplotlib.pyplot as plt


def quadratic_function(x):
    return x**2


def quadratic_derivative(x):
    return 2 * x


x = np.arange(-100, 100, 0.1)

y = quadratic_function(x)

currentpos = (80, quadratic_function(80))

learning_rate = 0.01

for _ in range(200):

    x_derivative = quadratic_derivative(currentpos[0])
    x_new = currentpos[0] - learning_rate * x_derivative
    currentpos = (x_new, quadratic_function(x_new))

    plt.plot(x, y)
    plt.scatter(currentpos[0], currentpos[1], color='red')
    plt.pause(0.01)
    plt.clf()

# Multivariable function
import numpy as np
import matplotlib.pyplot as plt


def multivariable_quadratic(x, y):
    return x**2 + y**2

def multivariable_quadratic_derivative(x, y):
    dx = 2 * x
    dy = 2 * y
    return dx, dy

x = np.arange(-100, 100, 0.1)
y = np.arange(-100, 100, 0.1)

X, Y = np.meshgrid(x, y)

Z = multivariable_quadratic(X, Y)

currentpos = (50, 50, multivariable_quadratic(50, 50))

learning_rate = 0.01

ax = plt.subplot(projection='3d', computed_zorder=False)  # 3d plot 

for _ in range(120):
    x_derivative, y_derivative = multivariable_quadratic_derivative(
        currentpos[0], currentpos[1])
    x_new, y_new = currentpos[0] - learning_rate * \
        x_derivative, currentpos[1] - learning_rate * y_derivative
    currentpos = (x_new, y_new, multivariable_quadratic(x_new, y_new))

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.scatter(currentpos[0], currentpos[1], currentpos[2], color='magenta')
    plt.pause(0.01)
    ax.clear()


##---------------------------------------------------------------------------------------------
##2nd  Approach 
# Rosenbrocks function alpha for(-3,-4) = 0.0003 , alpha for(-1,1) = 0.002 and try other data points


# import numpy as np
# import matplotlib.pyplot as plt


# def rosenbrock(x, y):
#     return (1 - x)**2 + 100 * (y - x**2)**2


# def f_derivative(x, y):
#     x_derivative = -2 * (1 - x) - 400 * x * (y - x**2)
#     y_derivative = 200 * (y - x**2)
#     return x_derivative, y_derivative


# x = np.arange(-2, 2, 0.05)
# y = np.arange(-2, 2, 0.05)

# X, Y = np.meshgrid(x, y)

# Z = rosenbrock(X, Y)

# currentpos = (-1,1, rosenbrock(-1,1))

# learning_rate = 0.002

# ax = plt.subplot(projection='3d', computed_zorder=False)

# for _ in range(3500):
#     x_derivative, y_derivative = f_derivative(currentpos[0], currentpos[1])
#     x_new, y_new = currentpos[0] - learning_rate * \
#         x_derivative, currentpos[1] - learning_rate * y_derivative
#     currentpos = (x_new, y_new, rosenbrock(x_new, y_new))

#     print(currentpos)
#     ax.plot_surface(X, Y, Z, cmap='viridis')
#     ax.scatter(currentpos[0], currentpos[1], currentpos[2], color='magenta')
#     plt.pause(0.0001)
#     ax.clear()

