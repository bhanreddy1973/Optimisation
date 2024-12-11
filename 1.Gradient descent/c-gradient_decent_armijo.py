#  findIing the minimum of the Rosenbrock function using gradient descent
# Function Definition:
# The rosenbrock_function computes the value of the Rosenbrock function for given coordinates 
# (x,y)
# The rosenbrock_gradient calculates the gradient (partial derivatives) of the function, which indicates the direction of steepest ascent.

#-------------------------------------------------------------------------------------------------------
# Learning Rate Adjustment:
# The learning_rate_function determines how much to adjust the current position based on the gradient and a specified learning rate. It uses a backtracking line search method to find an appropriate step size that satisfies the Armijo condition for sufficient decrease.

#-------------------------------------------------------------------------------------------------------
# Optimization Loop:
# The algorithm iteratively updates the current position by moving in the direction opposite to the gradient (steepest descent). The learning rate is adjusted dynamically to ensure convergence towards the minimum.
# The loop continues for a set number of iterations or until convergence criteria are met.

#-------------------------------------------------------------------------------------------------------
# Visualization:
# The optimization process is visualized using 3D plotting (or contour plots in an alternative version). The surface of the Rosenbrock function is plotted, and the current position during optimization is marked, allowing for a visual representation of how the algorithm approaches the minimum.

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

def learning_rate_function(learning_rate, point):
    return rosenbrock_function(point - learning_rate * rosenbrock_gradient(point))

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock_function([X, Y])

current_pos = np.array([0, 0])

ax = plt.subplot(projection="3d", computed_zorder=False)

for _ in range(1000):
    alpha = 1
    beta = 0.4
    c1 = 0.0001
    gradient = rosenbrock_gradient(current_pos)

    while learning_rate_function(alpha, current_pos) > \
            rosenbrock_function(current_pos) + c1 * alpha * np.dot(gradient, gradient):
        alpha = alpha * beta

    current_pos = current_pos - alpha * gradient

    z = rosenbrock_function(current_pos)
    print(
        f"Current position: {current_pos[0]:.6f}, {current_pos[1]:.6f}, {z:.6f}")

    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.scatter(current_pos[0], current_pos[1], z, color='magenta')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rosenbrock Function Optimization')
    plt.pause(0.001)

plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# def rosenbrock_function(point):
#     x, y = point
#     return (1 - x)**2 + 100 * (y - x**2)**2


# def rosenbrock_gradient(point):
#     x, y = point
#     df_dx = -2 * (1 - x) - 400 * x * (y - x**2)
#     df_dy = 200 * (y - x**2)
#     return np.array([df_dx, df_dy])


# def learning_rate_function(learning_rate, point):
#     return rosenbrock_function(point - learning_rate * rosenbrock_gradient(point))


# x = np.linspace(-2, 2, 400)
# y = np.linspace(-1, 3, 400)
# X, Y = np.meshgrid(x, y)
# Z = rosenbrock_function([X, Y])

# current_pos = np.array([0, -1])

# for _ in range(1000):
#     alpha = 1
#     beta = 0.4
#     c1 = 0.0001
#     gradient = rosenbrock_gradient(current_pos)

#     while learning_rate_function(alpha, current_pos) > \
#             rosenbrock_function(current_pos) + c1 * alpha * np.dot(gradient, gradient):
#         alpha = alpha * beta

#     current_pos = current_pos - alpha * gradient

#     z = rosenbrock_function(current_pos)
#     print(
#         f"Current position: {current_pos[0]:.6f}, {current_pos[1]:.6f}, {z:.6f}")

#     plt.clf()
#     plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 30), cmap='viridis')
#     plt.scatter(current_pos[0], current_pos[1],
#                 color='magenta', zorder=5, s=70)
#     plt.title('Rosenbrock Function Optimization')
#     plt.pause(0.001)
#     plt.clf()

# plt.show()
