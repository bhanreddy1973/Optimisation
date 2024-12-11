# Title: Optimization on the Himmelblau Function using Momentum-based Gradient Descent
# This code performs optimization on the Himmelblau function using a momentum-based gradient descent method.
# The optimization uses a momentum term to help smooth the descent path and achieve faster convergence.


#-------------------------------------------------------------------------------------------------------
# Key Components:
# - Himmelblau Function:
#   Defines the Himmelblau function, a well-known benchmark function used for testing optimization algorithms.
#   The function takes two variables, x and y, and is commonly used to test gradient-based optimization techniques.

#-------------------------------------------------------------------------------------------------------
# - Gradient Calculation:
#   The gradient of the Himmelblau function is computed for both x and y, guiding the descent steps during optimization.

#-------------------------------------------------------------------------------------------------------
# - 3D Plot Setup:
#   Uses numpy's linspace and meshgrid to generate a 2D grid of points representing the domain of the Himmelblau function.
#   The surface plot visually displays the function's 3D landscape.

#-------------------------------------------------------------------------------------------------------
# - Momentum-based Gradient Descent:
#   * The optimization starts at the initial position `[0.0, 0.0]` with velocity initialized to zero.
#   * At each iteration, the gradient of the Himmelblau function at the current position is calculated.
#   * The momentum term is updated using the equation `velocity = gamma * velocity + learning_rate * gradient`, 
#     where `gamma` is the momentum coefficient and `learning_rate` controls the step size.
#   * The current position is then updated using `current_pos = current_pos - velocity`, and the new position is printed.
#   * A 3D surface plot (using `matplotlib`) is updated at each step, visualizing the optimization process.

#-------------------------------------------------------------------------------------------------------
# - Visualization:
#   * The surface plot is updated in real-time to show the optimization path on the Himmelblau function's surface.
#   * The current position is marked in magenta on the plot to highlight the optimization process.
#   * The plot is animated using `plt.pause(0.001)` to visualize the movement of the optimization process.

#-------------------------------------------------------------------------------------------------------
# Final Output:
# - A real-time 3D surface plot showing the optimization path of the momentum-based gradient descent method.
# - The plot visually demonstrates how the optimizer converges towards the minimum of the Himmelblau function.


# -----------------------------------------------------------------------------------------------
#  1st Code begins below:


# import numpy as np
# import matplotlib.pyplot as plt


# def Himmelblau_Function(point):
#     x, y = point
#     return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


# def Himmelblau_Function_gradient(point):
#     x, y = point
#     df_dx = 2 * (x**2 + y - 11) * 2 * x + 2 * (x + y**2 - 7)
#     df_dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * 2 * y
#     return np.array([df_dx, df_dy])


# x = np.linspace(-5, 5, 500)
# y = np.linspace(-5, 5, 500)
# X, Y = np.meshgrid(x, y)

# Z = Himmelblau_Function([X, Y])

# current_pos = np.array([0.0, 0.0])

# ax = plt.subplot(projection="3d", computed_zorder=False)

# for _ in range(200):
#     learning_rate = 0.01
#     gamma = 0.75
#     velocity = np.array([0.0, 0.0])

#     gradient = Himmelblau_Function_gradient(current_pos)

#     velocity = gamma * velocity + learning_rate * gradient

#     current_pos = current_pos - velocity

#     z = Himmelblau_Function(current_pos)
#     print(
#         f"Current position: {current_pos[0]:.6f}, {current_pos[1]:.6f}, {z:.6f}")

#     ax.clear()
#     ax.plot_surface(X, Y, Z, cmap="viridis")
#     ax.scatter(current_pos[0], current_pos[1], z,
#                color='magenta', s=50, zorder=5)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Himmelblau Function Optimization Using Momentum')
#     plt.pause(0.001)

# plt.show()

#-------------------------------------------------------------------------------------------------------
# 2nd  Code  for  Contour Maps

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


x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)

Z = Himmelblau_Function([X, Y])

current_pos = np.array([0.0, 0.0])

for _ in range(100):
    learning_rate = 0.01
    gamma = 0.75
    velocity = np.array([0.0, 0.0])

    gradient = Himmelblau_Function_gradient(current_pos)

    velocity = gamma * velocity + learning_rate * gradient

    current_pos = current_pos - velocity

    z = Himmelblau_Function(current_pos)
    print(
        f"Current position: {current_pos[0]:.6f}, {current_pos[1]:.6f}, {z:.6f}")
    
    plt.clf()
    plt.contour(X, Y, Z, levels = np.logspace(-1, 3, 40),cmap="viridis")
    plt.scatter(current_pos[0], current_pos[1],color='magenta',s=50, zorder=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Himmelblau Function Optimization')
    plt.pause(0.001)

plt.show()
