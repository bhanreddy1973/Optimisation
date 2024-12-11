# Title: Optimization on the Himmelblau Function using Gradient Descent with Momentum

# This code applies gradient descent with momentum to minimize the Himmelblau function.
# The optimization method uses an updated velocity term, which is influenced by the past gradient directions.

#-------------------------------------------------------------------------------------------------------
# Key Components:
# - Himmelblau Function:
#   The Himmelblau function is a common test function for optimization algorithms, and it has multiple local minima.
#   It takes a 2D point (x, y) and calculates the function value using the formula:
#   (x^2 + y - 11)^2 + (x + y^2 - 7)^2.

#-------------------------------------------------------------------------------------------------------
# - Gradient Calculation:
#   The gradient of the Himmelblau function is calculated to determine the direction of steepest ascent/descent. 
#   The gradient consists of partial derivatives with respect to x and y.

#-------------------------------------------------------------------------------------------------------
# - Momentum-based Gradient Descent:
#   The optimization process starts at `[0, 0]` and iteratively updates the position using the gradient of the function.
#   The momentum term is updated as `velocity = velocity + learning_rate * gradient`, where the `learning_rate` 
#   controls the size of each step, and `gamma` controls the influence of the previous velocities.
#   The position is updated as `current_pos = current_pos - velocity`, and the optimization continues for 200 iterations.

#-------------------------------------------------------------------------------------------------------
# - Lookahead Gradient:
#   The gradient calculation is performed on a lookahead position, defined as `x_lookahead = current_pos - gamma * velocity`.
#   This allows the algorithm to "look ahead" in the direction of the current velocity before calculating the gradient.

#-------------------------------------------------------------------------------------------------------
# - Visualization:
#   A contour plot of the Himmelblau function is updated at each iteration, with the current position marked in magenta.
#   The contour plot helps visualize the optimization path on the function's surface.
#   `plt.pause(0.1)` allows for a brief animation pause at each iteration to show the optimization in action.
#   `plt.clf()` is used to clear the figure before plotting the next iteration, ensuring a smooth animation.

#-------------------------------------------------------------------------------------------------------
# Final Output:
# - The plot shows the trajectory of the optimization process as it converges towards one of the minima of the Himmelblau function.
# - The position of the optimization is marked by a magenta dot that moves towards the minimum.

# Code begins below:


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

current_pos = np.array([0, 0])
velocity = np.array([0, 0])

for _ in range(100):
    learning_rate = 0.01
    gamma = 0.75

    x_lookahead = current_pos - gamma * velocity

    gradient = Himmelblau_Function_gradient(x_lookahead)

    velocity = velocity + learning_rate * gradient

    current_pos = current_pos - velocity

    z = Himmelblau_Function([current_pos[0], current_pos[1]])
    print(
        f"Current position: {current_pos[0]:.6f}, {current_pos[1]:.6f}, {z:.6f}")

    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 18),
                cmap='plasma')
    plt.scatter(current_pos[0], current_pos[1], color='magenta', zorder=5)
    plt.pause(0.1)
    plt.clf() 
    plt.title('NGD')

plt.show()
