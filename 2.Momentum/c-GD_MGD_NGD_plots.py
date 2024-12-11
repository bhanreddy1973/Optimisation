# Title: Optimization on the Himmelblau Function using Different Gradient Descent Variants
# This code performs optimization on the Himmelblau function using three different variants of gradient descent:
# - NGD (Nesterov's Accelerated Gradient Descent)
# - MGD (Momentum Gradient Descent)
# - Constant Alpha Gradient Descent

#------------------------------------------------------------------------------------------------------- 
# Key Components:
# - Himmelblau Function:
#   Defines the Himmelblau function as a multi-variable function with two variables, x and y.
#   The function is commonly used for testing optimization algorithms.
# - Gradient Calculation:
#   Implements the gradient of the Himmelblau function which is used to guide each optimization step.

#------------------------------------------------------------------------------------------------------- 
# - Contour Plot Setup:
#   Uses numpy's linspace and meshgrid to create a grid of points representing the 2D domain of the Himmelblau function.
#   The contour plot will visually display the optimization landscape.

#------------------------------------------------------------------------------------------------------- 
# - Optimization Methods:
#   * **NGD (Nesterov's Accelerated Gradient Descent)**:
#     - Combines momentum and a lookahead step to improve convergence speed and stability.
#     - The position is updated using the momentum term which is updated based on the "lookahead" point.
#   * **MGD (Momentum Gradient Descent)**:
#     - Uses a moving average of past gradients to update the current position.
#     - The momentum term helps smooth the optimization trajectory, improving convergence.
#   * **Constant Alpha Gradient Descent**:
#     - A standard gradient descent method with a fixed learning rate (alpha).
#     - Updates the position directly based on the gradient without momentum or lookahead.

#------------------------------------------------------------------------------------------------------- 
# - Initialization:
#   * `current_pos` initializes the starting position of the optimization process.
#   * The momentum variables (`velocity_ngd`, `velocity_mgd`) are initialized to zero.

# - Optimization Loop:
#   * For each variant, the optimization runs for 20 iterations, updating the position and calculating the new gradient.
#   * For NGD, the momentum is updated based on a lookahead position.
#   * For MGD, the momentum is updated based on the previous gradient.
#   * For Constant Alpha, the position is updated directly using the gradient and a fixed learning rate.

#------------------------------------------------------------------------------------------------------- 

# - Visualization:
#   * The code generates a contour plot showing the function's landscape.
#   * It then plots the optimization paths for all three methods (NGD, MGD, and Constant Alpha) to visualize how each method converges to the minimum.
#   * The paths of each method are marked with different colors and symbols to distinguish between them.
#   * The starting point (0,0) is marked in magenta for clarity.

# Final Output:
# - A plot showing the contour of the Himmelblau function and the optimization paths taken by each method.
# - The plot visually demonstrates the convergence behavior of Nesterov's Accelerated Gradient Descent, Momentum Gradient Descent, and Constant Alpha Gradient Descent.




#------------------------------------------------------------------------------------------------------- 
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
NGD_line = [current_pos.copy()]
velocity_ngd = np.array([0, 0])

for _ in range(20):
    learning_rate = 0.01
    gamma = 0.75

    x_lookahead = current_pos - gamma * velocity_ngd

    gradient = Himmelblau_Function_gradient(x_lookahead)

    velocity_ngd = velocity_ngd + learning_rate * gradient

    current_pos = current_pos - velocity_ngd

    NGD_line.append(current_pos.copy())

current_pos2 = np.array([0, 0])
MGD_line = [current_pos2.copy()]
velocity_mgd = np.array([0.0, 0.0])

for _ in range(20):
    learning_rate = 0.01
    gamma = 0.75

    gradient = Himmelblau_Function_gradient(current_pos2)

    velocity_mgd = gamma * velocity_mgd + learning_rate * gradient

    current_pos2 = current_pos2 - velocity_mgd

    MGD_line.append(current_pos2.copy())

current_pos1 = np.array([0, 0])
constant_alpha_line = [current_pos1.copy()]

for _ in range(20):
    gradient = Himmelblau_Function_gradient(current_pos1)

    learning_rate = 0.01

    current_pos1 = current_pos1 - learning_rate * gradient
    constant_alpha_line.append(current_pos1.copy())

plt.figure(figsize=(10, 8))

plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')

NGD_line = np.array(NGD_line)
MGD_line = np.array(MGD_line)
constant_alpha_line = np.array(constant_alpha_line)

plt.plot(NGD_line[:, 0], NGD_line[:, 1], 'r-o',
         label='NGD Line', markersize=4)

plt.plot(MGD_line[:, 0], MGD_line[:, 1],
         'b-s', label='MGD Line', markersize=4)

plt.plot(constant_alpha_line[:, 0], constant_alpha_line[:,
         1], 'k-s', label='Constant Alpha', markersize=4)

plt.scatter(0, 0, color='magenta', s=50, label='Starting Point')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rosenbrock Function Optimization Paths')
plt.legend()
plt.show()
