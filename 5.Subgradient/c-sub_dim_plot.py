# Title: Subgradient Descent Optimization with Constant vs. Diminishing Step Size

# This code implements subgradient descent optimization on the absolute value function:
# f(x1, x2) = |x1| + 2|x2|, using two different step size strategies: 
# constant step size and diminishing step size.

#------------------------------------------------------------------------------------------------------- 

# Key Components:
# - Objective Function:
#   The function being optimized is f(x1, x2) = |x1| + 2|x2|, which is non-differentiable at x1 = 0 and x2 = 0.
#   The function uses absolute values, making it suitable for testing subgradient methods.

#------------------------------------------------------------------------------------------------------- 
# - Subgradient Computation:
#   The subgradient for each variable is computed based on the sign of x1 and x2. If either variable is zero,
#   a random value within the appropriate range is chosen to handle the non-differentiable points.

# - Step Size Strategies:
#   - Constant Step Size: The step size is fixed at 0.8 throughout the optimization process.
#   - Diminishing Step Size: The step size decreases with each iteration, calculated by multiplying the previous
#     step size by (1.5^-0.7).

#------------------------------------------------------------------------------------------------------- 
# - Optimization Process:
#   Both optimization strategies run for 50 iterations, with positions updated according to the subgradient and
#   the respective step sizes.

#------------------------------------------------------------------------------------------------------- 

# - Visualization:
#   The optimization paths for both strategies are plotted on a contour plot of the objective function,
#   with blue markers for constant step size and green for diminishing step size. The contour plot helps to
#   visualize the function's surface and how each strategy converges to the minimum.

#------------------------------------------------------------------------------------------------------- 
# Final Output:
# - A plot is generated showing the contour lines of the objective function and the optimization paths for both
#   constant and diminishing step size strategies. The optimization paths demonstrate the behavior of each method
#   in approaching the minimum of the function.
#------------------------------------------------------------------------------------------------------- 

# code :

import numpy as np
import matplotlib.pyplot as plt


def abs_function(point):
    x1, x2 = point
    return abs(x1) + 2*abs(x2)


def abs_subgradient(point):
    x1, x2 = point
    grad_x1 = np.random.uniform(-1, 1) if x1 == 0 else (1 if x1 > 0 else -1)
    grad_x2 = np.random.uniform(-2, 2) if x2 == 0 else (2 if x2 > 0 else -2)
    return np.array([grad_x1, grad_x2])


x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(lambda x, y: abs_function([x, y]))(X, Y)

initial_position = np.array([3.0, 4.0])
constant_step_size = 0.8
diminishing_step_size = 0.8
positions_const = [initial_position.copy()]
positions_dim = [initial_position.copy()]

current_pos_const = initial_position.copy()
for _ in range(50):
    grad_const = abs_subgradient(current_pos_const)
    current_pos_const = current_pos_const - constant_step_size * grad_const
    positions_const.append(current_pos_const.copy())

current_pos_dim = initial_position.copy()
for i in range(50):
    grad_dim = abs_subgradient(current_pos_dim)
    current_pos_dim = current_pos_dim - diminishing_step_size * grad_dim
    diminishing_step_size /= (1.5**0.7)
    positions_dim.append(current_pos_dim.copy())

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20, cmap='plasma')
plt.colorbar(label="Function Value (f)")
plt.plot(*zip(*positions_const), marker='o', color='blue',
         markersize=4, label='Constant Step Size')
plt.plot(*zip(*positions_dim), marker='x', color='green',
         markersize=4, label='Diminishing Step Size')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Subgradient Descent Paths: Constant vs. Diminishing Step Size')
plt.show()
