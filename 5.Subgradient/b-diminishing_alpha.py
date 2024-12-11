# Title: Subgradient Descent for Optimization of Different Functions
#
# This code applies subgradient descent optimization to minimize different types of functions:
# - Quadratic function: f(x1, x2) = x1^2 + x2^2
# - Absolute value function: f(x1, x2) = |x1| + |x2|
# - Modified absolute value function: f(x1, x2) = |x1| + 2|x2|
#------------------------------------------------------------------------------------------------------- 

# The optimization process involves:
# - Using subgradient methods for non-differentiable functions (e.g., absolute value functions).
# - For the quadratic function, the gradient descent is directly used as the function is differentiable.
# - The subgradient descent method is applied to both the absolute value and modified absolute value functions.
# - The learning rate decreases gradually, controlled by the factor `learning_rate/(1.5**0.2)` in each iteration.
#------------------------------------------------------------------------------------------------------- 

# Key Components:
# - Function Selection:
#   The function type can be selected between 'quadratic', 'absolute', and 'absolute2'.
#   The gradient or subgradient is computed depending on the selected function type.
# - Contour Plot:
#   A contour plot of the function is generated for visualization. The path of the optimization process is shown in magenta.
#   The optimization process iterates over a fixed number of steps (100 iterations), with the path plotted at each step.
# - Convergence:
#   The optimization stops when the function value reaches its minimum or the best value found during iterations.
# - Output:
#   The final output includes the best position and function value obtained by the optimization process.
#------------------------------------------------------------------------------------------------------- 

# Final Output:
# - The plot shows the optimization path over the function's surface. The path is shown in magenta with markers at each step.
# - The best position and function value are printed at the end.
#------------------------------------------------------------------------------------------------------- 

# code :
import numpy as np
import matplotlib.pyplot as plt


def quadratic_function(point):
    x1, x2 = point
    return x1**2 + x2**2


def quadratic_gradient(point):
    x1, x2 = point
    return np.array([2 * x1, 2 * x2])


def abs_function(point):
    x1, x2 = point
    return abs(x1) + abs(x2)


def abs_gradient(point):
    x1, x2 = point
    if x1 > 0 and x2 > 0:
        return np.array([1, 1])
    elif x1 < 0 and x2 > 0:
        return np.array([-1, 1])
    elif x1 < 0 and x2 < 0:
        return np.array([-1, -1])
    else:
        return np.array([1, -1])


def abs_subgradient(point):
    x1, x2 = point
    grad_x1 = np.random.uniform(-1, 1) if x1 == 0 else (1 if x1 > 0 else -1)
    grad_x2 = np.random.uniform(-1, 1) if x2 == 0 else (1 if x2 > 0 else -1)
    return np.array([grad_x1, grad_x2])


def abs2_function(point):
    x1, x2 = point
    return abs(x1) + 2*abs(x2)


def abs2_gradient(point):
    x1, x2 = point
    if x1 > 0 and x2 > 0:
        return np.array([1, 2])
    elif x1 < 0 and x2 > 0:
        return np.array([-1, 2])
    elif x1 < 0 and x2 < 0:
        return np.array([-1, -2])
    else:
        return np.array([1, -2])


def abs2_subgradient(point):
    x1, x2 = point
    grad_x1 = np.random.uniform(-1, 1) if x1 == 0 else (1 if x1 > 0 else -1)
    grad_x2 = np.random.uniform(-2, 2) if x2 == 0 else (2 if x2 > 0 else -2)
    return np.array([grad_x1, grad_x2])


function_type = 'absolute'
if function_type == 'quadratic':
    function = quadratic_function
    gradient = quadratic_gradient
    def subgradient(p): return gradient(p)
elif function_type == 'absolute':
    function = abs_function
    gradient = abs_gradient
    subgradient = abs_subgradient
elif function_type == 'absolute2':
    function = abs2_function
    gradient = abs2_gradient
    subgradient = abs2_gradient
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(lambda x, y: function([x, y]))(X, Y)
current_pos = np.array([3.0, 4.0])
f_best = function(current_pos)
learning_rate = 0.8
positions = [current_pos.copy()]
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20, cmap='plasma')
plt.colorbar(label="Function Value (f)")
for _ in range(100):
    learning_rate = learning_rate/(1.5**0.2)

    if not np.allclose(current_pos, 0):
        grad = gradient(current_pos)
    else:
        grad = subgradient(current_pos)

    temp_next = current_pos - learning_rate * grad
    f_new = function(temp_next)

    if f_new < f_best:
        f_best = f_new
        current_pos = temp_next
        positions.append(current_pos.copy())
    plt.plot(*zip(*positions), marker='o',
             color='magenta', markersize=4, zorder=5)
    plt.pause(0.1)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title(
    f'Subgradient Descent Path for {function_type.capitalize()} Function')
plt.show()

print(f"Best position: {current_pos}, Best function value: {f_best}")
