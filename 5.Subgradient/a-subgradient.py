# Title: Subgradient Descent Optimization
#------------------------------------------------------------------------------------------------------- 

# Summary:
# This script demonstrates the implementation of subgradient descent optimization for different types of functions:
# 1. Quadratic function: f(x1, x2) = x1^2 + x2^2
# 2. Absolute function: f(x1, x2) = |x1| + |x2|
# 3. Weighted Absolute function: f(x1, x2) = |x1| + 2*|x2|

#------------------------------------------------------------------------------------------------------- 

# The optimization process is visualized by plotting the descent paths on a contour map of the function.
# The descent paths are calculated using subgradient descent with a fixed learning rate.
# The script outputs the best position and function value found after 50 iterations of optimization.
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
function_type = 'quadratic'
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
for _ in range(50):
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
