# combined result of  armijo line search, golden section , constant alpha methods 

import numpy as np
import matplotlib.pyplot as plt
import math


def rosenbrock_function(point):
    x, y = point
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(point):
    x, y = point
    df_dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    df_dy = 200 * (y - x ** 2)
    return np.array([df_dx, df_dy])


def learning_rate_function(learning_rate, point):
    return rosenbrock_function(point - learning_rate * rosenbrock_gradient(point))


x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock_function([X, Y])

current_pos = np.array([0.0, 0.0])
armijo_line = [current_pos.copy()]

for _ in range(200):
    alpha = 1
    beta = 0.4
    c1 = 0.0001
    gradient = rosenbrock_gradient(current_pos)

    while learning_rate_function(alpha, current_pos) > \
            rosenbrock_function(current_pos) + c1 * alpha * np.dot(gradient, gradient):
        alpha = alpha * beta

    current_pos = current_pos - alpha * gradient
    armijo_line.append(current_pos.copy())


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


current_pos2 = np.array([0.0, 0.0])
golden_search_line = [current_pos2.copy()]

for _ in range(200):
    gradient = rosenbrock_gradient(current_pos2)

    learning_rate = golden_section_search(
        learning_rate_function, 0, 1, tol=1e-5, point=current_pos2)

    current_pos2 = current_pos2 - learning_rate * gradient
    golden_search_line.append(current_pos2.copy())

current_pos1 = np.array([0.0, 0.0])
constant_alpha_line = [current_pos1.copy()]

for _ in range(200):
    gradient = rosenbrock_gradient(current_pos1)

    learning_rate = 0.003

    current_pos1 = current_pos1 - learning_rate * gradient
    constant_alpha_line.append(current_pos1.copy())

plt.figure(figsize=(10, 8))

plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')

armijo_line = np.array(armijo_line)
golden_search_line = np.array(golden_search_line)
constant_alpha_line = np.array(constant_alpha_line)

plt.plot(armijo_line[:, 0], armijo_line[:, 1], 'r-o',
         label='Armijo Line Search', markersize=4)
plt.plot(golden_search_line[:, 0], golden_search_line[:, 1],
         'b-s', label='Golden Section Search', markersize=4)

plt.plot(constant_alpha_line[:, 0], constant_alpha_line[:,
         1], 'ks', label='Constant Alpha', markersize=4)

plt.scatter(0, 0, color='magenta', s=50, label='Starting Point')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rosenbrock Function Optimization Paths')
plt.legend()
plt.show()
