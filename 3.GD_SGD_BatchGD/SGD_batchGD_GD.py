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


learning_rate = 0.003
iterations = 500

start_gd = np.array([0.0, 0.0])
start_sgd = np.array([0.0, 0.0])
start_batchgd = np.array([0.0, 0.0])

path_gd = [start_gd]
path_sgd = [start_sgd]
path_batchgd = [start_batchgd]

X_gd = start_gd.copy()
for i in range(iterations):
    grad = rosenbrock_gradient(X_gd)
    X_gd -= learning_rate * grad
    path_gd.append(X_gd.copy())

X_sgd = start_sgd.copy()
for i in range(iterations):
    grad = rosenbrock_gradient(X_sgd) + np.random.randn(2) * 0.3
    X_sgd -= learning_rate * grad
    path_sgd.append(X_sgd.copy())

X_batchgd = start_batchgd.copy()
batch_size = 14
for i in range(iterations):
    grad = rosenbrock_gradient(X_batchgd) * (batch_size / 20)
    X_batchgd -= learning_rate * grad
    path_batchgd.append(X_batchgd.copy())

path_gd = np.array(path_gd)
path_sgd = np.array(path_sgd)
path_batchgd = np.array(path_batchgd)

x1_vals = np.linspace(-3, 3, 510)
x2_vals = np.linspace(-3, 3, 510)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = rosenbrock_function([X1, X2])

plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z, levels=np.logspace(-1, 3, 10), cmap='viridis')

plt.plot(path_gd[:, 0], path_gd[:, 1], 'bs-', label='GD')
plt.plot(path_sgd[:, 0], path_sgd[:, 1], 'r*-', label='SGD')
plt.plot(path_batchgd[:, 0], path_batchgd[:, 1], 'k*-', label='BatchGD')

plt.plot(0.0, 0.0, 'go', label='Start')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title("Optimization Path: GD vs SGD vs BatchGD")
plt.legend()
plt.grid(True)
plt.show()
