# combined   plots   GD, SGD, Batch GD 

import numpy as np
import matplotlib.pyplot as plt

# Data from the provided matrices
X = np.array([
    [1.2589, -1.3695],
    [1.6232, 1.8824],
    [-1.4921, 1.8287],
    [1.6535, -0.0585],
    [0.5294, 1.2011],
    [-1.6098, -1.4325],
    [-0.8860, -0.3130],
    [0.1875, 1.6629],
    [1.8300, 1.1688],
    [1.8596, 1.8380]
])

Y = np.array([
    [0.7783],
    [1.1335],
    [-0.9117],
    [1.0837],
    [0.3916],
    [-1.1060],
    [-0.5909],
    [0.1827],
    [1.2432],
    [1.2851]
])

# Parameters
alpha = 0.01      # Learning rate
max_it = 100      # Maximum number of iterations
batch_size = 5    # Batch size for Batch GD
N, d = X.shape    # Number of samples and features

# Augment X with a bias term
X_aug = np.hstack((X, np.ones((N, 1))))  # Shape: (N, d+1)

# Initialize weights
W_gd = np.zeros((d + 1, 1))              # Vanilla GD weights
W_sgd = np.zeros((d + 1, 1))             # SGD weights
W_batchgd = np.zeros((d + 1, 1))         # Batch GD weights

# Initialize history lists for tracking loss and weights
loss_history_gd, W_history_gd = [], []
loss_history_sgd, W_history_sgd = [], []
loss_history_batchgd, W_history_batchgd = [], []

# Vanilla Gradient Descent (GD)
for k in range(max_it):
    Y_pred = X_aug @ W_gd
    E = Y_pred - Y
    gradient = (1 / N) * X_aug.T @ E
    W_gd = W_gd - alpha * gradient
    loss = (1 / (2 * N)) * np.sum((X_aug @ W_gd - Y) ** 2)
    loss_history_gd.append(loss)
    W_history_gd.append(W_gd.copy())

# Stochastic Gradient Descent (SGD)
for k in range(max_it):
    i = np.random.randint(0, N)  # Random sample index
    X_i = X_aug[i, :].reshape(1, -1)
    Y_i = Y[i, :].reshape(1, -1)
    Y_pred_i = X_i @ W_sgd
    E_i = Y_pred_i - Y_i
    gradient_i = X_i.T @ E_i
    W_sgd = W_sgd - alpha * gradient_i
    loss = (1 / (2 * N)) * np.sum((X_aug @ W_sgd - Y) ** 2)
    loss_history_sgd.append(loss)
    W_history_sgd.append(W_sgd.copy())

# Batch Gradient Descent (Batch GD)
for k in range(max_it):
    indices = np.random.choice(N, batch_size, replace=False)
    X_batch = X_aug[indices]
    Y_batch = Y[indices]
    Y_pred_batch = X_batch @ W_batchgd
    E_batch = Y_pred_batch - Y_batch
    gradient_batch = (1 / batch_size) * X_batch.T @ E_batch
    W_batchgd = W_batchgd - alpha * gradient_batch
    loss = (1 / (2 * N)) * np.sum((X_aug @ W_batchgd - Y) ** 2)
    loss_history_batchgd.append(loss)
    W_history_batchgd.append(W_batchgd.copy())

# Convert histories to arrays for easy plotting
W_history_gd = np.array(W_history_gd).squeeze()
W_history_sgd = np.array(W_history_sgd).squeeze()
W_history_batchgd = np.array(W_history_batchgd).squeeze()

# Plotting Loss Reduction Comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(max_it), loss_history_gd, 'b-', label='GD Loss')
plt.plot(range(max_it), loss_history_sgd, 'r-', label='SGD Loss')
plt.plot(range(max_it), loss_history_batchgd, 'g-', label='Batch GD Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Reduction Comparison')
plt.legend()

# Plotting Weight Optimization Comparison for each weight parameter
plt.subplot(1, 2, 2)
for i in range(W_history_gd.shape[1]):
    plt.plot(range(max_it), W_history_gd[:, i], 'b--',
             label=f'GD Weight {i}' if i == 0 else "")
    plt.plot(range(max_it), W_history_sgd[:, i], 'r--',
             label=f'SGD Weight {i}' if i == 0 else "")
    plt.plot(range(max_it), W_history_batchgd[:, i], 'g--',
             label=f'Batch GD Weight {i}' if i == 0 else "")
plt.xlabel('Iteration')
plt.ylabel('Weight Value')
plt.title('Weight Optimization Comparison')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
