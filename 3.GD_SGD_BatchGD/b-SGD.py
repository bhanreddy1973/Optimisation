
# Topic: Stochastic Gradient Descent (SGD) for Linear Regression

# Description:
# This script implements linear regression using SGD for iterative weight optimization. 
# It takes data matrices X and Y, augments X with a bias term, and updates weights 
# by minimizing the mean squared error. After each iteration, it calculates and stores 
# the total loss for monitoring and plots the loss reduction and weight optimization over time.

import matplotlib.pyplot as plt
import numpy as np

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



# Parameters for SGD
alpha = 0.01  # Learning rate
max_it = 100  # Maximum number of iterations
N, d = X.shape
X_aug = np.hstack((X, np.ones((N, 1))))  # Augmenting X with a bias term
W_sgd = np.zeros((d + 1, 1))  # Initial weights including bias term

# Lists to store loss and weights history for plotting
loss_history_sgd = []
W_history_sgd = []

# SGD Implementation
for k in range(max_it):
    for i in range(N):  # Loop through each data point
        xi = X_aug[i:i+1]  # Select one sample (1x(d+1) matrix)
        yi = Y[i:i+1]      # Corresponding target value (scalar)
        
        # Step 1: Prediction and Error for the single data point
        Y_pred_i = xi @ W_sgd
        E_i = Y_pred_i - yi
        
        # Step 2: Gradient and Update
        gradient_i = xi.T @ E_i  # Gradient based on one sample
        W_sgd = W_sgd - alpha * gradient_i  # Weight update
        
    # Calculate total loss for monitoring (not for update)
    loss_sgd = (1 / (2 * N)) * np.sum((X_aug @ W_sgd - Y) ** 2)
    loss_history_sgd.append(loss_sgd)
    W_history_sgd.append(W_sgd.copy())

# Convert W_history to an array for plotting
W_history_sgd = np.array(W_history_sgd).squeeze()

# Plotting Loss Reduction and Weight Optimization for SGD
iterations = list(range(max_it))
plt.figure(figsize=(10, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(iterations, loss_history_sgd, 'b-', label='SGD Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('SGD Loss Reduction over Iterations')
plt.legend()

# Weight Optimization Plot for each weight parameter
plt.subplot(1, 2, 2)
for i in range(W_history_sgd.shape[1]):
    plt.plot(iterations, W_history_sgd[:, i], label=f'SGD Weight {i}')
plt.xlabel('Iteration')
plt.ylabel('Weight Value')
plt.title('SGD Weight Optimization over Iterations')
plt.legend()

plt.tight_layout()
plt.show()
