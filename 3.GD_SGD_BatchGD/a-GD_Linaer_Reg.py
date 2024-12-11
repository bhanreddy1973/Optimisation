# Topic: Linear Regression using Gradient Descent

# Description:
# This code implements a linear regression model with gradient descent optimization.
# Given input data (X, Y), it uses an augmented matrix for the input (to include the bias term) 
# and iteratively updates the weights to minimize the mean squared error. 
# It also plots the loss reduction and the optimization of weights over iterations.

import numpy as np
import matplotlib.pyplot as plt

# Input Data
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

# Augment X to include a bias term (intercept)
X_aug = np.hstack((np.ones((X.shape[0], 1)), X))  # Adding a column of ones for the bias term

# Initialize parameters
d = X_aug.shape[1]  # Number of features (including bias term)
W = np.zeros((d, 1))  # Initialize weights (with bias term)
alpha = 0.01  # Learning rate
max_it = 100  # Maximum number of iterations

# For storing history of weights and loss
W_history = [W.copy()]
loss_history = []

# Gradient Descent Algorithm
for k in range(max_it):
    # Step 1: Prediction
    Y_pred = X_aug @ W  # Hypothesis (Y_pred) = X_aug * W
    
    # Step 2: Error calculation
    E = Y_pred - Y  # Error (E) = Y_pred - Y
    
    # Step 3: Loss calculation (for monitoring)
    loss = (1 / (2 * len(Y))) * np.sum(E**2)  # Least square loss function
    loss_history.append(loss)
    
    # Step 4: Gradient computation
    gradient = (1 / len(Y)) * X_aug.T @ E  # Gradient of the loss w.r.t W
    
    # Step 5: Weight update
    W = W - alpha * gradient  # Update weights using learning rate
    
    # Store weights after each iteration
    W_history.append(W.copy())

# Visualization of Loss Reduction and Weight Optimization
iterations = list(range(max_it + 1))

# Plot Loss vs Iterations
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(iterations[:-1], loss_history, 'r-', label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Reduction over Iterations')
plt.legend()

# Weight Plot for each weight parameter
W_history = np.array(W_history).squeeze()  # Convert to a proper numpy array for indexing
for i in range(W_history.shape[1]):
    plt.subplot(1, 2, 2)
    plt.plot(iterations, W_history[:, i], label=f'Weight {i}')
plt.xlabel('Iteration')
plt.ylabel('Weight Value')
plt.title('Optimization of Weights over Iterations')
plt.legend()

plt.tight_layout()
plt.show()
