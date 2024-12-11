# Batch Gradient Descent Implementation in Python

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
 
# Description: Define input data matrices for feature set `X` and target values `Y`
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

# Topic: Parameters Initialization
# Description: Initialize parameters for Batch Gradient Descent
alpha = 0.01       # Learning rate
max_it = 100       # Maximum number of iterations
batch_size = 5     # Size of each batch
N, d = X.shape     # Number of samples and features

# Topic: Augmenting Data with Bias Term
# Description: Add a column of 1s to `X` for intercept term and initialize weights
X_aug = np.hstack((X, np.ones((N, 1))))  # Shape: (N, d+1)
W_batchgd = np.zeros((d + 1, 1))         # Initial weights including bias term

# Topic: Lists for Tracking Loss and Weight Updates
# Description: Track the loss and weight values for each iteration
loss_history_batchgd = []
W_history_batchgd = []

# Topic: Batch Gradient Descent Implementation
# Description: Run Batch Gradient Descent to optimize weights
for k in range(max_it):
    # Step 1: Select a random batch of data
    indices = np.random.choice(N, batch_size, replace=False)
    X_batch = X_aug[indices]      # Random batch for current iteration
    Y_batch = Y[indices]          # Corresponding Y values for the batch

    # Step 2: Prediction and Error Calculation
    Y_pred_batch = X_batch @ W_batchgd   # Predict Y values for the batch
    E_batch = Y_pred_batch - Y_batch     # Calculate error for the batch

    # Step 3: Compute Gradient and Update Weights
    gradient_batch = (1 / batch_size) * X_batch.T @ E_batch   # Calculate gradient
    W_batchgd = W_batchgd - alpha * gradient_batch            # Update weights

    # Step 4: Calculate Total Loss for Monitoring
    loss_batchgd = (1 / (2 * N)) * np.sum((X_aug @ W_batchgd - Y) ** 2)
    loss_history_batchgd.append(loss_batchgd)  # Store loss value
    W_history_batchgd.append(W_batchgd.copy()) # Store updated weights

# Topic: Plotting Loss and Weight Optimization
# Description: Generate plots for loss reduction and weight optimization over iterations
W_history_batchgd = np.array(W_history_batchgd).squeeze()  # Convert to array for plotting
iterations = list(range(max_it))

# Plot 1: Loss Reduction Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(iterations, loss_history_batchgd, 'g-', label='Batch GD Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Batch GD Loss Reduction over Iterations')
plt.legend()

# Plot 2: Weight Optimization Plot
plt.subplot(1, 2, 2)
for i in range(W_history_batchgd.shape[1]):
    plt.plot(iterations, W_history_batchgd[:, i], label=f'Batch GD Weight {i}')
plt.xlabel('Iteration')
plt.ylabel('Weight Value')
plt.title('Batch GD Weight Optimization over Iterations')
plt.legend()

plt.tight_layout()
plt.show()
