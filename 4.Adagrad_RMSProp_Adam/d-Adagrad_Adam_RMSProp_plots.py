# optimization of the Rosenbrock function using three different adaptive gradient descent algorithms: 
# Adagrad, RMSProp, and Adam. Each method adjusts the learning rate based on the gradients encountered during optimization, 
# allowing for more efficient convergence towards the function's minimum.
#------------------------------------------------------------------------------------------------------- 

# The Rosenbrock function is defined as:

# \[ f(x, y) = (1 - x)^2 + 100 (y - x2)2 \]

# This function has a global minimum at (1,1), where the function value is zero. 
# It is commonly used as a performance test problem for optimization algorithms due to its challenging shape.

#------------------------------------------------------------------------------------------------------- 

# Optimization Algorithms

# Adagrad
# Adagrad adapts the learning rate based on the historical sum of squared gradients.
# This means that parameters with larger gradients will have their learning rates reduced more significantly than those with smaller gradients.

# Formula:
# θ_t = θ_(t−1) − (η / (sqrt(G_t) + ε)) * g_t
# Where G_t is the sum of squares of past gradients, η is the learning rate, ε is a small constant for numerical stability, and g_t is the gradient at time t.
#------------------------------------------------------------------------------------------------------- 

# RMSProp
# RMSProp modifies Adagrad to include a decay factor, which prevents the learning rate from becoming too small too quickly.
# This allows for more stable convergence.

# Formula:
# v_t = β * v_(t−1) + (1−β) * g_t^2
# θ_t = θ_(t−1) − (η / (sqrt(v_t) + ε)) * g_t
# Where v_t is an exponentially decaying average of past squared gradients, β is the decay rate, η is the learning rate, ε is a small constant, and g_t is the gradient at time t.
#------------------------------------------------------------------------------------------------------- 

# Adam
# Adam combines ideas from both momentum and RMSProp. It maintains two moving averages: one for the gradients and one for the squared gradients.
# It also includes bias correction to account for initialization effects.
 
# Formula:
# m_t = β1 * m_(t−1) + (1−β1) * g_t
# v_t = β2 * v_(t−1) + (1−β2) * g_t^2
# m_hat_t = m_t / (1−β1^t)
# v_hat_t = v_t / (1−β2^t)
# θ_t = θ_(t−1) − (η / (sqrt(v_hat_t) + ε)) * m_hat_t
# Where m_t and v_t are moving averages of the gradient and its square, β1 and β2 are decay rates, η is the learning rate, and ε is a small constant.


#------------------------------------------------------------------------------------------------------- 

# Function Definitions:
# rosenbrock_function(point): Computes the value of the Rosenbrock function at a given point (x, y).
# rosenbrock_gradient(point): Computes the gradient of the Rosenbrock function.

# Grid Setup for Visualization:
# A grid is created using numpy's meshgrid to evaluate the Rosenbrock function over a specified range for contour plotting.
#------------------------------------------------------------------------------------------------------- 

# Adagrad Optimization:
# Initializes parameters and iteratively updates the current position based on the Adagrad algorithm.
# The history of squared gradients is accumulated to adjust the learning rate dynamically.
#------------------------------------------------------------------------------------------------------- 

# RMSProp Optimization:
# Similar to Adagrad but incorporates an exponential decay factor in updating the squared gradient history.
# Adam Optimization:
# Implements both momentum and RMSProp concepts, maintaining separate moving averages for gradients and squared gradients.
#------------------------------------------------------------------------------------------------------- 

# Visualization:
# A contour plot of the Rosenbrock function is created using matplotlib.
# The paths taken by each optimization algorithm are plotted in different colors, showing how they converge towards the minimum


#------------------------------------------------------------------------------------------------------- 
# code :
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


x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock_function([X, Y])

current_pos_adagrad = np.array([0.0, 0.0])
adagrad_line = [current_pos_adagrad]
gradient_vector_adagrad = np.zeros(2)
learning_rate_adagrad = 0.09
epsilon = 1e-8

for _ in range(500):
    gradient_adagrad = rosenbrock_gradient(current_pos_adagrad)
    gradient_vector_adagrad += gradient_adagrad ** 2
    current_pos_adagrad = current_pos_adagrad - \
        (learning_rate_adagrad /
         np.sqrt(gradient_vector_adagrad + epsilon)) * gradient_adagrad
    adagrad_line.append(current_pos_adagrad)

current_pos_RMSProp = np.array([0.0, 0.0])
RMSProp_line = [current_pos_RMSProp]
gradient_vector_RMSProp = np.zeros(2)
learning_rate_RMSProp = 0.007
beta_RMSProp = 0.90

for _ in range(500):
    gradient_RMSProp = rosenbrock_gradient(current_pos_RMSProp)
    gradient_vector_RMSProp = beta_RMSProp * gradient_vector_RMSProp + \
        (1 - beta_RMSProp) * gradient_RMSProp ** 2
    current_pos_RMSProp = current_pos_RMSProp - \
        (learning_rate_RMSProp /
         np.sqrt(gradient_vector_RMSProp + epsilon)) * gradient_RMSProp
    RMSProp_line.append(current_pos_RMSProp)

current_pos_Adam = np.array([0.0, 0.0])
Adam_line = [current_pos_Adam]
gradient_vector_Adam = np.zeros(2)
momentum_vector_Adam = np.zeros(2)
learning_rate_Adam = 0.01
epsilon = 1e-8
beta1 = 0.90
beta2 = 0.99

for i in range(500):
    gradient_Adam = rosenbrock_gradient(current_pos_Adam)

    momentum_vector_Adam = beta1 * \
        momentum_vector_Adam + (1 - beta1) * gradient_Adam

    gradient_vector_Adam = beta2 * gradient_vector_Adam + \
        (1 - beta2) * (gradient_Adam ** 2)

    momentum_vector_Adam_corrected = momentum_vector_Adam / \
        (1 - beta1 ** (i + 1))
    gradient_vector_Adam_corrected = gradient_vector_Adam / \
        (1 - beta2 ** (i + 1))

    current_pos_Adam = current_pos_Adam - (learning_rate_Adam / np.sqrt(
        gradient_vector_Adam_corrected + epsilon)) * momentum_vector_Adam_corrected
    Adam_line.append(current_pos_Adam)

plt.figure(figsize=(10, 8))

plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')

adagrad_line = np.array(adagrad_line)
RMSProp_line = np.array(RMSProp_line)
Adam_line = np.array(Adam_line)


plt.plot(adagrad_line[:, 0], adagrad_line[:, 1], 'r-o',
         label='Adagrad Line Search', markersize=4)
plt.plot(RMSProp_line[:, 0], RMSProp_line[:, 1], 'b-s',
         label='RMSProp Line Search', markersize=4)
plt.plot(Adam_line[:, 0], Adam_line[:, 1], 'k-s',
         label='Adam Line Search', markersize=4)

plt.scatter(0, 0, color='magenta', s=50, label='Starting Point')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rosenbrock Function Optimization Paths')
plt.legend()
plt.show()
