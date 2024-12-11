# Title: Rosenbrock Function Optimization using NGD, MGD, and Constant Learning Rate GD

# Description:
# This code visualizes the optimization paths of three different gradient descent algorithms 
# applied to the Rosenbrock function, a commonly used test function in optimization.
# The Rosenbrock function is often referred to as the "banana function" due to its shape, 
# and it is used to test optimization algorithms due to its challenging nature with a narrow, curved valley.

# Key Components:
# 1. Rosenbrock Function:
#    The Rosenbrock function takes a 2D point (x, y) and calculates the function value as:
#    (1 - x)^2 + 100 * (y - x^2)^2.
#    The function has a global minimum at (1, 1), where the function value is 0.

#------------------------------------------------------------------------------------------------------- 

# 2. Gradient Calculation:
#    The gradient of the Rosenbrock function is computed for each point, which consists of 
#    partial derivatives with respect to x and y:
#    - df_dx = -2 * (1 - x) - 400 * x * (y - x^2)
#    - df_dy = 200 * (y - x^2)

#------------------------------------------------------------------------------------------------------- 

# 3. Optimization Algorithms:
#    - Nesterov Gradient Descent (NGD): 
#      This method uses a lookahead step based on the previous velocity (momentum) before 
#      calculating the gradient, which can lead to faster convergence.
#    - Momentum Gradient Descent (MGD):
#      This is a variant of gradient descent that adds a momentum term, which helps accelerate 
#      convergence in the relevant direction and dampens oscillations.
#    - Constant Learning Rate Gradient Descent:
#      This is the classic gradient descent method where the learning rate is fixed throughout 
#      the optimization process.

#------------------------------------------------------------------------------------------------------- 

# 4. Visualization:
#    The code plots the Rosenbrock function as a contour plot, showing the optimization paths 
#    of the three algorithms:
#    - NGD is represented by a red line with circle markers.
#    - MGD is represented by a blue line with square markers.
#    - Constant Learning Rate GD is represented by a black line with square markers.
#    The starting point is marked in magenta.

#------------------------------------------------------------------------------------------------------- 

# Final Output:
# A plot showing the optimization paths of all three algorithms on the Rosenbrock function 
# and their respective convergence to the global minimum.


#------------------------------------------------------------------------------------------------------- 

# Code begins below:


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


# Grid for contour plot
x = np.linspace(-4, 4, 500)
y = np.linspace(-4, 4, 500)
X, Y = np.meshgrid(x, y)
Z = rosenbrock_function([X, Y])

# NGD and MGD initialization
start_pos = np.array([0, 0])  # Challenging start point

# Nesterov Gradient Descent (NGD)
current_pos_ngd = start_pos.copy()
NGD_line = [current_pos_ngd.copy()]
velocity_ngd = np.array([0, 0])

for _ in range(200):
    learning_rate = 0.0002
    gamma = 0.75

    x_lookahead = current_pos_ngd - gamma * velocity_ngd
    gradient = rosenbrock_gradient(x_lookahead)

    velocity_ngd = velocity_ngd + learning_rate * gradient
    current_pos_ngd = current_pos_ngd - velocity_ngd

    NGD_line.append(current_pos_ngd.copy())

# Momentum Gradient Descent (MGD)
current_pos_mgd = start_pos.copy()
MGD_line = [current_pos_mgd.copy()]
velocity_mgd = np.array([0.0, 0.0])

for _ in range(200):
    learning_rate = 0.0002
    gamma = 0.75

    gradient = rosenbrock_gradient(current_pos_mgd)
    velocity_mgd = gamma * velocity_mgd + learning_rate * gradient
    current_pos_mgd = current_pos_mgd - velocity_mgd

    MGD_line.append(current_pos_mgd.copy())

# Constant Learning Rate Gradient Descent
current_pos_gd = start_pos.copy()
constant_alpha_line = [current_pos_gd.copy()]

for _ in range(200):
    gradient = rosenbrock_gradient(current_pos_gd)
    learning_rate = 0.0002
    current_pos_gd = current_pos_gd - learning_rate * gradient
    constant_alpha_line.append(current_pos_gd.copy())

# Plotting
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.logspace(-1, 4, 13), cmap='viridis')

NGD_line = np.array(NGD_line)
MGD_line = np.array(MGD_line)
constant_alpha_line = np.array(constant_alpha_line)

# Plot optimization paths
plt.plot(NGD_line[:, 0], NGD_line[:, 1], 'r-o', label='NGD Line', markersize=4)
plt.plot(MGD_line[:, 0], MGD_line[:, 1], 'b-s', label='MGD Line', markersize=4)
plt.plot(constant_alpha_line[:, 0], constant_alpha_line[:,
         1], 'k-s', label='Constant Alpha', markersize=4)

# Mark starting point
plt.scatter(start_pos[0], start_pos[1], color='magenta',
            s=50, label='Starting Point')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rosenbrock Function Optimization Paths')
plt.legend()
plt.show()
