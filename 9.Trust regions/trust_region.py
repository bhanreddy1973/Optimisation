# Title: Trust Region Optimization with Steihaug Conjugate Gradient Method
# Summary:
# This script implements the Trust Region Optimization algorithm using the Steihaug Conjugate Gradient (CG) method to solve the Rosenbrock optimization problem.

#------------------------------------------------------------------------------------------------------- 
# It includes the following key components:
# 1. **Rosenbrock Function**: The Rosenbrock function is used as the optimization objective, defined as f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2.
# 2. **Gradient and Hessian**: The gradient and Hessian of the Rosenbrock function are used to guide the optimization process.
# 3. **Trust Region Optimization**: The algorithm iteratively updates the solution within a trust region, adjusting the region's size based on the ratio of actual to predicted reduction.
# 4. **Steihaug Conjugate Gradient**: A conjugate gradient method is used within the trust region to solve for the search direction, ensuring that the step size stays within the trust region.
# 5. **Visualization**: The optimization path is animated on a contour plot of the Rosenbrock function, showing the path of the algorithm, the current point, and the trust region at each iteration.
# 6. **Animation**: The optimization process is visualized with an animated plot, showing how the optimization progresses and how the trust region evolves.
# 7. **Convergence**: The script prints the position and function value at each iteration, and highlights the starting and converged points.
# 8. **Optional**: The animation can be saved as a GIF for further analysis or presentation.

#------------------------------------------------------------------------------------------------------- 
# code :

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

def rosenbrock_function(point):
    x, y = point
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(point):
    x, y = point
    df_dx = -2 * (1 - x) - 400 * x * (y - x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

def rosenbrock_hessian(point):
    x, y = point
    h_xx = 2 - 400 * y + 1200 * x**2
    h_xy = -400 * x
    h_yy = 200
    return np.array([[h_xx, h_xy], [h_xy, h_yy]])

def steihaug_cg(g, H, radius, max_iter=1000, tol=1e-8):
    n = len(g)
    p = np.zeros(n)
    r = g.copy()
    d = -r.copy()
    
    r_norm_sq = r.dot(r)
    
    for _ in range(max_iter):
        if r_norm_sq < tol:
            return p
            
        Hd = H.dot(d)
        dHd = d.dot(Hd)
        
        if dHd <= 0:
            a = p.dot(p)
            b = 2 * p.dot(d)
            c = d.dot(d)
            tau = (-b + np.sqrt(b**2 - 4*a*(c - radius**2))) / (2*c)
            return p + tau * d
            
        alpha = r_norm_sq / dHd
        p_next = p + alpha * d
        
        if np.linalg.norm(p_next) >= radius:
            a = d.dot(d)
            b = 2 * p.dot(d)
            c = p.dot(p) - radius**2
            tau = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            return p + tau * d
            
        r_next = r + alpha * Hd
        r_next_norm_sq = r_next.dot(r_next)
        beta = r_next_norm_sq / r_norm_sq
        
        p = p_next
        r = r_next
        r_norm_sq = r_next_norm_sq
        d = -r + beta * d
        
    return p

def trust_region_optimization(f, grad, hess, x0, initial_radius=1.0, max_radius=10.0, 
                            eta=0.1, epsilon=1e-6, max_iter=1000):
    x = x0
    radius = initial_radius
    positions = [x.copy()]
    radii = [radius]
    
    for i in range(max_iter):
        g = grad(x)
        H = hess(x)
        
        if np.linalg.norm(g) < epsilon:
            break
            
        p = steihaug_cg(g, H, radius)
        
        actual_reduction = f(x) - f(x + p)
        predicted_reduction = -(g.dot(p) + 0.5 * p.dot(H.dot(p)))
        
        if predicted_reduction == 0:
            ratio = 1.0 if actual_reduction == 0 else 0.0
        else:
            ratio = actual_reduction / predicted_reduction
            
        if ratio < 0.25:
            radius = 0.25 * radius
        elif ratio > 0.75 and np.linalg.norm(p) >= 0.99 * radius:
            radius = min(2.0 * radius, max_radius)
            
        if ratio > eta:
            x = x + p
            positions.append(x.copy())
            radii.append(radius)
            print(f'Iteration {i+1}: x = {x[0]:.6f}, y = {x[1]:.6f}, f(x,y) = {f(x):.6f}, radius = {radius:.6f}')
            
    return np.array(positions), np.array(radii)

# Set up the visualization
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(lambda x, y: rosenbrock_function([x, y]))(X, Y)

# Run Trust Region optimization
initial_point = np.array([-2.0, 3.0])
positions, radii = trust_region_optimization(rosenbrock_function, rosenbrock_gradient, 
                                          rosenbrock_hessian, initial_point)

# Create animated visualization
fig, ax = plt.subplots(figsize=(12, 10))

# Plot contour
contour = ax.contour(X, Y, Z, levels=np.logspace(-3, 3, 30), cmap='plasma')
plt.colorbar(contour, label="Function Value (f)")

# Initialize plots that will be animated
path_line, = ax.plot([], [], 'ro-', label="Optimization Path", linewidth=1.5, markersize=3)
current_point, = ax.plot([], [], 'ko', markersize=10, label="Current Point")
trust_region = Circle((0, 0), 0, fill=False, color='red', linestyle='--', alpha=0.5)
ax.add_patch(trust_region)

# Plot start and end points
ax.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label="Starting Point")
ax.scatter(positions[-1, 0], positions[-1, 1], color='blue', s=100, label="Converged Point")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Trust Region Optimization Path")
ax.grid(True)
ax.legend()

def init():
    path_line.set_data([], [])
    current_point.set_data([], [])
    trust_region.center = (0, 0)
    trust_region.radius = 0
    return path_line, current_point, trust_region

def animate(frame):
    # Update path
    path_line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
    
    # Update current point
    current_point.set_data([positions[frame, 0]], [positions[frame, 1]])
    
    # Update trust region
    trust_region.center = (positions[frame, 0], positions[frame, 1])
    trust_region.radius = radii[frame]
    
    return path_line, current_point, trust_region

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(positions),
                    interval=500, blit=True, repeat=True)

plt.show()

# Optional: Save animation
# anim.save('trust_region_optimization.gif', writer='pillow')