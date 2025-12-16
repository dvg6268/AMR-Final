import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec


fig = plt.figure(figsize=(14, 6))
fig.suptitle('Potential Fields Simulation - Robot Navigation', fontsize=16, fontweight='bold')


gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])


ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])


def attractive_potential(x, y, goal, k_att=0.5):
    return 0.5 * k_att * ((x - goal[0])**2 + (y - goal[1])**2)

def repulsive_potential(x, y, obstacles, k_rep=20.0, d0=1.5):
    pot = 0
    for obs in obstacles:
        dist = np.sqrt((x - obs[0])**2 + (y - obs[1])**2)
        if dist < d0:
            pot += 0.5 * k_rep * (1/dist - 1/d0)**2
    return pot

def total_potential(x, y, goal, obstacles):
    return attractive_potential(x, y, goal, k_att=0.5) + repulsive_potential(x, y, obstacles, k_rep=20.0)


x = np.linspace(-1, 6, 100)
y = np.linspace(-1, 6, 100)
X, Y = np.meshgrid(x, y)


start1 = np.array([0.5, 0.5])
goal1 = np.array([5, 5])
obstacles1 = []  
robot1 = start1.copy()
path1 = [start1.copy()]


Z1 = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        Z1[j, i] = attractive_potential(X[j, i], Y[j, i], goal1, k_att=0.5)


contour1 = ax1.contourf(X, Y, Z1, levels=20, alpha=0.6, cmap='viridis')
ax1.set_xlim(-1, 6)
ax1.set_ylim(-1, 6)
ax1.set_title('Scenario 1: Reaches Goal\n(Clear Path)', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')


ax1.plot(start1[0], start1[1], 'go', markersize=20, label='Start (A)', markeredgecolor='black')
ax1.plot(goal1[0], goal1[1], 'yo', markersize=20, label='Goal (B)', markeredgecolor='black')
robot_dot1, = ax1.plot([], [], 'wo', markersize=15, markeredgecolor='black', label='Robot')
robot_path_line1, = ax1.plot([], [], 'b-', linewidth=3, alpha=0.8, label='Path')


start2 = np.array([0.5, 0.5])
goal2 = np.array([5, 5])


obstacles2 = [
    (2.5, 2.5),  
    (3.0, 3.0),  
]

robot2 = start2.copy()
path2 = [start2.copy()]


Z2 = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        Z2[j, i] = total_potential(X[j, i], Y[j, i], goal2, obstacles2)


contour2 = ax2.contourf(X, Y, Z2, levels=20, alpha=0.6, cmap='viridis')
ax2.set_xlim(-1, 6)
ax2.set_ylim(-1, 6)
ax2.set_title('Scenario 2: Stuck in Local Minimum\n(Obstacle on Path)', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')


for obs in obstacles2:
    ax2.add_patch(Circle(obs, radius=0.4, color='red', alpha=0.8, label='Obstacle'))


ax2.plot(start2[0], start2[1], 'go', markersize=20, label='Start (A)', markeredgecolor='black')
ax2.plot(goal2[0], goal2[1], 'yo', markersize=20, label='Goal (B)', markeredgecolor='black')
robot_dot2, = ax2.plot([], [], 'wo', markersize=15, markeredgecolor='black', label='Robot')
robot_path_line2, = ax2.plot([], [], 'b-', linewidth=3, alpha=0.8, label='Path')


ax1.legend(loc='upper left')
ax2.legend(loc='upper left')


def simple_gradient_step(position, goal, obstacles, step_size=0.15):
    h = 0.01
    x, y = position
    
  
    grad_x = (total_potential(x + h, y, goal, obstacles) - 
              total_potential(x - h, y, goal, obstacles)) / (2*h)
    grad_y = (total_potential(x, y + h, goal, obstacles) - 
              total_potential(x, y - h, goal, obstacles)) / (2*h)
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
   
    if grad_mag < 0.05:
        return position, True
    
   
    if grad_mag > 0:
        grad_x /= grad_mag
        grad_y /= grad_mag
    
    new_position = position - step_size * np.array([grad_x, grad_y])
    new_position[0] = np.clip(new_position[0], -0.5, 5.5)
    new_position[1] = np.clip(new_position[1], -0.5, 5.5)
    
    return new_position, False


step_count = 0
stuck_in_minimum = False
success1 = False


def update(frame):
    global robot1, robot2, step_count, stuck_in_minimum, success1
    
   
    if not success1:
        direction = goal1 - robot1
        dist = np.linalg.norm(direction)
        if dist > 0.1:
            direction = direction / dist
            robot1 = robot1 + 0.2 * direction
            path1.append(robot1.copy())
        else:
            success1 = True
    
    
    if not stuck_in_minimum:
        new_pos, stuck = simple_gradient_step(robot2, goal2, obstacles2, step_size=0.12)
        
        if stuck:
            stuck_in_minimum = True
            robot2 = new_pos  
        else:
            robot2 = new_pos
            path2.append(robot2.copy())
        
        step_count += 1
    
    
    if len(path1) > 0:
        path_array1 = np.array(path1)
        robot_dot1.set_data([robot1[0]], [robot1[1]])
        robot_path_line1.set_data(path_array1[:, 0], path_array1[:, 1])
    
    
    if len(path2) > 0:
        path_array2 = np.array(path2)
        robot_dot2.set_data([robot2[0]], [robot2[1]])
        robot_path_line2.set_data(path_array2[:, 0], path_array2[:, 1])
    
   
    if success1 and not hasattr(update, 'success_shown'):
        ax1.text(3, 0, 'SUCCESS!', ha='center', color='green', 
                fontweight='bold', fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        update.success_shown = True
    
    if stuck_in_minimum and not hasattr(update, 'stuck_shown'):
        ax2.text(robot2[0], robot2[1]+0.5, 'STUCK!', ha='center', color='red', 
                fontweight='bold', fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        
        ax2.plot(robot2[0], robot2[1], 'rx', markersize=20, markeredgewidth=3)
        
       
        distance = np.linalg.norm(robot2 - goal2)
        ax2.text(3, 0, f'Distance to goal: {distance:.1f}', ha='center', color='yellow',
                fontweight='bold', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        update.stuck_shown = True
    
    
    ax1.text(0.5, 5.8, f'Step: {step_count}', ha='left', color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7))
    ax2.text(0.5, 5.8, f'Step: {step_count}', ha='left', color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7))
    
    return [robot_dot1, robot_path_line1, robot_dot2, robot_path_line2]


ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=100,  
    interval=100,  
    blit=False, 
    repeat=False
)

plt.tight_layout()
plt.show()


print("="*50)
print("SIMULATION RESULTS")
print("="*50)
print("\nScenario 1: Clear Path")
print(f"  Start: ({start1[0]}, {start1[1]})")
print(f"  Goal: ({goal1[0]}, {goal1[1]})")
print(f"  Result: SUCCESS - Reached goal")
print(f"  Steps taken: {len(path1)}")

print("\nScenario 2: Obstacle on Path")
print(f"  Start: ({start2[0]}, {start2[1]})")
print(f"  Goal: ({goal2[0]}, {goal2[1]})")
print(f"  Obstacle positions: {obstacles2}")
print(f"  Result: FAILED - Stuck in local minimum")
print(f"  Stuck at: ({robot2[0]:.2f}, {robot2[1]:.2f})")
print(f"  Distance from goal: {np.linalg.norm(robot2 - goal2):.2f}")
print(f"  Steps taken before stuck: {len(path2)}")
print("="*50)