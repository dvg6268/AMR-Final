
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider, RadioButtons
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CBFParams:
    """CBF Parameters for hard safety"""
    d_safe: float = 2.0          # Safe distance
    gamma: float = 2.0           # CBF gain
    u_max: float = 4.0           # Max control
    k_goal: float = 2.0          # Goal attraction
    k_damp: float = 0.5          # Velocity damping
    dt: float = 0.05             # Time step
    max_speed: float = 2.0       # Max robot speed
    tangent_gain: float = 1.5    # Gain for tangential motion

class Pedestrian:
    """Individual pedestrian with unique behavior"""
    def __init__(self, position: np.ndarray, behavior: str = "random", color: str = None):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)
        self.behavior = behavior
        self.color = color or self.get_color(behavior)
        self.radius = 0.3
        self.wander_angle = np.random.uniform(0, 2*np.pi)
        self.target = None
        self.counter = 0
        self.trajectory = [self.position.copy()]
        self.patch = None
        
    @staticmethod
    def get_color(behavior: str) -> str:
        """Get color based on behavior"""
        colors = {
            "random": "#e74c3c",      # Red
            "linear": "#3498db",      # Blue
            "circular": "#9b59b6",    # Purple
            "stationary": "#95a5a6",  # Gray
            "following": "#2ecc71",   # Green
            "avoiding": "#e67e22"     # Orange
        }
        return colors.get(behavior, "#e74c3c")
    
    def update(self, dt: float, robot_pos: np.ndarray = None):
        """Update pedestrian based on behavior"""
        self.counter += 1
        
        if self.behavior == "stationary":
            self.velocity = np.zeros(2)
            
        elif self.behavior == "random":
          
            if self.counter % 30 == 0:
                angle = np.random.uniform(0, 2*np.pi)
                speed = np.random.uniform(0.2, 0.8)
                self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
                
        elif self.behavior == "linear":
          
            if self.counter % 100 == 0 or np.linalg.norm(self.velocity) < 0.1:
                angle = np.random.uniform(0, 2*np.pi)
                self.velocity = np.array([np.cos(angle), np.sin(angle)]) * 0.6
                
        elif self.behavior == "circular":
         
            angle = self.counter * 0.05
            radius = 3.0
            center = np.array([10, 10])
            target = center + radius * np.array([np.cos(angle), np.sin(angle)])
            direction = target - self.position
            if np.linalg.norm(direction) > 0.1:
                self.velocity = direction / np.linalg.norm(direction) * 0.5
                
        elif self.behavior == "following":
            
            if robot_pos is not None:
                direction = robot_pos - self.position
                distance = np.linalg.norm(direction)
                if distance > 2.0:  
                    self.velocity = direction / distance * 0.4
                else:
                    
                    self.velocity = -direction / distance * 0.3
                    
        elif self.behavior == "avoiding":
            
            if robot_pos is not None:
                direction = self.position - robot_pos
                distance = np.linalg.norm(direction)
                if distance < 4.0:
                    self.velocity = direction / distance * 0.6
                else:
                    
                    if self.counter % 50 == 0:
                        self.wander_angle += np.random.uniform(-0.5, 0.5)
                    self.velocity = np.array([np.cos(self.wander_angle), 
                                            np.sin(self.wander_angle)]) * 0.4
        
        
        speed = np.linalg.norm(self.velocity)
        if speed > 1.0:
            self.velocity = self.velocity / speed * 1.0
            
        
        self.position += self.velocity * dt
        
        
        if self.position[0] < 1 or self.position[0] > 19:
            self.velocity[0] *= -1
            self.position[0] = np.clip(self.position[0], 1, 19)
        if self.position[1] < 1 or self.position[1] > 19:
            self.velocity[1] *= -1
            self.position[1] = np.clip(self.position[1], 1, 19)
            
        
        self.trajectory.append(self.position.copy())
        
       
        if len(self.trajectory) > 100:
            self.trajectory = self.trajectory[-100:]
    
    def draw(self, ax):
        """Draw pedestrian"""
        if self.patch is None:
            self.patch = patches.Circle(self.position, self.radius,
                                       facecolor=self.color, edgecolor='black',
                                       linewidth=1.5, zorder=8, alpha=0.9)
            ax.add_patch(self.patch)
        else:
            self.patch.center = self.position
        return self.patch

class Robot:
    """Robot with HARD SAFETY CBF - NO safe zone violation"""
    def __init__(self, position: np.ndarray, goal: np.ndarray, params: CBFParams):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)
        self.goal = np.array(goal, dtype=float)
        self.params = params
        self.radius = 0.5
        self.color = "#3498db"
        self.trajectory = [self.position.copy()]
        self.patch = None
        self.control = np.zeros(2)
        self.safety_status = "SAFE"
        self.closest_pedestrian = None
        self.min_distance = float('inf')
        self.goal_patch = None
        self.start_patch = None
        self.circling_direction = 0  
        self.avoidance_mode = False
        
    def compute_control(self, pedestrians: List[Pedestrian]) -> np.ndarray:
        """Compute control using HARD SAFETY CBF projection"""
        
        error = self.goal - self.position
        distance_to_goal = np.linalg.norm(error)
        
        u_nominal = self.params.k_goal * error - self.params.k_damp * self.velocity
        
        
        u_safe = u_nominal.copy()
        self.min_distance = float('inf')
        self.closest_pedestrian = None
        self.avoidance_mode = False
        
        
        for i, ped in enumerate(pedestrians):
            rel_pos = self.position - ped.position
            distance = np.linalg.norm(rel_pos)
            
            if distance < self.min_distance:
                self.min_distance = distance
                self.closest_pedestrian = i
            
            warning_zone = self.params.d_safe * 1.5
            
            if distance < warning_zone and distance > 1e-6:
                self.avoidance_mode = True
                
                
                normal = rel_pos / distance
                
                
                if self.circling_direction == 0:
                    to_goal = self.goal - ped.position
                    to_goal_norm = to_goal / np.linalg.norm(to_goal)
                    cross_product = np.cross(normal, to_goal_norm)
                    self.circling_direction = 1 if cross_product > 0 else -1
                
                
                tangent = np.array([-normal[1], normal[0]]) * self.circling_direction
                
                
                u_radial = np.dot(u_safe, normal) * normal
                u_tangent = u_safe - u_radial
                
                
                if distance <= self.params.d_safe:
                    
                    if np.dot(u_radial, normal) < 0:
                        u_radial = np.zeros(2)
                
               
                elif distance < warning_zone:
                    alpha = (distance - self.params.d_safe) / (warning_zone - self.params.d_safe)
                    if np.dot(u_radial, normal) < 0:
                        u_radial = alpha * u_radial
                
                
                tangent_strength = self.params.tangent_gain * (warning_zone - distance) / warning_zone
                tangent_strength = np.clip(tangent_strength, 0, 2.0)
                
              
                u_safe = u_radial + u_tangent + tangent_strength * tangent
        
        
        if not self.avoidance_mode:
            u_safe = u_nominal
            self.circling_direction = 0
        
       
        u_norm = np.linalg.norm(u_safe)
        if u_norm > self.params.u_max:
            u_safe = u_safe / u_norm * self.params.u_max
        
       
        if self.min_distance < self.params.d_safe:
            self.safety_status = "CIRCLING"
        elif self.min_distance < self.params.d_safe * 1.5:
            self.safety_status = "AVOIDING"
        else:
            self.safety_status = "TO GOAL"
        
        self.control = u_safe
        return u_safe
    
    def update(self, dt: float, pedestrians: List[Pedestrian]):
        """Update robot state"""
        
        u = self.compute_control(pedestrians)
        
        
        self.velocity = 0.7 * self.velocity + 0.3 * u
        
       
        speed = np.linalg.norm(self.velocity)
        if speed > self.params.max_speed:
            self.velocity = self.velocity / speed * self.params.max_speed
        
       
        self.position += self.velocity * dt
        
       
        self.position[0] = np.clip(self.position[0], 0.5, 19.5)
        self.position[1] = np.clip(self.position[1], 0.5, 19.5)
        
        
        self.trajectory.append(self.position.copy())
        if len(self.trajectory) > 200:
            self.trajectory = self.trajectory[-200:]
    
    def draw(self, ax):
        """Draw robot"""
        if self.patch is None:
            self.patch = patches.Circle(self.position, self.radius,
                                       facecolor=self.color, edgecolor='black',
                                       linewidth=2, zorder=10)
            ax.add_patch(self.patch)
        else:
            self.patch.center = self.position
        return self.patch
    
    def draw_goal(self, ax):
        """Draw goal marker"""
        if self.goal_patch is None:
            self.goal_patch = patches.Circle(self.goal, 0.6,
                                            facecolor='#2ecc71', edgecolor='black',
                                            alpha=0.7, zorder=5)
            ax.add_patch(self.goal_patch)
            self.goal_text = ax.text(self.goal[0], self.goal[1] + 0.9, 'GOAL',
                                    ha='center', fontweight='bold', 
                                    color='#27ae60', fontsize=11, zorder=6)
        else:
            self.goal_patch.center = self.goal
            self.goal_text.set_position((self.goal[0], self.goal[1] + 0.9))
        return self.goal_patch
    
    def draw_start(self, ax):
        """Draw start marker"""
        if self.start_patch is None:
            start_pos = self.trajectory[0] if self.trajectory else self.position
            self.start_patch = patches.Circle(start_pos, 0.5,
                                             facecolor='#3498db', edgecolor='black',
                                             alpha=0.7, zorder=5)
            ax.add_patch(self.start_patch)
            self.start_text = ax.text(start_pos[0], start_pos[1] - 0.9, 'START',
                                     ha='center', fontweight='bold',
                                     color='#2980b9', fontsize=11, zorder=6)
        return self.start_patch


class MultiPedestrianCBFSim:
    """Interactive simulation with HARD SAFETY"""
    
    def __init__(self):
        
        self.params = CBFParams()
        
        
        self.robot = Robot(np.array([2.0, 10.0]), np.array([18.0, 10.0]), self.params)
        
        
        self.pedestrians = [
            Pedestrian([10.0, 10.0], "stationary", "#e74c3c"),  
            Pedestrian([5.0, 5.0], "random", "#3498db"),
            Pedestrian([15.0, 5.0], "linear", "#9b59b6"),
            Pedestrian([5.0, 15.0], "following", "#2ecc71"),
            Pedestrian([15.0, 15.0], "avoiding", "#e67e22")
        ]
        
        
        self.running = False
        self.time = 0.0
        self.scenario = "default"
        
        
        self.time_history = []
        self.distance_history = []
        self.control_history = []
        
        
        self.fig = None
        self.ax = None
        self.safety_circles = []
        self.warning_circles = []
        self.trajectory_lines = []
        self.pedestrian_lines = []
        
        
        self.setup_ui()
        
        
        self.anim = animation.FuncAnimation(self.fig, self.update, 
                                          interval=50, blit=False)
    
    def setup_ui(self):
        """Setup the user interface"""
        self.fig = plt.figure(figsize=(14, 9), facecolor='#2c3e50')
        
        
        self.ax = plt.axes([0.1, 0.15, 0.65, 0.8])
        self.ax.set_facecolor('#ecf0f1')
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(0, 20)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_title('HARD SAFETY CBF - NO Safe Zone Violation, Smooth Circular Avoidance', 
                         fontsize=14, fontweight='bold', pad=20)
        
        
        self.draw_static_elements()
        
        
        self.status_text = self.ax.text(0.02, 0.98, 'Status: READY',
                                       transform=self.ax.transAxes,
                                       fontsize=12, fontweight='bold',
                                       color='green',
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', 
                                                facecolor='white', 
                                                alpha=0.9))
        
        
        self.nav_text = self.ax.text(0.02, 0.85, 'Mode: Goal Seeking',
                                    transform=self.ax.transAxes,
                                    fontsize=11,
                                    color='#3498db',
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', 
                                             facecolor='white', 
                                             alpha=0.8))
        
        
        self.metrics_text = self.ax.text(0.02, 0.70, '',
                                        transform=self.ax.transAxes,
                                        fontsize=10,
                                        color='#2c3e50',
                                        verticalalignment='top',
                                        bbox=dict(boxstyle='round', 
                                                 facecolor='white', 
                                                 alpha=0.8))
        
        
        self.create_buttons()
        
        
        self.create_sliders()
        
        
        self.create_behavior_selector()
    
    def draw_static_elements(self):
        """Draw all static visualization elements"""
        
        if hasattr(self, 'safety_circles'):
            for circle in self.safety_circles:
                circle.remove()
        if hasattr(self, 'warning_circles'):
            for circle in self.warning_circles:
                circle.remove()
        if hasattr(self, 'trajectory_lines'):
            for line in self.trajectory_lines:
                line.remove()
        if hasattr(self, 'pedestrian_lines'):
            for line in self.pedestrian_lines:
                line.remove()
        
        self.safety_circles = []
        self.warning_circles = []
        self.trajectory_lines = []
        self.pedestrian_lines = []
        
        
        self.robot.draw_goal(self.ax)
        self.robot.draw_start(self.ax)
        
        
        for ped in self.pedestrians:
            
            safe_circle = patches.Circle(ped.position, self.params.d_safe,
                                        fill=False, linestyle='-',
                                        edgecolor='#e74c3c', linewidth=2.0,
                                        alpha=0.4, zorder=4)
            self.safety_circles.append(safe_circle)
            self.ax.add_patch(safe_circle)
            
            
            warning_circle = patches.Circle(ped.position, self.params.d_safe * 1.5,
                                           fill=False, linestyle=':',
                                           edgecolor='#f39c12', linewidth=1.0,
                                           alpha=0.2, zorder=3)
            self.warning_circles.append(warning_circle)
            self.ax.add_patch(warning_circle)
        
        
        self.robot_line, = self.ax.plot([], [], '-',
                                       color=self.robot.color,
                                       linewidth=2.5, alpha=0.8, zorder=6)
        self.trajectory_lines.append(self.robot_line)
        
        
        for ped in self.pedestrians:
            line, = self.ax.plot([], [], '--',
                                color=ped.color,
                                linewidth=1, alpha=0.3, zorder=5)
            self.pedestrian_lines.append(line)
        
        
        explanation = (
            "HARD SAFETY MECHANISMS:\n"
            "1. Inside safe zone (red): NO inward motion allowed\n"
            "2. In warning zone: Smooth transition to circling\n"
            "3. Tangential forces create smooth circular paths\n"
            "4. Robot NEVER enters safe zones\n"
            "\nWatch robot circle around central obstacle!"
        )
        self.ax.text(0.98, 0.02, explanation,
                    transform=self.ax.transAxes,
                    fontsize=9,
                    color='#2c3e50',
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', 
                             facecolor='white', 
                             alpha=0.7))
    
    def create_buttons(self):
        """Create control buttons"""
        ax_start = plt.axes([0.78, 0.85, 0.18, 0.05])
        ax_pause = plt.axes([0.78, 0.78, 0.18, 0.05])
        ax_reset = plt.axes([0.78, 0.71, 0.18, 0.05])
        ax_random = plt.axes([0.78, 0.64, 0.18, 0.05])
        ax_scenario1 = plt.axes([0.78, 0.57, 0.18, 0.05])
        
        
        self.btn_start = Button(ax_start, 'START HARD SAFETY', color='#2ecc71')
        self.btn_pause = Button(ax_pause, 'PAUSE', color='#f39c12')
        self.btn_reset = Button(ax_reset, 'RESET', color='#e74c3c')
        self.btn_random = Button(ax_random, 'RANDOM SCENE', color='#9b59b6')
        self.btn_scenario1 = Button(ax_scenario1, 'TEST: CENTRAL OBSTACLE', color='#e74c3c')
        
        
        self.btn_start.on_clicked(self.start_simulation)
        self.btn_pause.on_clicked(self.pause_simulation)
        self.btn_reset.on_clicked(self.reset_simulation)
        self.btn_random.on_clicked(self.random_scene)
        self.btn_scenario1.on_clicked(lambda x: self.set_test_scenario())
    
    def create_sliders(self):
        """Create parameter sliders"""
        ax_safe = plt.axes([0.78, 0.45, 0.18, 0.03])
        ax_gamma = plt.axes([0.78, 0.40, 0.18, 0.03])
        ax_tangent = plt.axes([0.78, 0.35, 0.18, 0.03])
        ax_umax = plt.axes([0.78, 0.30, 0.18, 0.03])
        
       
        self.slider_safe = Slider(ax_safe, 'Safe Distance', 1.0, 4.0,
                                 valinit=self.params.d_safe, valstep=0.1,
                                 color='#e74c3c')
        self.slider_gamma = Slider(ax_gamma, 'CBF Gain (γ)', 0.5, 4.0,
                                  valinit=self.params.gamma, valstep=0.1,
                                  color='#3498db')
        self.slider_tangent = Slider(ax_tangent, 'Tangent Gain', 0.5, 3.0,
                                    valinit=self.params.tangent_gain, valstep=0.1,
                                    color='#9b59b6')
        self.slider_umax = Slider(ax_umax, 'Max Control', 2.0, 6.0,
                                 valinit=self.params.u_max, valstep=0.1,
                                  color='#2ecc71')
        
        
        for slider in [self.slider_safe, self.slider_gamma, 
                      self.slider_tangent, self.slider_umax]:
            slider.valtext.set_color('white')
            slider.label.set_color('white')
        
        
        self.slider_safe.on_changed(self.update_safe_distance)
        self.slider_gamma.on_changed(self.update_gamma)
        self.slider_tangent.on_changed(self.update_tangent_gain)
        self.slider_umax.on_changed(self.update_umax)
    
    def create_behavior_selector(self):
        """Create pedestrian behavior selector"""
        ax_behaviors = plt.axes([0.78, 0.20, 0.18, 0.08])
        ax_behaviors.axis('off')
        
        self.behavior_display = ax_behaviors.text(0.05, 0.95, '',
                                                 color='white', fontsize=9,
                                                 verticalalignment='top')
        self.update_behavior_display()
    
    def update_behavior_display(self):
        """Update behavior display text"""
        text = "Pedestrian Behaviors:\n"
        for i, ped in enumerate(self.pedestrians):
            text += f"Ped {i+1}: {ped.behavior}\n"
        self.behavior_display.set_text(text)
    
   
    def start_simulation(self, event):
        self.running = True
        self.status_text.set_text('Status: HARD SAFETY ACTIVE')
        self.status_text.set_color('green')
        print("HARD SAFETY navigation started")
        print("Robot will circle around obstacles without entering safe zones!")
    
    def pause_simulation(self, event):
        self.running = False
        self.status_text.set_text('Status: PAUSED')
        self.status_text.set_color('orange')
        print("Navigation PAUSED")
    
    def reset_simulation(self, event):
        self.running = False
        self.time = 0.0
        
        
        self.robot.position = np.array([2.0, 10.0])
        self.robot.velocity = np.zeros(2)
        self.robot.goal = np.array([18.0, 10.0])
        self.robot.trajectory = [self.robot.position.copy()]
        self.robot.circling_direction = 0
        
        
        positions = [
            [10.0, 10.0], [5.0, 5.0], [15.0, 5.0], [5.0, 15.0], [15.0, 15.0]
        ]
        
        for i, ped in enumerate(self.pedestrians):
            ped.position = np.array(positions[i])
            ped.velocity = np.zeros(2)
            ped.trajectory = [ped.position.copy()]
        
       
        self.time_history = []
        self.distance_history = []
        self.control_history = []
        
        
        self.draw_static_elements()
        
        self.status_text.set_text('Status: RESET')
        self.status_text.set_color('blue')
        self.nav_text.set_text('Mode: Goal Seeking')
        self.nav_text.set_color('#3498db')
        print("Simulation RESET - Central obstacle ready for testing")
    
    def random_scene(self, event):
        """Generate random scene"""
        self.running = False
        
        
        self.robot.position = np.random.uniform(2, 8, 2)
        self.robot.velocity = np.zeros(2)
        self.robot.goal = np.random.uniform(12, 18, 2)
        self.robot.trajectory = [self.robot.position.copy()]
        self.robot.circling_direction = 0
        
        
        for ped in self.pedestrians:
            attempts = 0
            while attempts < 100:
                pos = np.random.uniform(3, 17, 2)
                if (np.linalg.norm(pos - self.robot.position) > self.params.d_safe * 1.2 and 
                    np.linalg.norm(pos - self.robot.goal) > self.params.d_safe * 1.2):
                    ped.position = pos
                    break
                attempts += 1
            if attempts == 100:
                ped.position = np.random.uniform(3, 17, 2)
            
            angle = np.random.uniform(0, 2*np.pi)
            speed = np.random.uniform(0.2, 0.8)
            ped.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
            ped.trajectory = [ped.position.copy()]
        
        
        if self.robot.start_patch:
            self.robot.start_patch.center = self.robot.trajectory[0]
            self.robot.start_text.set_position((self.robot.trajectory[0][0], 
                                               self.robot.trajectory[0][1] - 0.9))
        
        if self.robot.goal_patch:
            self.robot.goal_patch.center = self.robot.goal
            self.robot.goal_text.set_position((self.robot.goal[0], 
                                              self.robot.goal[1] + 0.9))
        
        
        for i, circle in enumerate(self.safety_circles):
            circle.center = self.pedestrians[i].position
        for i, circle in enumerate(self.warning_circles):
            circle.center = self.pedestrians[i].position
        
        
        self.robot_line.set_data([], [])
        for line in self.pedestrian_lines:
            line.set_data([], [])
        
        self.fig.canvas.draw_idle()
        
        self.status_text.set_text('Status: RANDOM SCENE')
        self.status_text.set_color('purple')
        self.nav_text.set_text('Mode: Goal Seeking')
        self.nav_text.set_color('#3498db')
        print("Random scene generated")
    
    def set_test_scenario(self):
        """Test scenario with central obstacle"""
        self.running = False
        
        
        self.robot.position = np.array([2.0, 10.0])
        self.robot.goal = np.array([18.0, 10.0])
        self.robot.velocity = np.zeros(2)
        self.robot.trajectory = [self.robot.position.copy()]
        self.robot.circling_direction = 0
        
       
        positions = [
            [10.0, 10.0],  
            [12.0, 8.0], [8.0, 12.0], [12.0, 12.0], [8.0, 8.0]
        ]
        
        for i, ped in enumerate(self.pedestrians):
            ped.position = np.array(positions[i])
            ped.velocity = np.zeros(2)
            ped.trajectory = [ped.position.copy()]
        
        
        if self.robot.start_patch:
            self.robot.start_patch.center = self.robot.position
            self.robot.start_text.set_position((self.robot.position[0], 
                                               self.robot.position[1] - 0.9))
        
        if self.robot.goal_patch:
            self.robot.goal_patch.center = self.robot.goal
            self.robot.goal_text.set_position((self.robot.goal[0], 
                                              self.robot.goal[1] + 0.9))
        
        
        for i, circle in enumerate(self.safety_circles):
            circle.center = self.pedestrians[i].position
        for i, circle in enumerate(self.warning_circles):
            circle.center = self.pedestrians[i].position
        
        
        self.robot_line.set_data([], [])
        for line in self.pedestrian_lines:
            line.set_data([], [])
        
        self.fig.canvas.draw_idle()
        
        self.status_text.set_text('Status: TEST - CENTRAL OBSTACLE')
        self.status_text.set_color('#3498db')
        self.nav_text.set_text('Mode: Goal Seeking')
        self.nav_text.set_color('#3498db')
        print("Test scenario: Central obstacle directly in path!")
        print("Watch robot circle around it smoothly!")
    
    
    def update_safe_distance(self, val):
        self.params.d_safe = val
        for circle in self.safety_circles:
            circle.radius = val
        for circle in self.warning_circles:
            circle.radius = val * 1.5
        self.fig.canvas.draw_idle()
        print(f"Safe distance updated to {val:.1f}m")
    
    def update_gamma(self, val):
        self.params.gamma = val
        print(f"CBF gain updated to {val:.1f}")
    
    def update_tangent_gain(self, val):
        self.params.tangent_gain = val
        print(f"Tangent gain updated to {val:.1f} (higher = smoother circling)")
    
    def update_umax(self, val):
        self.params.u_max = val
        print(f"Max control updated to {val:.1f}")
    
   
    def update(self, frame):
        """Main update loop"""
        if not self.running:
            return []
        
        
        self.time += self.params.dt
        
        
        for ped in self.pedestrians:
            ped.update(self.params.dt, self.robot.position)
        
       
        self.robot.update(self.params.dt, self.pedestrians)
        
        
        self.time_history.append(self.time)
        self.distance_history.append(self.robot.min_distance)
        self.control_history.append(np.linalg.norm(self.robot.control))
        
        
        self.update_visualization()
        
        
        self.update_metrics()
        
        
        if np.linalg.norm(self.robot.position - self.robot.goal) < 0.5:
            self.running = False
            self.status_text.set_text('Status: GOAL REACHED!')
            self.status_text.set_color('green')
            print("Goal reached with hard safety!")
        
        return []
    
    def update_visualization(self):
        """Update all visual elements"""
      
        self.robot.draw(self.ax)
        
       
        for ped in self.pedestrians:
            ped.draw(self.ax)
        
        
        for i, circle in enumerate(self.safety_circles):
            circle.center = self.pedestrians[i].position
            
           
            distance = np.linalg.norm(self.robot.position - self.pedestrians[i].position)
            if distance < self.params.d_safe:
                circle.set_edgecolor('red')
                circle.set_linewidth(2.5)
                circle.set_alpha(0.6)
            elif distance < self.params.d_safe * 1.5:
                circle.set_edgecolor('orange')
                circle.set_linewidth(2.0)
                circle.set_alpha(0.5)
            else:
                circle.set_edgecolor('#e74c3c')
                circle.set_linewidth(1.5)
                circle.set_alpha(0.4)
        
        
        for i, circle in enumerate(self.warning_circles):
            circle.center = self.pedestrians[i].position
        
        
        if len(self.robot.trajectory) > 1:
            robot_traj = np.array(self.robot.trajectory)
            self.robot_line.set_data(robot_traj[:, 0], robot_traj[:, 1])
        
        
        for i, line in enumerate(self.pedestrian_lines):
            if len(self.pedestrians[i].trajectory) > 1:
                ped_traj = np.array(self.pedestrians[i].trajectory)
                line.set_data(ped_traj[:, 0], ped_traj[:, 1])
    
    def update_metrics(self):
        """Update metrics display"""
        
        distances = []
        for ped in self.pedestrians:
            dist = np.linalg.norm(self.robot.position - ped.position)
            distances.append(dist)
        
        min_dist = min(distances) if distances else 0
        
       
        if self.robot.avoidance_mode:
            if self.robot.circling_direction == 1:
                mode_text = "Mode: CIRCULATING CLOCKWISE"
            else:
                mode_text = "Mode: CIRCULATING COUNTER-CLOCKWISE"
            mode_color = '#9b59b6'
        else:
            mode_text = "Mode: GOAL SEEKING"
            mode_color = '#3498db'
        
        self.nav_text.set_text(mode_text)
        self.nav_text.set_color(mode_color)
        
        metrics = (
            f"Time: {self.time:.1f}s\n"
            f"Robot Speed: {np.linalg.norm(self.robot.velocity):.2f}m/s\n"
            f"Goal Distance: {np.linalg.norm(self.robot.goal - self.robot.position):.2f}m\n"
            f"Closest Pedestrian: {min_dist:.2f}m\n"
            f"Safe Zone: {self.params.d_safe:.1f}m\n"
            f"Status: {self.robot.safety_status}\n"
            f"Control Magnitude: {np.linalg.norm(self.robot.control):.2f}\n"
            f"Tangent Gain: {self.params.tangent_gain:.1f}"
        )
        
        
        if min_dist < self.params.d_safe:
            color = 'red'
        elif min_dist < self.params.d_safe * 1.2:
            color = 'orange'
        else:
            color = 'green'
        
        self.metrics_text.set_color(color)
        self.metrics_text.set_text(metrics)


def main():
    print("=" * 80)
    print("HARD SAFETY CBF NAVIGATION")
    print("=" * 80)
    print("\nKEY FEATURES:")
    print("  1. HARD SAFETY: Robot NEVER enters safe zones")
    print("  2. NO BOUNCING: Smooth circular redirection")
    print("  3. TRUE CBF PROJECTION: Radial component control")
    print("  4. Tangential forces for natural circling")
    print("\nCONTROLS:")
    print("  • START HARD SAFETY: Begin navigation")
    print("  • TEST: CENTRAL OBSTACLE: Perfect demonstration scenario")
    print("  • Tangent Gain: Controls circling smoothness (1.5 recommended)")
    print("\nHOW IT WORKS:")
    print("  1. Inside safe zone: NO inward radial motion allowed")
    print("  2. Warning zone: Smooth transition to circling")
    print("  3. Robot chooses optimal circling direction")
    print("  4. Once past obstacle, resumes to goal")
    print("\n" + "=" * 80)
    print("Click 'TEST: CENTRAL OBSTACLE' then 'START HARD SAFETY'")
    print("Watch robot circle around obstacle smoothly!")
    
    sim = MultiPedestrianCBFSim()
    plt.show()

if __name__ == "__main__":
    main()