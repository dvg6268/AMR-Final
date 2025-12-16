import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

class MathUtils:
    @staticmethod
    def angle_wrap(angle: float) -> float:
        return (angle + np.pi) % (2*np.pi) - np.pi
    
    @staticmethod
    def rotation_matrix_2d(angle: float) -> np.ndarray:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    @staticmethod
    def bound_value(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
    
    @staticmethod
    def smooth_saturation(value: float, threshold: float) -> float:
        if abs(value) < threshold:
            return value / threshold
        return np.sign(value) if value != 0 else 0.0

@dataclass
class VehicleConfiguration:
    wheelbase: float = 0.35
    track_width: float = 0.28
    wheel_radius: float = 0.06
    cog_to_front: float = 0.175
    cog_to_rear: float = 0.175
    vehicle_mass: float = 12.0
    yaw_inertia: float = 0.8
    long_damping: float = 4.0
    lat_damping: float = 8.0
    yaw_damping: float = 2.0
    front_cornering_stiffness: float = 150.0
    rear_cornering_stiffness: float = 180.0
    friction_coefficient: float = 0.9
    gravity: float = 9.81
    chassis_length: float = 0.45
    chassis_width: float = 0.30
    wheel_display_length: float = 0.10
    wheel_display_width: float = 0.04

@dataclass
class SimulationSettings:
    time_step: float = 0.01
    max_steering_angle: float = 45.0

@dataclass
class SMCParameters:
    max_target_velocity: float = 4.0
    max_lateral_accel: float = 3.0
    sliding_gain: float = 1.2
    steering_gain: float = 0.8
    lateral_boundary: float = 0.25
    velocity_gain: float = 250.0
    velocity_boundary: float = 0.4
    max_wheel_torque: float = 8.0
    max_steer_angle: float = 45.0
    search_range: float = 3.0
    search_samples: int = 41

class TrajectoryManager:
    def __init__(self, waypoints: np.ndarray, closed_loop: bool = False):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.is_closed = closed_loop
        
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 2:
            raise ValueError("Waypoints must be Nx2 array")
        if len(self.waypoints) < 3:
            raise ValueError("At least 3 waypoints required")
        
        if closed_loop and np.linalg.norm(self.waypoints[0] - self.waypoints[-1]) > 1e-9:
            self.waypoints = np.vstack([self.waypoints, self.waypoints[0]])
        
        segment_lengths = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.cumulative_distance = np.hstack([0.0, np.cumsum(segment_lengths)])
        self.total_path_length = float(self.cumulative_distance[-1])
        
        boundary_type = "periodic" if closed_loop else "natural"
        self.x_spline = CubicSpline(self.cumulative_distance, self.waypoints[:, 0], 
                                   bc_type=boundary_type)
        self.y_spline = CubicSpline(self.cumulative_distance, self.waypoints[:, 1],
                                   bc_type=boundary_type)
    
    def get_path_point(self, distance: float) -> Tuple[float, float, float, float]:
        if not self.is_closed:
            distance = MathUtils.bound_value(distance, 0.0, self.total_path_length)
        else:
            distance = distance % self.total_path_length
        
        x = float(self.x_spline(distance))
        y = float(self.y_spline(distance))
        
        dx = float(self.x_spline(distance, 1))
        dy = float(self.y_spline(distance, 1))
        
        ddx = float(self.x_spline(distance, 2))
        ddy = float(self.y_spline(distance, 2))
        
        heading = np.arctan2(dy, dx)
        
        denominator = (dx*dx + dy*dy)**1.5 + 1e-12
        curvature = (dx*ddy - dy*ddx) / denominator
        
        return x, y, heading, curvature
    
    def find_closest_point(self, x: float, y: float, prev_distance: float, 
                          search_range: float, num_samples: int) -> float:
        if not self.is_closed:
            start_dist = max(0.0, prev_distance - search_range)
            end_dist = min(self.total_path_length, prev_distance + search_range)
            search_distances = np.linspace(start_dist, end_dist, num_samples)
        else:
            search_distances = np.linspace(prev_distance - search_range, 
                                          prev_distance + search_range, num_samples)
            search_distances = [self._wrap_distance(d) for d in search_distances]
        
        best_distance = self._wrap_distance(prev_distance) if self.is_closed else prev_distance
        min_distance_squared = float('inf')
        
        for dist in search_distances:
            px, py, _, _ = self.get_path_point(dist)
            distance_squared = (x - px)**2 + (y - py)**2
            
            if distance_squared < min_distance_squared:
                min_distance_squared = distance_squared
                best_distance = dist
        
        return best_distance
    
    def _wrap_distance(self, distance: float) -> float:
        if not self.is_closed:
            return MathUtils.bound_value(distance, 0.0, self.total_path_length)
        return distance % self.total_path_length

class VehicleDynamics:
    def __init__(self, config: VehicleConfiguration):
        self.config = config
        
        half_track = config.track_width / 2
        self.wheel_positions = {
            'rear_left': np.array([-config.cog_to_rear, half_track]),
            'rear_right': np.array([-config.cog_to_rear, -half_track]),
            'front_left': np.array([config.cog_to_front, half_track]),
            'front_right': np.array([config.cog_to_front, -half_track])
        }
    
    def compute_steering_angles(self, curvature: float, max_steer_deg: float) -> Tuple[float, float]:
        max_steer = np.radians(max_steer_deg)
        half_track = self.config.track_width / 2
        
        if abs(curvature) < 1e-12:
            return 0.0, 0.0
        
        turning_radius = 1.0 / curvature
        abs_radius = abs(turning_radius)
        
        if abs_radius <= half_track + 1e-6:
            abs_radius = half_track + 1e-6
        
        inner_angle = np.arctan(self.config.wheelbase / (abs_radius - half_track))
        outer_angle = np.arctan(self.config.wheelbase / (abs_radius + half_track))
        
        if curvature > 0:
            left_angle = inner_angle
            right_angle = outer_angle
        else:
            left_angle = outer_angle
            right_angle = inner_angle
        
        left_angle = MathUtils.bound_value(left_angle, -max_steer, max_steer)
        right_angle = MathUtils.bound_value(right_angle, -max_steer, max_steer)
        
        return left_angle, right_angle
    
    def compute_inertia_matrix(self) -> np.ndarray:
        return np.diag([self.config.vehicle_mass, 
                       self.config.vehicle_mass, 
                       self.config.yaw_inertia])
    
    def compute_coriolis_matrix(self, velocities: np.ndarray) -> np.ndarray:
        u, v, r = velocities
        m = self.config.vehicle_mass
        
        return np.array([[0, -m*r, 0],
                        [m*r, 0, 0],
                        [0, 0, 0]], dtype=float)
    
    def compute_damping_matrix(self) -> np.ndarray:
        return np.diag([self.config.long_damping,
                       self.config.lat_damping,
                       self.config.yaw_damping])
    
    def compute_force_transformation(self, steering_angles: Dict[str, float]) -> np.ndarray:
        half_track = self.config.track_width / 2
        wheel_keys = ['rear_left', 'rear_right', 'front_left', 'front_right']
        
        transformation = np.zeros((3, 4), dtype=float)
        
        for idx, key in enumerate(wheel_keys):
            pos_x, pos_y = self.wheel_positions[key]
            steer = steering_angles[key]
            
            cos_steer = np.cos(steer)
            sin_steer = np.sin(steer)
            
            transformation[0, idx] = cos_steer
            transformation[1, idx] = sin_steer
            transformation[2, idx] = pos_x * sin_steer - pos_y * cos_steer
            
            transformation[:, idx] /= self.config.wheel_radius
        
        return transformation
    
    def compute_tire_forces(self, body_velocities: np.ndarray, 
                           average_steer: float, min_speed: float = 0.2) -> Tuple[np.ndarray, Tuple[float, float]]:
        u, v, r = body_velocities
        
        effective_speed = u if abs(u) > min_speed else (min_speed * np.sign(u) if abs(u) > 1e-6 else min_speed)
        
        front_slip = np.arctan2(v + self.config.cog_to_front * r, effective_speed) - average_steer
        rear_slip = np.arctan2(v - self.config.cog_to_rear * r, effective_speed)
        
        front_lateral = -self.config.front_cornering_stiffness * front_slip
        rear_lateral = -self.config.rear_cornering_stiffness * rear_slip
        
        front_longitudinal = -front_lateral * np.sin(average_steer)
        front_lateral_body = front_lateral * np.cos(average_steer)
        
        rear_longitudinal = 0.0
        rear_lateral_body = rear_lateral
        
        total_longitudinal = front_longitudinal + rear_longitudinal
        total_lateral = front_lateral_body + rear_lateral_body
        total_yaw_moment = (self.config.cog_to_front * front_lateral_body - 
                          self.config.cog_to_rear * rear_lateral_body)
        
        max_force = self.config.friction_coefficient * self.config.vehicle_mass * self.config.gravity
        total_force = np.hypot(total_longitudinal, total_lateral)
        
        if total_force > max_force and total_force > 1e-12:
            scaling = max_force / total_force
            total_longitudinal *= scaling
            total_lateral *= scaling
        
        forces = np.array([total_longitudinal, total_lateral, total_yaw_moment])
        slip_angles = (front_slip, rear_slip)
        
        return forces, slip_angles

class AdvancedSMCController:
    def __init__(self, trajectory: TrajectoryManager, 
                 vehicle_config: VehicleConfiguration,
                 sim_settings: SimulationSettings,
                 controller_params: SMCParameters):
        
        self.trajectory = trajectory
        self.vehicle_config = vehicle_config
        self.sim_settings = sim_settings
        self.params = controller_params
        
        self.previous_path_distance = 0.0
        
        self.error_history = []
        self.sliding_history = []
        self.phase_data = []
        
    def compute_control(self, time: float, vehicle_state: Tuple, 
                       prev_distance: float) -> Tuple[Dict, Dict, Dict, float]:
        x, y, heading, body_velocities = vehicle_state
        u, v, r = body_velocities
        
        current_distance = self.trajectory.find_closest_point(
            x, y, prev_distance, 
            self.params.search_range, 
            self.params.search_samples
        )
        
        ref_x, ref_y, ref_heading, curvature = self.trajectory.get_path_point(current_distance)
        
        dx, dy = x - ref_x, y - ref_y
        
        cross_track_error = -dx * np.sin(ref_heading) + dy * np.cos(ref_heading)
        
        along_track_error = dx * np.cos(ref_heading) + dy * np.sin(ref_heading)
        
        heading_error = MathUtils.angle_wrap(heading - ref_heading)
        
        current_errors = np.array([along_track_error, cross_track_error, heading_error])
        self.error_history.append(current_errors)
        
        target_velocity = min(self.params.max_target_velocity,
                            np.sqrt(self.params.max_lateral_accel / (abs(curvature) + 1e-3)))
        
        lateral_sliding = heading_error + self.params.sliding_gain * cross_track_error
        self.sliding_history.append(lateral_sliding)
        
        if len(self.error_history) > 1:
            dt = self.sim_settings.time_step
            current_errors = np.array(self.error_history[-1])
            previous_errors = np.array(self.error_history[-2])
            error_rate = (current_errors - previous_errors) / dt
            self.phase_data.append((cross_track_error, error_rate[1]))
        
        feedforward_steer = np.arctan(self.vehicle_config.wheelbase * curvature)
        
        smooth_correction = MathUtils.smooth_saturation(lateral_sliding, self.params.lateral_boundary)
        corrected_steer = feedforward_steer - self.params.steering_gain * smooth_correction
        
        max_steer_rad = np.radians(self.params.max_steer_angle)
        corrected_steer = MathUtils.bound_value(corrected_steer, -max_steer_rad, max_steer_rad)
        
        effective_curvature = np.tan(corrected_steer) / self.vehicle_config.wheelbase
        left_steer, right_steer = VehicleDynamics(self.vehicle_config).compute_steering_angles(
            effective_curvature, self.params.max_steer_angle)
        
        velocity_error = u - target_velocity
        velocity_sliding = MathUtils.smooth_saturation(velocity_error, self.params.velocity_boundary)
        desired_force = self.vehicle_config.long_damping * u - self.params.velocity_gain * velocity_sliding
        
        wheel_torque = 0.5 * self.vehicle_config.wheel_radius * desired_force
        wheel_torque = MathUtils.bound_value(wheel_torque, -self.params.max_wheel_torque, 
                                           self.params.max_wheel_torque)
        
        steering_commands = {
            'rear_left': 0.0, 'rear_right': 0.0,
            'front_left': left_steer, 'front_right': right_steer
        }
        
        torque_commands = {
            'rear_left': wheel_torque, 'rear_right': wheel_torque,
            'front_left': 0.0, 'front_right': 0.0
        }
        
        debug_info = {
            'path_distance': current_distance,
            'reference_point': (ref_x, ref_y),
            'reference_heading': ref_heading,
            'curvature': curvature,
            'cross_track_error': cross_track_error,
            'heading_error': heading_error,
            'lateral_sliding': lateral_sliding,
            'target_velocity': target_velocity,
            'corrected_steer': corrected_steer,
            'desired_force': desired_force
        }
        
        return steering_commands, torque_commands, debug_info, current_distance

class SimulationEngine:
    def __init__(self, vehicle_config: VehicleConfiguration, 
                 sim_settings: SimulationSettings):
        self.config = vehicle_config
        self.settings = sim_settings
        self.dynamics = VehicleDynamics(vehicle_config)
        
        self.inertia_matrix = self.dynamics.compute_inertia_matrix()
        self.damping_matrix = self.dynamics.compute_damping_matrix()
        
    def run_simulation(self, total_time: float, controller) -> Dict:
        dt = self.settings.time_step
        num_steps = int(np.ceil(total_time / dt))
        
        time = np.arange(num_steps) * dt
        position_x = np.zeros(num_steps)
        position_y = np.zeros(num_steps)
        heading = np.zeros(num_steps)
        velocity_long = np.zeros(num_steps)
        velocity_lat = np.zeros(num_steps)
        yaw_rate = np.zeros(num_steps)
        
        steering_log = {key: np.zeros(num_steps) for key in 
                       ['rear_left', 'rear_right', 'front_left', 'front_right']}
        torque_log = {key: np.zeros(num_steps) for key in 
                     ['rear_left', 'rear_right', 'front_left', 'front_right']}
        
        cross_track_error = np.zeros(num_steps)
        heading_error = np.zeros(num_steps)
        lateral_sliding = np.zeros(num_steps)
        target_velocity = np.zeros(num_steps)
        path_curvature = np.zeros(num_steps)
        
        path_distance = 0.0
        
        for step in range(1, num_steps):
            body_velocities = np.array([velocity_long[step-1], 
                                       velocity_lat[step-1], 
                                       yaw_rate[step-1]])
            state = (position_x[step-1], position_y[step-1], 
                    heading[step-1], body_velocities)
            
            steering, torque, debug, path_distance = controller.compute_control(
                time[step-1], state, path_distance)
            
            cross_track_error[step-1] = debug['cross_track_error']
            heading_error[step-1] = debug['heading_error']
            lateral_sliding[step-1] = debug['lateral_sliding']
            target_velocity[step-1] = debug['target_velocity']
            path_curvature[step-1] = debug['curvature']
            
            for key in steering_log:
                steering_log[key][step-1] = steering[key]
                torque_log[key][step-1] = torque[key]
            
            avg_steer = 0.5 * (steering['front_left'] + steering['front_right'])
            
            tire_forces, _ = self.dynamics.compute_tire_forces(body_velocities, avg_steer)
            
            force_matrix = self.dynamics.compute_force_transformation(steering)
            torque_vector = np.array([torque[key] for key in 
                                    ['rear_left', 'rear_right', 'front_left', 'front_right']])
            wheel_force = force_matrix @ torque_vector
            
            total_force = wheel_force + tire_forces
            
            coriolis_matrix = self.dynamics.compute_coriolis_matrix(body_velocities)
            
            right_hand_side = total_force - coriolis_matrix @ body_velocities - \
                            self.damping_matrix @ body_velocities
            acceleration = np.linalg.solve(self.inertia_matrix, right_hand_side)
            
            velocity_long[step] = velocity_long[step-1] + acceleration[0] * dt
            velocity_lat[step] = velocity_lat[step-1] + acceleration[1] * dt
            yaw_rate[step] = yaw_rate[step-1] + acceleration[2] * dt
            
            rotation = MathUtils.rotation_matrix_2d(heading[step-1])
            global_velocity = rotation @ np.array([velocity_long[step-1], velocity_lat[step-1]])
            
            position_x[step] = position_x[step-1] + global_velocity[0] * dt
            position_y[step] = position_y[step-1] + global_velocity[1] * dt
            heading[step] = MathUtils.angle_wrap(heading[step-1] + yaw_rate[step-1] * dt)
        
        cross_track_error[-1] = cross_track_error[-2]
        heading_error[-1] = heading_error[-2]
        lateral_sliding[-1] = lateral_sliding[-2]
        target_velocity[-1] = target_velocity[-2]
        path_curvature[-1] = path_curvature[-2]
        
        for key in steering_log:
            steering_log[key][-1] = steering_log[key][-2]
            torque_log[key][-1] = torque_log[key][-2]
        
        results = {
            'time': time,
            'position': {'x': position_x, 'y': position_y},
            'orientation': heading,
            'velocities': {'longitudinal': velocity_long, 
                          'lateral': velocity_lat, 
                          'yaw': yaw_rate},
            'steering': steering_log,
            'torque': torque_log,
            'tracking': {
                'cross_track_error': cross_track_error,
                'heading_error': heading_error,
                'lateral_sliding': lateral_sliding,
                'target_velocity': target_velocity,
                'curvature': path_curvature
            },
            'vehicle_config': self.config,
            'simulation_settings': self.settings
        }
        
        return results

class ResultsAnalyzer:
    @staticmethod
    def plot_phase_portrait(controller):
        if not controller.phase_data:
            print("No phase data available")
            return
        
        phase_data = np.array(controller.phase_data)
        errors = phase_data[:, 0]
        error_rates = phase_data[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        sc1 = axes[0, 0].scatter(errors, error_rates, c=np.arange(len(errors)), 
                                cmap='viridis', s=15, alpha=0.7)
        axes[0, 0].set_xlabel('Cross-track Error e [m]', fontsize=12)
        axes[0, 0].set_ylabel('Error Rate de/dt [m/s]', fontsize=12)
        axes[0, 0].set_title('Phase Portrait: Error Dynamics', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.5)
        plt.colorbar(sc1, ax=axes[0, 0], label='Time Step')
        
        if controller.sliding_history:
            axes[0, 1].plot(range(len(controller.sliding_history)), controller.sliding_history, 
                           'b-', linewidth=2)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Zero line')
            axes[0, 1].fill_between(range(len(controller.sliding_history)), 
                                   -controller.params.lateral_boundary,
                                   controller.params.lateral_boundary,
                                   alpha=0.2, color='green', label='Boundary layer')
            axes[0, 1].set_xlabel('Time Step', fontsize=12)
            axes[0, 1].set_ylabel('Sliding Surface s', fontsize=12)
            axes[0, 1].set_title('Sliding Surface Evolution', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].hist(errors, bins=30, density=True, edgecolor='black', 
                       alpha=0.7, color='skyblue')
        if len(errors) > 1:
            mu, sigma = np.mean(errors), np.std(errors)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            y = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x - mu)/sigma)**2)
            axes[1, 0].plot(x, y, 'r-', linewidth=2, label=f'N(μ={mu:.3f}, σ={sigma:.3f})')
            axes[1, 0].legend()
        axes[1, 0].set_xlabel('Cross-track Error [m]', fontsize=12)
        axes[1, 0].set_ylabel('Probability Density', fontsize=12)
        axes[1, 0].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        if controller.sliding_history:
            V = 0.5 * np.array(controller.sliding_history)**2
            axes[1, 1].plot(range(len(V)), V, 'purple', linewidth=2, label='V = 0.5*s²')
            if len(V) > 1:
                V_dot = np.diff(V)
                axes2 = axes[1, 1].twinx()
                axes2.plot(range(len(V_dot)), V_dot, 'orange', linewidth=1, 
                          alpha=0.7, label='dV/dt')
                axes2.set_ylabel('dV/dt', fontsize=12)
                axes2.legend(loc='upper right')
            axes[1, 1].set_xlabel('Time Step', fontsize=12)
            axes[1, 1].set_ylabel('Lyapunov Function V', fontsize=12, color='purple')
            axes[1, 1].set_title('Lyapunov Stability Analysis', fontsize=14, fontweight='bold')
            axes[1, 1].legend(loc='upper left')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_trajectory(results, trajectory):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        distances = np.linspace(0, trajectory.total_path_length, 500)
        path_points = np.array([trajectory.get_path_point(d)[:2] for d in distances])
        ax.plot(path_points[:, 0], path_points[:, 1], '--', 
               linewidth=1.5, color='red', label='Reference Path', alpha=0.7)
        
        ax.plot(results['position']['x'], results['position']['y'], 
               linewidth=2, color='blue', label='Vehicle Path')
        
        ax.plot(results['position']['x'][0], results['position']['y'][0], 
               'go', markersize=10, label='Start')
        ax.plot(results['position']['x'][-1], results['position']['y'][-1], 
               'ro', markersize=10, label='End')
        
        ax.set_xlabel('X Position [m]', fontsize=12)
        ax.set_ylabel('Y Position [m]', fontsize=12)
        ax.set_title('Vehicle Trajectory Tracking', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        plt.show()
    
    @staticmethod
    def plot_tracking_performance(results):
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        time = results['time']
        
        axes[0, 0].plot(time, results['tracking']['cross_track_error'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time [s]', fontsize=12)
        axes[0, 0].set_ylabel('Cross-track Error [m]', fontsize=12)
        axes[0, 0].set_title('Lateral Tracking Performance', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time, np.degrees(results['tracking']['heading_error']), 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time [s]', fontsize=12)
        axes[0, 1].set_ylabel('Heading Error [deg]', fontsize=12)
        axes[0, 1].set_title('Heading Angle Tracking', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(time, results['tracking']['lateral_sliding'], 'g-', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Time [s]', fontsize=12)
        axes[1, 0].set_ylabel('Sliding Surface s', fontsize=12)
        axes[1, 0].set_title('Sliding Surface Dynamics', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time, results['velocities']['longitudinal'], 'b-', 
                       label='Actual Velocity', linewidth=2)
        axes[1, 1].plot(time, results['tracking']['target_velocity'], 'r--', 
                       label='Target Velocity', linewidth=2)
        axes[1, 1].set_xlabel('Time [s]', fontsize=12)
        axes[1, 1].set_ylabel('Velocity [m/s]', fontsize=12)
        axes[1, 1].set_title('Longitudinal Velocity Control', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 0].plot(time, np.degrees(results['steering']['front_left']), 'b-', 
                       label='Front Left', linewidth=2)
        axes[2, 0].plot(time, np.degrees(results['steering']['front_right']), 'r-', 
                       label='Front Right', linewidth=2)
        axes[2, 0].set_xlabel('Time [s]', fontsize=12)
        axes[2, 0].set_ylabel('Steering Angle [deg]', fontsize=12)
        axes[2, 0].set_title('Ackermann Steering Angles', fontsize=14, fontweight='bold')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(time, results['tracking']['curvature'], 'purple', linewidth=2)
        axes[2, 1].set_xlabel('Time [s]', fontsize=12)
        axes[2, 1].set_ylabel('Curvature [1/m]', fontsize=12)
        axes[2, 1].set_title('Path Curvature Profile', fontsize=14, fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def create_figure_eight_path(scale: float = 6.0, points: int = 40) -> np.ndarray:
    angles = np.linspace(0, 2*np.pi, points, endpoint=False)
    waypoints = []
    
    for angle in angles:
        x = scale * np.sin(angle)
        y = 0.5 * scale * np.sin(angle) * np.cos(angle)
        waypoints.append([x, y])
    
    waypoints.append(waypoints[0])
    return np.array(waypoints)

def main():
    print("=" * 60)
    print("ADVANCED SMC CONTROLLER FOR 4-WHEEL ACKERMANN VEHICLE")
    print("Nonlinear Control HW Implementation")
    print("=" * 60)
    
    vehicle_config = VehicleConfiguration()
    sim_settings = SimulationSettings()
    controller_params = SMCParameters()
    
    waypoints = create_figure_eight_path(scale=6.0, points=40)
    trajectory = TrajectoryManager(waypoints, closed_loop=True)
    
    controller = AdvancedSMCController(trajectory, vehicle_config, 
                                      sim_settings, controller_params)
    
    simulator = SimulationEngine(vehicle_config, sim_settings)
    
    print("Running simulation...")
    results = simulator.run_simulation(total_time=25.0, controller=controller)
    
    print("\n" + "=" * 60)
    print("GENERATING PLOTS AND ANALYSIS")
    print("=" * 60)
    
    analyzer = ResultsAnalyzer()
    
    print("\n1. Generating Phase Portraits (Error vs Error Rate)...")
    analyzer.plot_phase_portrait(controller)
    
    print("\n2. Generating Trajectory Plot...")
    analyzer.plot_trajectory(results, trajectory)
    
    print("\n3. Generating Performance Analysis Plots...")
    analyzer.plot_tracking_performance(results)
    
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS AND VALIDATION")
    print("=" * 60)
    
    cross_track_errors = results['tracking']['cross_track_error']
    rms_error = np.sqrt(np.mean(cross_track_errors**2))
    max_error = np.max(np.abs(cross_track_errors))
    mean_error = np.mean(np.abs(cross_track_errors))
    
    heading_errors = np.degrees(results['tracking']['heading_error'])
    rms_heading = np.sqrt(np.mean(heading_errors**2))
    
    print(f"\nTracking Performance:")
    print(f"  RMS Cross-track Error: {rms_error:.4f} m")
    print(f"  Maximum Cross-track Error: {max_error:.4f} m")
    print(f"  Mean Absolute Error: {mean_error:.4f} m")
    print(f"  RMS Heading Error: {rms_heading:.2f} deg")
    
    sliding_values = results['tracking']['lateral_sliding']
    in_boundary = np.mean(np.abs(sliding_values) <= controller_params.lateral_boundary) * 100
    
    if len(sliding_values) > 1:
        s_dot = np.diff(sliding_values) / sim_settings.time_step
        s_product = sliding_values[:-1] * s_dot
        eta = 0.2
        condition_satisfied = np.mean(s_product < -eta * np.abs(sliding_values[:-1])) * 100
    else:
        condition_satisfied = 0.0
    
    print(f"\nSliding Mode Controller Performance:")
    print(f"  Sliding Surface in Boundary Layer: {in_boundary:.1f}% of time")
    print(f"  Sliding Condition (s·ṡ < -η|s|) Satisfied: {condition_satisfied:.1f}% of time")
    
    print("\n" + "=" * 60)
    print("HW REQUIREMENTS CHECKLIST")
    print("=" * 60)
    print("✓ 1. Phase portraits of error dynamics: e vs ė")
    print("✓ 2. Robust Sliding Mode Controller implemented")
    print("✓ 3. 4-Wheel Ackermann vehicle model")
    print("✓ 4. Path following with SMC")
    print("✓ 5. Sliding condition analysis")
    print("✓ 6. Comprehensive performance metrics")
    print("✓ 7. Lyapunov stability analysis")

if __name__ == "__main__":
    main()