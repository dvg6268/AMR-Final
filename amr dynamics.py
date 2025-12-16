import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class DynamicAckermannVehicle:
    def __init__(self, wheelbase=2.5, track_width=1.8, mass=1200, Izz=2000):
        self.L = wheelbase
        self.W = track_width
        self.m = mass
        self.Izz = Izz
        self.C_alpha = 80000
        self.mu = 0.9
        self.max_torque = 800
        self.wheel_radius = 0.3
        
    def tire_forces(self, alpha, Fz):
        Fy = -self.C_alpha * alpha
        Fy_max = self.mu * Fz
        return np.clip(Fy, -Fy_max, Fy_max)
    
    def vehicle_dynamics(self, t, state, delta, throttle, brake):
        X, Y, psi, vx, vy, omega = state
        Fz_front = 0.6 * self.m * 9.81 / 2
        Fz_rear = 0.4 * self.m * 9.81 / 2
        if abs(vx) < 0.1:
            vx = 0.1
        alpha_fl = np.arctan2(vy + self.L/2 * omega, vx) - delta
        alpha_fr = np.arctan2(vy + self.L/2 * omega, vx) - delta
        alpha_rl = np.arctan2(vy - self.L/2 * omega, vx)
        alpha_rr = np.arctan2(vy - self.L/2 * omega, vx)
        Fy_fl = self.tire_forces(alpha_fl, Fz_front)
        Fy_fr = self.tire_forces(alpha_fr, Fz_front)
        Fy_rl = self.tire_forces(alpha_rl, Fz_rear)
        Fy_rr = self.tire_forces(alpha_rr, Fz_rear)
        Fx_total = (throttle * self.max_torque / self.wheel_radius - brake * self.m * 9.81 * 0.7) / 4
        Fx = 4 * Fx_total - 0.5 * 0.3 * 1.225 * vx**2
        Fy = Fy_fl + Fy_fr + Fy_rl + Fy_rr
        Mz = (Fy_fl + Fy_fr) * self.L/2 - (Fy_rl + Fy_rr) * self.L/2 + (Fy_fr - Fy_fl) * self.W/2 + (Fy_rr - Fy_rl) * self.W/2
        vx_dot = Fx/self.m + vy * omega
        vy_dot = Fy/self.m - vx * omega
        omega_dot = Mz / self.Izz
        X_dot = vx * np.cos(psi) - vy * np.sin(psi)
        Y_dot = vx * np.sin(psi) + vy * np.cos(psi)
        psi_dot = omega
        return [X_dot, Y_dot, psi_dot, vx_dot, vy_dot, omega_dot]
    
    def simulate_dynamic(self, delta_func, throttle_func, brake_func, initial_state=None, T=10, dt=0.01):
        if initial_state is None:
            initial_state = [0, 0, 0, 5.0, 0, 0]
        t_eval = np.arange(0, T, dt)
        def dynamics_wrapper(t, state):
            delta = delta_func(t)
            throttle = throttle_func(t)
            brake = brake_func(t)
            return self.vehicle_dynamics(t, state, delta, throttle, brake)
        solution = solve_ivp(dynamics_wrapper, [0, T], initial_state, t_eval=t_eval, method='RK45')
        return solution.t, solution.y

def test_straight_line():
    vehicle = DynamicAckermannVehicle()
    def delta_func(t):
        return 0.0
    def throttle_func(t):
        if t < 3:
            return 0.8
        elif t < 6:
            return 0.2
        else:
            return 0.0
    def brake_func(t):
        if t > 8:
            return 0.5
        return 0.0
    t, states = vehicle.simulate_dynamic(delta_func, throttle_func, brake_func, T=10)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    ax1.plot(states[0], states[1], 'b-', linewidth=2)
    ax1.set_xlabel('X Position [m]')
    ax1.set_ylabel('Y Position [m]')
    ax1.set_title('Straight Line: Trajectory')
    ax1.grid(True)
    ax1.axis('equal')
    ax2.plot(t, states[3], 'r-', label='vx (longitudinal)')
    ax2.plot(t, states[4], 'g-', label='vy (lateral)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_title('Velocity Components')
    ax2.legend()
    ax2.grid(True)
    ax3.plot(t, states[5], 'purple', linewidth=2)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Yaw Rate [rad/s]')
    ax3.set_title('Yaw Rate')
    ax3.grid(True)
    throttle = [throttle_func(ti) for ti in t]
    brake = [brake_func(ti) for ti in t]
    ax4.plot(t, throttle, 'g-', label='Throttle')
    ax4.plot(t, brake, 'r-', label='Brake')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Control Input')
    ax4.set_title('Control Inputs')
    ax4.legend()
    ax4.grid(True)
    plt.tight_layout()
    plt.show()

def test_lane_change():
    vehicle = DynamicAckermannVehicle()
    def delta_func(t):
        if 2 < t < 4:
            return 0.1 * np.sin(2 * np.pi * (t - 2) / 2)
        elif 6 < t < 8:
            return -0.1 * np.sin(2 * np.pi * (t - 6) / 2)
        return 0.0
    def throttle_func(t):
        return 0.3
    def brake_func(t):
        return 0.0
    t, states = vehicle.simulate_dynamic(delta_func, throttle_func, brake_func, T=12)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    ax1.plot(states[0], states[1], 'b-', linewidth=2)
    ax1.set_xlabel('X Position [m]')
    ax1.set_ylabel('Y Position [m]')
    ax1.set_title('Lane Change: Trajectory')
    ax1.grid(True)
    ax1.axis('equal')
    ax2.plot(t, states[3], 'r-', label='vx (longitudinal)')
    ax2.plot(t, states[4], 'g-', label='vy (lateral)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_title('Velocity Components During Lane Change')
    ax2.legend()
    ax2.grid(True)
    steering = [delta_func(ti) for ti in t]
    ax3.plot(t, np.degrees(steering), 'orange', label='Steering [deg]')
    ax3.plot(t, states[5], 'purple', label='Yaw Rate [rad/s]')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Steering / Yaw Rate')
    ax3.set_title('Steering Input and Yaw Response')
    ax3.legend()
    ax3.grid(True)
    beta = np.arctan2(states[4], states[3])
    ax4.plot(t, np.degrees(beta), 'b-', label='Vehicle Sideslip [deg]')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Sideslip Angle [deg]')
    ax4.set_title('Vehicle Sideslip Angle')
    ax4.legend()
    ax4.grid(True)
    plt.tight_layout()
    plt.show()

def test_circular_motion():
    vehicle = DynamicAckermannVehicle()
    def delta_func(t):
        return 0.15
    def throttle_func(t):
        return 0.4
    def brake_func(t):
        return 0.0
    t, states = vehicle.simulate_dynamic(delta_func, throttle_func, brake_func, T=15)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    ax1.plot(states[0], states[1], 'b-', linewidth=2)
    ax1.set_xlabel('X Position [m]')
    ax1.set_ylabel('Y Position [m]')
    ax1.set_title('Circular Motion: Trajectory')
    ax1.grid(True)
    ax1.axis('equal')
    ax2.plot(t, states[3], 'r-', label='Longitudinal Velocity')
    ax2.plot(t, states[4], 'g-', label='Lateral Velocity')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_title('Velocity Components')
    ax2.legend()
    ax2.grid(True)
    ax3.plot(t, states[5], 'purple', label='Yaw Rate')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Yaw Rate [rad/s]')
    ax3.set_title('Yaw Rate Response')
    ax3.legend()
    ax3.grid(True)
    lateral_accel = np.gradient(states[4], t) + states[3] * states[5]
    ax4.plot(t, lateral_accel, 'orange', label='Lateral Acceleration')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Acceleration [m/s²]')
    ax4.set_title('Lateral Acceleration')
    ax4.legend()
    ax4.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== DYNAMIC SIMULATION OF 4-WHEEL ACKERMANN VEHICLE ===")
    print("\nTest 1: Straight Line Acceleration and Braking")
    test_straight_line()
    print("\nTest 2: Lane Change Maneuver")
    test_lane_change()
    print("\nTest 3: Circular Motion Analysis")
    test_circular_motion()
