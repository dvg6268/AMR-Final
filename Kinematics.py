import numpy as np
import matplotlib.pyplot as plt

class FourWheelAckermannCar:
    def __init__(self, wheelbase=2.5, track_width=1.8, max_steering=0.5):
        self.L = wheelbase          # Wheelbase
        self.W = track_width        # Track width
        self.max_steer = max_steering  # Max steering angle (rad)
    
    def steering_geometry(self, delta):
        """Ackermann steering angles for inner and outer wheels"""
        if abs(delta) < 1e-6:  # Straight line
            return delta, delta
        
        R = self.L / np.tan(delta)  # Turning radius
        
        # Inner and outer wheel angles
        delta_i = np.arctan(self.L / (R - self.W/2))
        delta_o = np.arctan(self.L / (R + self.W/2))
        
        return delta_i, delta_o
    
    def step(self, state, velocity, steering_angle, dt):
        """Update state using Ackermann kinematics"""
        x, y, theta = state
        
        # Limit steering angle
        delta = np.clip(steering_angle, -self.max_steer, self.max_steer)
        
        # Kinematic equations
        dx = velocity * np.cos(theta)
        dy = velocity * np.sin(theta)
        dtheta = (velocity / self.L) * np.tan(delta)
        
        # Update state
        x += dx * dt
        y += dy * dt
        theta += dtheta * dt
        
        return [x, y, theta]
    
    def simulate(self, velocity_profile, steering_profile, T=10, dt=0.01):
        """Simulate with time-varying inputs"""
        state = [0.0, 0.0, 0.0]
        X, Y, Theta = [], [], []
        times = np.arange(0, T, dt)
        
        for i, t in enumerate(times):
            # Get inputs at current time
            if callable(velocity_profile):
                v = velocity_profile(t)
            else:
                v = velocity_profile
                
            if callable(steering_profile):
                delta = steering_profile(t)
            else:
                delta = steering_profile
            
            # Update state
            state = self.step(state, v, delta, dt)
            X.append(state[0])
            Y.append(state[1])
            Theta.append(state[2])
        
        return np.array(X), np.array(Y), np.array(Theta)

if __name__ == "__main__":
    car = FourWheelAckermannCar()
    
    # Example 1: Straight line
    X1, Y1, _ = car.simulate(
        velocity_profile=3.0,
        steering_profile=0.0,
        T=10
    )
    
    # Example 2: Constant radius turn
    X2, Y2, _ = car.simulate(
        velocity_profile=3.0,
        steering_profile=0.2,  # 11.5 degrees
        T=10
    )
    
    # Example 3: Lane change maneuver
    def steering_for_lane_change(t):
        if t < 2:
            return 0.0
        elif t < 4:
            return 0.15  # Turn right
        elif t < 6:
            return -0.15  # Turn left
        else:
            return 0.0
    
    X3, Y3, _ = car.simulate(
        velocity_profile=5.0,
        steering_profile=steering_for_lane_change,
        T=10
    )
    
    # Plot results
    plt.figure(figsize=(10, 8))
    plt.plot(X1, Y1, 'b-', label="Straight line (δ=0°)", linewidth=2)
    plt.plot(X2, Y2, 'r-', label="Constant radius turn (δ=11.5°)", linewidth=2)
    plt.plot(X3, Y3, 'g-', label="Lane change maneuver", linewidth=2)
    
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("4-Wheel Ackermann Steering Kinematics")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()