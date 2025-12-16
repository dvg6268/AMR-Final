# AMR-Final Project: Autonomous Vehicle Control & Navigation

##  Project Overview
This project implements a comprehensive autonomous vehicle control and navigation system with 4-wheel Ackermann steering, featuring robust control algorithms, safety guarantees, and motion planning techniques.

##  System Modeling
- **Comprehensive vehicle simulation** with 4-wheel Ackermann steering kinematics and dynamics
- **Realistic physics implementation** including mass, inertia, tire forces, and slip dynamics  
- **Ackermann geometry principles** where front wheels turn at different angles for smooth rotation
- **Bridging theory to practice** through virtual testbeds that mimic real robotic behavior

##  Control Methodology  
- **Robust Sliding Mode Controller (SMC)** for precise trajectory tracking despite uncertainties
- **Mathematically proven stability** using Lyapunov analysis and sliding surface convergence  
- **Control Barrier Functions (CBF)** providing absolute safety guarantees for collision avoidance
- **Multi-scenario validation** including straight lines, lane changes, and complex figure-8 paths

##  Motion Planning
- **Potential field navigation** creating virtual attractive (goal) and repulsive (obstacle) forces
- **Hybrid obstacle avoidance** combining reactive responses with proactive path planning
- **Demonstration of limitations** including local minima problems and their practical implications
- **Safety-first approach** maintaining protective buffers while optimizing for goal efficiency

## Setup
**pip install - r req.txt**
All required Python packages for simulations, controls, and visualizations are included in req.txt
