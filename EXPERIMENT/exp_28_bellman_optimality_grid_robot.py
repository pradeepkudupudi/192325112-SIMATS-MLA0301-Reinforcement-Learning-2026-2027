"""
Experiment 28: Bellman Optimality Equation for Grid Robot
Objective: Compute the optimal state-value function V*(s) for a robot's grid navigation task using Bellman's Optimality Equation.
Method: Value Iteration (Dynamic Programming)
"""

import numpy as np
import matplotlib.pyplot as plt

class BellmanGridRobot:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = [(1, 2), (2, 2), (3, 2)]  # Obstacle wall in the middle
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        
        # Initialize state-values
        self.V = np.zeros((size, size))
        self.gamma = 0.95
        self.theta = 1e-5
        
    def step(self, r, c, action):
        if (r, c) == self.goal:
            return r, c, 0  # Goal state is terminal
            
        move = self.actions[action]
        nr, nc = r + move[0], c + move[1]
        
        # Hit wall or obstacle
        if not (0 <= nr < self.size and 0 <= nc < self.size) or (nr, nc) in self.obstacles:
            return r, c, -2  # Stay in place with penalty
            
        return nr, nc, -1  # Step cost

    def solve_bellman_optimality(self):
        iterations = 0
        while True:
            delta = 0
            new_V = np.copy(self.V)
            
            for r in range(self.size):
                for c in range(self.size):
                    if (r, c) == self.goal:
                        new_V[r, c] = 100.0  # Goal state terminal reward
                        continue
                    if (r, c) in self.obstacles:
                        new_V[r, c] = -50.0  # Obstacle value
                        continue
                        
                    action_values = []
                    for a in range(len(self.actions)):
                        nr, nc, reward = self.step(r, c, a)
                        val = reward + self.gamma * self.V[nr, nc]
                        action_values.append(val)
                        
                    new_val = max(action_values)
                    delta = max(delta, abs(new_val - self.V[r, c]))
                    new_V[r, c] = new_val
                    
            self.V = new_V
            iterations += 1
            if delta < self.theta:
                break
        print(f"Bellman optimality equation solved in {iterations} iterations.")

    def get_optimal_path(self):
        path = [self.start]
        curr = self.start
        steps = 0
        
        while curr != self.goal and steps < 30:
            steps += 1
            best_val = -float('inf')
            best_next = curr
            
            for a in range(len(self.actions)):
                nr, nc, _ = self.step(curr[0], curr[1], a)
                val = self.V[nr, nc]
                if val > best_val:
                    best_val = val
                    best_next = (nr, nc)
                    
            curr = best_next
            path.append(curr)
        return path

if __name__ == "__main__":
    robot = BellmanGridRobot()
    robot.solve_bellman_optimality()
    
    # Trace path
    path = robot.get_optimal_path()
    print("Optimal Path computed:", " -> ".join([str(p) for p in path]))
    
    # Save a visualization of state values and the optimal path
    plt.figure(figsize=(7, 6))
    plt.imshow(robot.V, cmap='YlGnBu')
    plt.colorbar(label="State Value V*(s)")
    plt.title("Bellman Optimality State-Value Function & Optimal Path")
    
    # Overlay path coordinates
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    plt.plot(path_x, path_y, color='red', linewidth=3, marker='o', label='Robot Path')
    
    # Mark objects
    plt.text(robot.start[1], robot.start[0], 'START', ha='center', va='center', color='magenta', fontweight='bold')
    plt.text(robot.goal[1], robot.goal[0], 'GOAL', ha='center', va='center', color='green', fontweight='bold')
    for obs in robot.obstacles:
        plt.text(obs[1], obs[0], 'WALL', ha='center', va='center', color='white', fontweight='bold')
        
    plt.legend()
    plt.savefig("exp_28_bellman_optimality_grid_robot.png")
    plt.close()
    
    print("State value visualization saved to exp_28_bellman_optimality_grid_robot.png")
