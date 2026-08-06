"""
Experiment 27: Autonomous Car Navigation at Intersections
Objective: Train a policy to navigate an intersection road network while obeying traffic light rules.
State: [distance_to_intersection, traffic_light_color (0: Red, 1: Green), velocity]
Actions: 0: Accelerate, 1: Coast/Brake, 2: Stop/Wait
Method: Q-Learning with Traffic Rule Constraints
"""

import numpy as np
import matplotlib.pyplot as plt

class IntersectionNavigationEnv:
    def __init__(self):
        self.n_states = 10 * 2 * 3 # Distances (0-9) * traffic light states (Red, Green) * velocities (0, 1, 2) = 60 states
        self.n_actions = 3  # 0: Accelerate, 1: Coast, 2: Stop
        self.reset()
        
    def reset(self):
        self.distance = 9  # starts 9 units away from intersection
        self.light = 0     # 0: Red, 1: Green
        self.speed = 1     # 0: Stopped, 1: Medium, 2: High
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return (self.distance * 6) + (self.light * 3) + self.speed
        
    def step(self, action):
        self.steps += 1
        
        # Traffic light state changes periodically (stochastic duration)
        if np.random.rand() < 0.25:
            self.light = 1 - self.light
            
        reward = 0
        done = False
        
        # Apply action to speed
        if action == 0:     # Accelerate
            self.speed = min(2, self.speed + 1)
            reward = -1 # engine noise cost
        elif action == 1:   # Coast/Brake
            self.speed = max(0, self.speed - 1)
            reward = -0.5
        elif action == 2:   # Stop/Wait
            self.speed = 0
            reward = -0.5
            
        # Move vehicle
        self.distance -= self.speed
        
        # Check rule violations
        if self.distance <= 0:
            # Reached intersection
            if self.distance < 0: # overshoot
                self.distance = 0
                
            if self.light == 0:  # RED
                if self.speed > 0:
                    reward = -150.0  # Run a red light collision penalty
                    done = True
                else:
                    reward = 10.0   # Properly waiting at red light
            else:  # GREEN
                if self.speed > 0:
                    reward = 100.0   # Safely crossed intersection
                    done = True
                else:
                    reward = -5.0   # Blocked intersection on green
                    
        # Max steps timeout
        if self.steps >= 40:
            done = True
            
        return self._get_obs(), reward, done

def train_intersection_car(env, episodes=1000, lr=0.1, gamma=0.95, epsilon=0.15):
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])
                
            next_state, reward, done = env.step(action)
            
            # Q-update
            best_next = np.argmax(Q[next_state])
            Q[state, action] += lr * (reward + gamma * Q[next_state, best_next] - Q[state, action])
            
            state = next_state
            total_r += reward
            
        rewards_history.append(total_r)
        
    return Q, rewards_history

if __name__ == "__main__":
    env = IntersectionNavigationEnv()
    Q, history = train_intersection_car(env, episodes=800)
    
    # Smooth history
    window = 30
    smooth_history = np.convolve(history, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(8, 4))
    plt.plot(smooth_history, color='crimson')
    plt.xlabel("Episode")
    plt.ylabel("Reward (Rule Compliance)")
    plt.title("Car Intersection Navigation RL Training")
    plt.grid(True)
    plt.savefig("exp_27_car_intersection_navigation.png")
    plt.close()
    
    print("Intersection navigation car training complete. Saved plot to exp_27_car_intersection_navigation.png")
