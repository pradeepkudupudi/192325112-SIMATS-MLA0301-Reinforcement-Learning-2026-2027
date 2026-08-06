"""
Experiment 2: Smart Home Robot Navigation
Objective: Develop a Q-learning RL agent for a smart home robot that learns optimal navigation to reach a charging station.
Method: Tabular Q-learning
"""

import numpy as np
import matplotlib.pyplot as plt

class SmartHomeEnv:
    def __init__(self, size=5):
        self.size = size
        self.state_space_size = size * size
        self.action_space_size = 4  # 0: Up, 1: Down, 2: Left, 3: Right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Grid layout: Start at (0,0), target charging station at (4,4), obstacles at (1,2), (2,2), (3,2)
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.obstacles = [(1, 2), (2, 2), (3, 2)]
        self.reset()
        
    def reset(self):
        self.agent_pos = self.start_pos
        return self.pos_to_state(self.agent_pos)
        
    def pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]
        
    def state_to_pos(self, state):
        return (state // self.size, state % self.size)
        
    def step(self, action_idx):
        move = self.actions[action_idx]
        next_r = self.agent_pos[0] + move[0]
        next_c = self.agent_pos[1] + move[1]
        
        # Boundary check
        if not (0 <= next_r < self.size and 0 <= next_c < self.size):
            # Hit wall, stay in same state, small penalty
            reward = -2
            done = False
        # Obstacle check
        elif (next_r, next_c) in self.obstacles:
            # Hit obstacle, stay in same state, larger penalty
            reward = -5
            done = False
        else:
            self.agent_pos = (next_r, next_c)
            reward = -1  # step penalty to encourage shortest path
            done = False
            
        if self.agent_pos == self.goal_pos:
            reward = 50
            done = True
            
        return self.pos_to_state(self.agent_pos), reward, done

def train_q_learning(env, episodes=500, lr=0.1, gamma=0.95, epsilon=0.1):
    q_table = np.zeros((env.state_space_size, env.action_space_size))
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space_size)
            else:
                action = np.argmax(q_table[state])
                
            next_state, reward, done = env.step(action)
            
            # Q-update
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state, best_next_action]
            q_table[state, action] += lr * (td_target - q_table[state, action])
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
    return q_table, rewards_history

if __name__ == "__main__":
    env = SmartHomeEnv()
    q_table, history = train_q_learning(env, episodes=300)
    
    # Plot reward convergence
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Smart Home Robot Q-Learning Convergence")
    plt.grid(True)
    plt.savefig("exp_02_smart_home_robot_nav.png")
    plt.close()
    
    print("Training finished. Optimal path plot saved to exp_02_smart_home_robot_nav.png")
    
    # Trace optimal path
    state = env.reset()
    path = [env.agent_pos]
    done = False
    steps = 0
    while not done and steps < 20:
        action = np.argmax(q_table[state])
        state, _, done = env.step(action)
        path.append(env.agent_pos)
        steps += 1
    print("Robot's Navigation Path to Charging Station:", path)
