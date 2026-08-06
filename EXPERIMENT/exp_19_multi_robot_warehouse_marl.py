"""
Experiment 19: Multi-Agent Reinforcement Learning (MARL) for Multi-Robot Warehouse
Objective: Train two independent Q-learning agents to navigate a grid warehouse and reach target destinations without colliding.
Method: Independent Q-learning (MARL)
"""

import numpy as np
import matplotlib.pyplot as plt

class MultiAgentWarehouseEnv:
    def __init__(self, size=4):
        self.size = size
        self.r1_start = (0, 0)
        self.r2_start = (0, 3)
        self.t1 = (3, 0)
        self.t2 = (3, 3)
        
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        self.n_actions = len(self.actions)
        self.reset()
        
    def reset(self):
        self.r1 = self.r1_start
        self.r2 = self.r2_start
        return self._get_obs()
        
    def _get_obs(self):
        # Return local obs for each agent as flattened state index
        obs1 = self.r1[0] * self.size + self.r1[1]
        obs2 = self.r2[0] * self.size + self.r2[1]
        return obs1, obs2
        
    def step(self, action1, action2):
        move1 = self.actions[action1]
        move2 = self.actions[action2]
        
        nr1 = (max(0, min(self.size - 1, self.r1[0] + move1[0])),
               max(0, min(self.size - 1, self.r1[1] + move1[1])))
               
        nr2 = (max(0, min(self.size - 1, self.r2[0] + move2[0])),
               max(0, min(self.size - 1, self.r2[1] + move2[1])))
               
        reward1, reward2 = -1, -1 # Default movement cost
        done = False
        
        # Collision check
        if nr1 == nr2:
            # Robots collide, stay in current positions, heavy penalty
            reward1 -= 10
            reward2 -= 10
        else:
            self.r1 = nr1
            self.r2 = nr2
            
        # Target reaching
        if self.r1 == self.t1:
            reward1 += 20
        if self.r2 == self.t2:
            reward2 += 20
            
        if self.r1 == self.t1 and self.r2 == self.t2:
            done = True
            
        return self._get_obs(), reward1, reward2, done

def train_marl(env, episodes=500, lr=0.1, gamma=0.9, epsilon=0.15):
    # Initialize separate Q-tables for each agent
    n_states = env.size * env.size
    Q1 = np.zeros((n_states, env.n_actions))
    Q2 = np.zeros((n_states, env.n_actions))
    
    rewards_history = []
    
    for ep in range(episodes):
        obs1, obs2 = env.reset()
        done = False
        total_r = 0
        steps = 0
        
        while not done and steps < 60:
            steps += 1
            # E-greedy for Agent 1
            if np.random.rand() < epsilon:
                act1 = np.random.randint(env.n_actions)
            else:
                act1 = np.argmax(Q1[obs1])
                
            # E-greedy for Agent 2
            if np.random.rand() < epsilon:
                act2 = np.random.randint(env.n_actions)
            else:
                act2 = np.argmax(Q2[obs2])
                
            (n_obs1, n_obs2), r1, r2, done = env.step(act1, act2)
            
            # Independent Q-updates
            best_n1 = np.argmax(Q1[n_obs1])
            Q1[obs1, act1] += lr * (r1 + gamma * Q1[n_obs1, best_n1] - Q1[obs1, act1])
            
            best_n2 = np.argmax(Q2[n_obs2])
            Q2[obs2, act2] += lr * (r2 + gamma * Q2[n_obs2, best_n2] - Q2[obs2, act2])
            
            obs1, obs2 = n_obs1, n_obs2
            total_r += (r1 + r2)
            
        rewards_history.append(total_r)
        
    return Q1, Q2, rewards_history

if __name__ == "__main__":
    env = MultiAgentWarehouseEnv()
    Q1, Q2, history = train_marl(env, episodes=400)
    
    plt.figure(figsize=(8, 4))
    plt.plot(history, color='purple')
    plt.xlabel("Episode")
    plt.ylabel("Joint Reward")
    plt.title("MARL Multi-Robot Warehouse Cooperative Training Curve")
    plt.grid(True)
    plt.savefig("exp_19_multi_robot_warehouse_marl.png")
    plt.close()
    
    print("MARL warehousing training finished. Saved performance comparison to exp_19_multi_robot_warehouse_marl.png")
