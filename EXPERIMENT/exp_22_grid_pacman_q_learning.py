"""
Experiment 22: Q-Learning for Grid-Based Pac-Man Game
Objective: Develop a Q-learning agent to collect food rewards while avoiding a moving ghost on a 4x4 grid.
States: (pacman_row, pacman_col, ghost_row, ghost_col)
Actions: 0: Up, 1: Down, 2: Left, 3: Right
Method: Tabular Q-Learning
"""

import numpy as np
import matplotlib.pyplot as plt

class PacmanGridEnv:
    def __init__(self, size=4):
        self.size = size
        self.food_pos = (3, 3)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        self.n_actions = len(self.actions)
        self.n_states = (size * size) * (size * size)
        self.reset()
        
    def reset(self):
        self.pacman = (0, 0)
        self.ghost = (2, 2)
        return self._get_state()
        
    def _get_state(self):
        p_idx = self.pacman[0] * self.size + self.pacman[1]
        g_idx = self.ghost[0] * self.size + self.ghost[1]
        return p_idx * 16 + g_idx
        
    def step(self, action):
        move = self.actions[action]
        
        # Move Pacman
        nr = max(0, min(self.size - 1, self.pacman[0] + move[0]))
        nc = max(0, min(self.size - 1, self.pacman[1] + move[1]))
        self.pacman = (nr, nc)
        
        # Check instant collision
        if self.pacman == self.ghost:
            return self._get_state(), -100, True
            
        # Move Ghost randomly
        g_move = self.actions[np.random.randint(self.n_actions)]
        gnr = max(0, min(self.size - 1, self.ghost[0] + g_move[0]))
        gnc = max(0, min(self.size - 1, self.ghost[1] + g_move[1]))
        self.ghost = (gnr, gnc)
        
        # Check collision post ghost-move
        if self.pacman == self.ghost:
            return self._get_state(), -100, True
            
        # Food reward
        if self.pacman == self.food_pos:
            return self._get_state(), 100, True
            
        return self._get_state(), -1, False

def train_pacman(env, episodes=1000, lr=0.1, gamma=0.9, epsilon=0.15):
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        steps = 0
        
        while not done and steps < 100:
            steps += 1
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])
                
            next_state, reward, done = env.step(action)
            
            # Q-Update
            best_next = np.argmax(Q[next_state])
            Q[state, action] += lr * (reward + gamma * Q[next_state, best_next] - Q[state, action])
            
            state = next_state
            total_r += reward
            
        rewards_history.append(total_r)
        
    return Q, rewards_history

if __name__ == "__main__":
    env = PacmanGridEnv()
    Q, history = train_pacman(env, episodes=800)
    
    # Smooth history
    window = 20
    smooth_history = np.convolve(history, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(8, 4))
    plt.plot(smooth_history, color='gold')
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Return")
    plt.title("Q-Learning Pac-Man Agent Training curve")
    plt.grid(True)
    plt.savefig("exp_22_grid_pacman_q_learning.png")
    plt.close()
    
    print("Pac-man Q-learning completed. Saved plot to exp_22_grid_pacman_q_learning.png")
