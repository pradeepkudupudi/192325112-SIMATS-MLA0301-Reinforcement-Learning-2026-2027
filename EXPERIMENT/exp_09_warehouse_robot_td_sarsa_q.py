"""
Experiment 9: Temporal Difference Learning (TD(0), SARSA, Q-Learning)
Objective: Implement and compare TD(0), SARSA, and Q-Learning for a warehouse robot navigating obstacles.
Environment: Gridworld with walls/obstacles.
Method: Tabular TD evaluation and TD control
"""

import numpy as np
import matplotlib.pyplot as plt

class WarehouseGrid:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = [(1, 1), (1, 2), (2, 3), (3, 1), (3, 3)]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        self.n_actions = len(self.actions)
        self.n_states = size * size
        self.reset()
        
    def reset(self):
        self.pos = self.start
        return self._get_state()
        
    def _get_state(self):
        return self.pos[0] * self.size + self.pos[1]
        
    def step(self, action_idx):
        move = self.actions[action_idx]
        nr = self.pos[0] + move[0]
        nc = self.pos[1] + move[1]
        
        # Check boundary
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return self._get_state(), -2, False
            
        # Check obstacle
        if (nr, nc) in self.obstacles:
            return self._get_state(), -5, False
            
        self.pos = (nr, nc)
        if self.pos == self.goal:
            return self._get_state(), 100, True
        return self._get_state(), -1, False

def run_td0_evaluation(env, policy, episodes=200, lr=0.1, gamma=0.9):
    """TD(0) State Value Evaluation"""
    V = np.zeros(env.n_states)
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy[state]
            next_state, reward, done = env.step(action)
            V[state] += lr * (reward + gamma * V[next_state] - V[state])
            state = next_state
    return V

def run_sarsa(env, episodes=500, lr=0.1, gamma=0.9, epsilon=0.15):
    """SARSA Control"""
    Q = np.zeros((env.n_states, env.n_actions))
    reward_history = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        
        # Choose action A from state S using policy derived from Q (e-greedy)
        if np.random.rand() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(Q[state])
            
        while not done:
            next_state, reward, done = env.step(action)
            
            # Choose next action A' from S' using policy derived from Q (e-greedy)
            if np.random.rand() < epsilon:
                next_action = np.random.randint(env.n_actions)
            else:
                next_action = np.argmax(Q[next_state])
                
            # SARSA Update: Q(S,A) <- Q(S,A) + lr * (R + gamma * Q(S',A') - Q(S,A))
            Q[state, action] += lr * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            state = next_state
            action = next_action
            total_r += reward
            
        reward_history.append(total_r)
        
    return Q, reward_history

def run_q_learning(env, episodes=500, lr=0.1, gamma=0.9, epsilon=0.15):
    """Q-Learning Control"""
    Q = np.zeros((env.n_states, env.n_actions))
    reward_history = []
    
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
            
            # Q-learning Update: Q(S,A) <- Q(S,A) + lr * (R + gamma * max_a Q(S',a) - Q(S,A))
            best_next_a = np.argmax(Q[next_state])
            Q[state, action] += lr * (reward + gamma * Q[next_state, best_next_a] - Q[state, action])
            
            state = next_state
            total_r += reward
            
        reward_history.append(total_r)
        
    return Q, reward_history

if __name__ == "__main__":
    env = WarehouseGrid()
    
    # Define a simple policy for TD(0) evaluation (always go RIGHT or DOWN)
    policy = np.zeros(env.n_states, dtype=int)
    for r in range(env.size):
        for c in range(env.size):
            state = r * env.size + c
            if c < env.size - 1:
                policy[state] = 3  # Right
            else:
                policy[state] = 1  # Down
                
    V_td = run_td0_evaluation(env, policy)
    _, sarsa_rewards = run_sarsa(env, episodes=400)
    _, q_rewards = run_q_learning(env, episodes=400)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(sarsa_rewards, label="SARSA (On-Policy)", color="purple")
    plt.plot(q_rewards, label="Q-Learning (Off-Policy)", color="cyan")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Warehouse Navigation: SARSA vs Q-Learning")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_09_warehouse_robot_td_sarsa_q.png")
    plt.close()
    
    print("TD / SARSA / Q-learning completed. Plot saved as exp_09_warehouse_robot_td_sarsa_q.png")
    print(f"Final 10-episode mean SARSA reward: {np.mean(sarsa_rewards[-10:]):.2f}")
    print(f"Final 10-episode mean Q-learning reward: {np.mean(q_rewards[-10:]):.2f}")
