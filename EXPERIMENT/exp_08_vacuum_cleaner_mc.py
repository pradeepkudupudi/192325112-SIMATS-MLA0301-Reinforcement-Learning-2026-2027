"""
Experiment 8: Monte Carlo Prediction and Control for Robot Vacuum Cleaner
Objective: Implement First-Visit Monte Carlo Control for a robot vacuum cleaner optimizing cleaning and battery life.
States: (row, col, battery) where battery is 0..5
Actions: 0: Up, 1: Down, 2: Left, 3: Right, 4: Clean, 5: Charge
Method: Epsilon-Greedy On-Policy Monte Carlo Control
"""

import numpy as np
import matplotlib.pyplot as plt

class VacuumEnv:
    def __init__(self, size=3):
        self.size = size
        self.battery_max = 5
        self.charging_station = (0, 0)
        self.dirty_squares = [(0, 2), (2, 0), (2, 2)]
        
        # State space: (r, c, battery)
        # We also track which squares are currently dirty. To simplify the state space,
        # we focus on the robot's physical location and its battery level: 3 * 3 * 6 = 54 states.
        self.states = [(r, c, b) for r in range(size) for c in range(size) for b in range(self.battery_max + 1)]
        self.n_states = len(self.states)
        self.n_actions = 6 # Up, Down, Left, Right, Clean, Charge
        self.reset()
        
    def reset(self):
        self.pos = (0, 0)
        self.battery = self.battery_max
        self.cleaned = {loc: False for loc in self.dirty_squares}
        return self._get_state()
        
    def _get_state(self):
        return (self.pos[0], self.pos[1], self.battery)
        
    def step(self, action):
        r, c = self.pos
        reward = 0
        done = False
        
        # Dead robot
        if self.battery <= 0:
            return self._get_state(), -100, True
            
        if action < 4:
            # Movement: Up, Down, Left, Right
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            move = moves[action]
            nr = max(0, min(self.size - 1, r + move[0]))
            nc = max(0, min(self.size - 1, c + move[1]))
            self.pos = (nr, nc)
            self.battery -= 1
            reward = -1  # small movement cost
            
        elif action == 4:
            # Clean
            if self.pos in self.dirty_squares and not self.cleaned[self.pos]:
                self.cleaned[self.pos] = True
                reward = 15
            else:
                reward = -5  # cleaning a clean spot penalty
            self.battery -= 1
            
        elif action == 5:
            # Charge
            if self.pos == self.charging_station:
                self.battery = self.battery_max
                reward = 5
            else:
                reward = -10  # charging away from station is illegal
                self.battery -= 1
                
        # Win condition: Cleaned everything and returned to base
        if all(self.cleaned.values()) and self.pos == self.charging_station:
            reward += 50
            done = True
            
        # Battery exhaustion
        if self.battery <= 0 and not done:
            reward = -50
            done = True
            
        return self._get_state(), reward, done

def mc_control_epsilon_greedy(env, episodes=2000, epsilon=0.15, gamma=0.9):
    # Initialize Q-values and returns list
    Q = {}
    returns = {}
    policy = {}
    
    for s in env.states:
        Q[s] = np.zeros(env.n_actions)
        returns[s] = [[] for _ in range(env.n_actions)]
        policy[s] = np.zeros(env.n_actions) + 1.0 / env.n_actions
        
    ep_rewards = []
    
    for ep in range(episodes):
        # Generate an episode
        state = env.reset()
        episode = []
        done = False
        
        while not done:
            # Select action based on policy
            probs = policy[state]
            action = np.random.choice(env.n_actions, p=probs)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            
        total_ep_reward = sum(x[2] for x in episode)
        ep_rewards.append(total_ep_reward)
        
        # Calculate Returns and update Q
        G = 0
        visited_sa = set()
        for idx in reversed(range(len(episode))):
            s, a, r = episode[idx]
            G = r + gamma * G
            if (s, a) not in visited_sa:
                visited_sa.add((s, a))
                returns[s][a].append(G)
                Q[s][a] = np.mean(returns[s][a])
                
                # Epsilon-greedy update
                best_a = np.argmax(Q[s])
                for act in range(env.n_actions):
                    if act == best_a:
                        policy[s][act] = 1 - epsilon + (epsilon / env.n_actions)
                    else:
                        policy[s][act] = epsilon / env.n_actions
                        
    return Q, ep_rewards

if __name__ == "__main__":
    env = VacuumEnv()
    Q, rewards = mc_control_epsilon_greedy(env, episodes=1000)
    
    # Plot rewards
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, color='green')
    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward")
    plt.title("Robot Vacuum MC Control Progress")
    plt.grid(True)
    plt.savefig("exp_08_vacuum_cleaner_mc.png")
    plt.close()
    
    print("Monte Carlo control finished. Performance graph saved to exp_08_vacuum_cleaner_mc.png")
