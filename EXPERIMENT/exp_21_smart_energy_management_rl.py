"""
Experiment 21: RL-Based Safe and Responsible Energy Management
Objective: Develop a Safe RL energy management policy to balance load demands, solar usage, and battery.
State: [grid_load_level, battery_charge, pricing_tier]
Actions: 0: Draw Grid, 1: Draw Battery, 2: Sell Solar, 3: Load Shed (Responsible/Fair constraint)
Method: Q-Learning with Safety Constraints (Safe RL)
"""

import numpy as np
import matplotlib.pyplot as plt

class SmartEnergyEnv:
    def __init__(self):
        self.battery_capacity = 4
        self.n_states = 3 * 5 * 3 # Grid load (3 levels) * battery (5 levels) * pricing tier (3 levels) = 45 states
        self.n_actions = 4  # 0: GRID, 1: BATTERY, 2: SELL_SOLAR, 3: LOAD_SHED
        self.reset()
        
    def reset(self):
        self.grid_load = np.random.randint(0, 3) # 0: Low, 1: Medium, 2: High
        self.battery = 2  # start half full
        self.pricing = np.random.randint(0, 3)   # 0: Low, 1: Normal, 2: Peak
        return self._get_state()
        
    def _get_state(self):
        return (self.grid_load * 15) + (self.battery * 3) + self.pricing
        
    def step(self, action):
        reward = 0
        safety_violation = 0
        done = False
        
        # 0: Draw from Grid
        if action == 0:
            if self.pricing == 2:  # peak hours
                reward = -8        # high cost
            else:
                reward = -2
            # Safe constraint: Drawing from grid when load is critical is irresponsible
            if self.grid_load == 2:
                reward -= 15  # Unsafe grid overload penalty
                safety_violation += 1
                
        # 1: Draw from Battery
        elif action == 1:
            if self.battery > 0:
                self.battery -= 1
                reward = 4  # positive reward for self-sufficiency
            else:
                reward = -12  # battery empty penalty
                
        # 2: Sell Solar
        elif action == 2:
            # We assume solar is available. If battery is full, we get maximum feed-in profit
            if self.battery >= self.battery_capacity:
                reward = 8
            else:
                self.battery = min(self.battery_capacity, self.battery + 1)
                reward = 3
                
        # 3: Load Shed
        elif action == 3:
            # Fair/Responsible constraint: Shedding load causes household discomfort
            # Should only be done as a last resort (e.g. battery is empty AND grid load is critical)
            if self.battery == 0 and self.grid_load == 2:
                reward = -1  # necessary shedding, low penalty
            else:
                reward = -20  # irresponsible/unfair shedding penalty
                safety_violation += 1
                
        # Update state stochastically
        self.grid_load = np.random.randint(0, 3)
        self.pricing = np.random.randint(0, 3)
        
        return self._get_obs_indices(), reward, safety_violation, done

    def _get_obs_indices(self):
        return (self.grid_load * 15) + (self.battery * 3) + self.pricing

def train_safe_energy_rl(env, episodes=800, lr=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.n_states, env.n_actions))
    rewards = []
    violations = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_violations = 0
        steps = 0
        
        while not done and steps < 50:
            steps += 1
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])
                
            next_state, reward, violation, done = env.step(action)
            
            # Q update
            best_next = np.argmax(Q[next_state])
            Q[state, action] += lr * (reward + gamma * Q[next_state, best_next] - Q[state, action])
            
            state = next_state
            ep_reward += reward
            ep_violations += violation
            
        rewards.append(ep_reward)
        violations.append(ep_violations)
        
    return Q, rewards, violations

if __name__ == "__main__":
    env = SmartEnergyEnv()
    Q, rewards, violations = train_safe_energy_rl(env)
    
    # Save training curves (Rolling average)
    window = 20
    rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    violations_smooth = np.convolve(violations, np.ones(window)/window, mode='valid')
    
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    color = 'tab:green'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(rewards_smooth, color=color, label='Smoothed Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Safety Violations', color=color)
    ax2.plot(violations_smooth, color=color, linestyle='--', label='Violations')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Responsible Smart Energy Management RL Training Progress")
    fig.tight_layout()
    plt.savefig("exp_21_smart_energy_management_rl.png")
    plt.close()
    
    print("Safe Energy Management training complete. Plot saved to exp_21_smart_energy_management_rl.png")
    print(f"Final average safety violations per episode: {np.mean(violations[-50:]):.3f}")
