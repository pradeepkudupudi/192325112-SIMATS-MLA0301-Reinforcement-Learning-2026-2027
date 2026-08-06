"""
Experiment 39: Reinforcement Learning for Healthcare Management
Objective: Train an RL policy to schedule admissions and allocate staff overtime to maximize patient recovery and minimize stress.
State: [waiting_patients_count, available_beds, staff_stress (0: Calm, 1: Tired, 2: Burned Out)]
Actions: 0: Normal Admissions, 1: Divert Patients (Admit none), 2: Call Overtime (Boost treatment rate)
Method: Q-Learning with Healthcare Constraint Safety
"""

import numpy as np
import matplotlib.pyplot as plt

class HealthcareEnv:
    def __init__(self):
        # State: waiting patients (0..5), beds available (0..3), staff stress (0..2)
        # Total state space = 6 * 4 * 3 = 72 states
        self.n_states = 72
        self.n_actions = 3
        self.reset()
        
    def reset(self):
        self.waiting = 2
        self.beds = 3
        self.stress = 0
        return self._get_obs()
        
    def _get_obs(self):
        return (self.waiting * 12) + (self.beds * 3) + self.stress
        
    def step(self, action):
        # Patient arrivals: Poisson arrivals
        arrivals = np.random.poisson(lam=1.5)
        self.waiting = min(5, self.waiting + arrivals)
        
        reward = 0
        done = False
        
        # Action implementation
        treatment_rate = 1
        if action == 0:     # Normal
            # Maintain normal treatment
            self.stress = max(0, self.stress - 1)
        elif action == 1:   # Divert incoming patients
            self.waiting = max(0, self.waiting - 2)
            reward -= 10.0  # Diverting penalty / transfer cost
        elif action == 2:   # Overtime
            treatment_rate = 3
            self.stress = min(2, self.stress + 1)
            reward -= 5.0   # Overtime wage cost
            
        # Treat patients
        treated = min(self.waiting, self.beds, treatment_rate)
        self.waiting -= treated
        self.beds -= treated # beds occupied
        
        # Simulate patient recovery / discharge
        discharges = np.random.randint(0, (3 - self.beds) + 1)
        self.beds = min(3, self.beds + discharges)
        
        # Rewards based on performance
        reward += treated * 15.0  # Recovery reward
        reward -= self.waiting * 2.0  # Waiting list penalty
        
        # Burnout penalty
        if self.stress == 2:
            reward -= 10.0
            
        # Overflow penalty (irresponsible care capacity)
        if self.waiting >= 5:
            reward -= 30.0
            
        # Transition state stochastically
        return self._get_obs(), reward, done

def train_healthcare(env, episodes=1000, lr=0.1, gamma=0.9, epsilon=0.15):
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        steps = 0
        
        while not done and steps < 60:
            steps += 1
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
    env = HealthcareEnv()
    Q, history = train_healthcare(env, episodes=800)
    
    # Smooth history
    window = 30
    smooth_history = np.convolve(history, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(8, 4))
    plt.plot(smooth_history, color='teal')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Hospital Score")
    plt.title("Healthcare Allocation Q-Learning Model Progress")
    plt.grid(True)
    plt.savefig("exp_39_healthcare_management_rl.png")
    plt.close()
    
    print("Healthcare model training finished. Output saved as exp_39_healthcare_management_rl.png")
