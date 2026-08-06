"""
Experiment 17: Hierarchical Reinforcement Learning (MAXQ Decomposition)
Objective: Implement Hierarchical Reinforcement Learning using MAXQ for a household robot executing multiple subtasks.
Method: MAXQ-Q Learning (Hierarchical Decomposition)
"""

import numpy as np
import matplotlib.pyplot as plt

class HouseholdRobotEnv:
    def __init__(self):
        # Locations: 0: Charging Dock, 1: Living Room, 2: Laundry Room
        self.loc = 0
        self.battery = 100
        self.room_dirty = True
        self.clothes_dirty = True
        self.reset()
        
    def reset(self):
        self.loc = 0
        self.battery = 100
        self.room_dirty = True
        self.clothes_dirty = True
        return self._get_state()
        
    def _get_state(self):
        return (self.loc, self.battery, int(self.room_dirty), int(self.clothes_dirty))
        
    def step(self, primitive_action):
        # Actions: 0: Navigate to Dock, 1: Navigate to Living Room, 2: Navigate to Laundry, 3: Clean, 4: Wash, 5: Charge
        cost = -1
        done = False
        
        # Battery drain for actions
        self.battery = max(0, self.battery - 10)
        if self.battery <= 0 and primitive_action != 5:
            return self._get_state(), -100, True
            
        if primitive_action in [0, 1, 2]:
            self.loc = primitive_action
            cost = -2
        elif primitive_action == 3: # Clean
            if self.loc == 1 and self.room_dirty:
                self.room_dirty = False
                cost = 50
            else:
                cost = -10
        elif primitive_action == 4: # Wash
            if self.loc == 2 and self.clothes_dirty:
                self.clothes_dirty = False
                cost = 50
            else:
                cost = -10
        elif primitive_action == 5: # Charge
            if self.loc == 0:
                self.battery = 100
                cost = 10
            else:
                cost = -20
                
        # Main Task Goal: Room clean, clothes washed, returned to charging dock
        if not self.room_dirty and not self.clothes_dirty and self.loc == 0 and self.battery > 50:
            cost += 100
            done = True
            
        return self._get_state(), cost, done

class MAXQAgent:
    """Simplified hierarchical MAXQ agent with Q-tables for subtasks"""
    def __init__(self, env):
        self.env = env
        # Subtasks: Root, Navigate, Tidy, Laundry, Charge
        # Q-tables mapping states to subtasks / primitive actions
        self.Q_root = {}       # Actions: navigate(0,1,2), tidy(3), laundry(4), charge(5)
        self.Q_tidy = {}       # Actions: navigate_to_living(1), clean(3)
        self.Q_laundry = {}    # Actions: navigate_to_laundry(2), wash(4)
        
    def get_q_value(self, Q, state, action):
        if state not in Q:
            Q[state] = np.zeros(6) # size of action space
        return Q[state][action]
        
    def update_q(self, Q, state, action, value):
        if state not in Q:
            Q[state] = np.zeros(6)
        Q[state][action] = value
        
    def choose_action(self, Q, state, allowed_actions, epsilon=0.15):
        if np.random.rand() < epsilon:
            return np.random.choice(allowed_actions)
        q_vals = [self.get_q_value(Q, state, a) for a in allowed_actions]
        return allowed_actions[np.argmax(q_vals)]

def train_maxq(env, episodes=400, lr=0.1, gamma=0.95):
    agent = MAXQAgent(env)
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        
        while not done:
            # Hierarchical decision loop
            # Determine if charging is needed first
            if state[1] <= 30: # low battery, choose charge subtask
                subtask_action = agent.choose_action(agent.Q_root, state, [0, 5])
            elif state[2] == 1: # room is dirty, tidy subtask
                subtask_action = agent.choose_action(agent.Q_tidy, state, [1, 3])
            elif state[3] == 1: # clothes dirty, laundry subtask
                subtask_action = agent.choose_action(agent.Q_laundry, state, [2, 4])
            else: # Go home and idle
                subtask_action = agent.choose_action(agent.Q_root, state, [0])
                
            next_state, reward, done = env.step(subtask_action)
            total_r += reward
            
            # Simple hierarchical updates
            if subtask_action in [1, 3]:
                best_next_a = np.argmax([agent.get_q_value(agent.Q_tidy, next_state, a) for a in [1, 3]])
                q_target = reward + gamma * agent.get_q_value(agent.Q_tidy, next_state, best_next_a)
                old_q = agent.get_q_value(agent.Q_tidy, state, subtask_action)
                agent.update_q(agent.Q_tidy, state, subtask_action, old_q + lr * (q_target - old_q))
            elif subtask_action in [2, 4]:
                best_next_a = np.argmax([agent.get_q_value(agent.Q_laundry, next_state, a) for a in [2, 4]])
                q_target = reward + gamma * agent.get_q_value(agent.Q_laundry, next_state, best_next_a)
                old_q = agent.get_q_value(agent.Q_laundry, state, subtask_action)
                agent.update_q(agent.Q_laundry, state, subtask_action, old_q + lr * (q_target - old_q))
            else:
                best_next_a = np.argmax([agent.get_q_value(agent.Q_root, next_state, a) for a in [0, 5]])
                q_target = reward + gamma * agent.get_q_value(agent.Q_root, next_state, best_next_a)
                old_q = agent.get_q_value(agent.Q_root, state, subtask_action)
                agent.update_q(agent.Q_root, state, subtask_action, old_q + lr * (q_target - old_q))
                
            state = next_state
            
        rewards_history.append(total_r)
        
    return rewards_history

if __name__ == "__main__":
    env = HouseholdRobotEnv()
    history = train_maxq(env, episodes=500)
    
    plt.figure(figsize=(8, 4))
    plt.plot(history, color='brown')
    plt.xlabel("Episode")
    plt.ylabel("Hierarchical Return")
    plt.title("HRL MAXQ Household Robot Training Curve")
    plt.grid(True)
    plt.savefig("exp_17_household_robot_hrl_ham_maxq.png")
    plt.close()
    
    print("MAXQ Hierarchical agent training completed. Saved plot to exp_17_household_robot_hrl_ham_maxq.png")
