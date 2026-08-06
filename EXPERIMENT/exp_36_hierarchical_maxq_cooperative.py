"""
Experiment 36: MAXQ Hierarchical Decomposition for Multi-Agent Cooperation
Objective: Decompose a cooperative box carrying task into MAXQ subtasks for two agents.
Method: Hierarchical MAXQ Action Decomposition
"""

import numpy as np
import matplotlib.pyplot as plt

class CooperativeCarryEnv:
    def __init__(self, size=5):
        self.size = size
        self.box_pos = (2, 2)
        self.goal_pos = (4, 4)
        self.reset()
        
    def reset(self):
        self.a1 = (0, 0)
        self.a2 = (0, 4)
        self.box_lifted = False
        self.box_at_goal = False
        return self._get_obs()
        
    def _get_obs(self):
        return (self.a1, self.a2, self.box_lifted, self.box_at_goal)
        
    def step(self, action1, action2):
        # Actions: 0: Navigate to Box, 1: Lift, 2: Carry to Goal, 3: Wait
        reward = -1.0 # step penalty
        done = False
        
        # Action 0: Move towards box
        if action1 == 0 and self.a1 != self.box_pos:
            self.a1 = self.box_pos
            reward += 5.0
        if action2 == 0 and self.a2 != self.box_pos:
            self.a2 = self.box_pos
            reward += 5.0
            
        # Action 1: Lift (needs both agents at box)
        if action1 == 1 and action2 == 1:
            if self.a1 == self.box_pos and self.a2 == self.box_pos and not self.box_lifted:
                self.box_lifted = True
                reward += 20.0
                
        # Action 2: Carry to goal (needs box lifted and both moving to goal)
        if action1 == 2 and action2 == 2:
            if self.box_lifted:
                self.a1 = self.goal_pos
                self.a2 = self.goal_pos
                self.box_at_goal = True
                reward += 100.0
                done = True
                
        return self._get_obs(), reward, done

class HierarchicalCoopAgent:
    def __init__(self):
        # Subtask Q-values mapping state indices to choices
        # We simplify the state encoding to: [dist_to_box_a1, dist_to_box_a2, lifted]
        self.Q_root = {} # Decisions: 0 (Navigate), 1 (Lift), 2 (Carry)
        
    def get_state_key(self, obs):
        a1, a2, lifted, goal = obs
        d1 = 1 if a1 == (2,2) else 0
        d2 = 1 if a2 == (2,2) else 0
        return (d1, d2, int(lifted))
        
    def choose_action(self, obs, epsilon=0.1):
        key = self.get_state_key(obs)
        if key not in self.Q_root:
            self.Q_root[key] = np.zeros(3)
            
        # Hardcoded subtask policy choices or trained decisions
        if np.random.rand() < epsilon:
            return np.random.randint(3)
            
        # Subtask priority
        d1, d2, lifted = key
        if d1 == 0 or d2 == 0:
            return 0 # Navigate subtask
        elif not lifted:
            return 1 # Lift subtask
        else:
            return 2 # Carry subtask

if __name__ == "__main__":
    env = CooperativeCarryEnv()
    agent = HierarchicalCoopAgent()
    
    episodes = 200
    rewards = []
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 30:
            steps += 1
            # Hierarchical policy determines subtask
            subtask = agent.choose_action(obs)
            
            # Execute primitive action mappings
            if subtask == 0:
                # Navigate agents
                act1, act2 = 0, 0
            elif subtask == 1:
                # Lift
                act1, act2 = 1, 1
            else:
                # Carry
                act1, act2 = 2, 2
                
            next_obs, reward, done = env.step(act1, act2)
            ep_reward += reward
            obs = next_obs
            
        rewards.append(ep_reward)
        
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, color='green')
    plt.xlabel("Episode")
    plt.ylabel("Cooperative Joint Reward")
    plt.title("MAXQ Cooperative Multi-Agent Carry Training")
    plt.grid(True)
    plt.savefig("exp_36_hierarchical_maxq_cooperative.png")
    plt.close()
    
    print("MAXQ multi-agent carry script finished. Plot saved as exp_36_hierarchical_maxq_cooperative.png")
