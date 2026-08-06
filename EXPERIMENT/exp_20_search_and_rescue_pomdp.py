"""
Experiment 20: Partially Observable MDP (POMDP) for Search-and-Rescue
Objective: Model a search-and-rescue robot locating a victim under sensor noise.
Hidden States: 0: Victim in Left Room, 1: Victim in Right Room
Actions: 0: Listen (low cost, noisy signal), 1: Search Left (terminal), 2: Search Right (terminal)
Observations: 0: Beep Left, 1: Beep Right, 2: Silence
Method: Belief State Updating and Planning
"""

import numpy as np
import matplotlib.pyplot as plt

class SearchRescuePOMDP:
    def __init__(self):
        # States, Actions, Observations
        self.n_states = 2
        self.n_actions = 3
        self.n_obs = 3
        
        # Transition probabilities: Target is stationary
        # T(s' | s, a) = Identity matrix
        self.T = np.eye(self.n_states)
        
        # Observation probabilities: O(o | s, a)
        # If action is 'Listen' (0), sensor is 85% accurate:
        # P(Beep L | State L) = 0.85, P(Beep R | State L) = 0.10, P(Silence | State L) = 0.05
        # If actions are Search L (1) or Search R (2), no new sensory data (Silence with 1.0 probability)
        self.O = np.zeros((self.n_actions, self.n_states, self.n_obs))
        
        # Listen action observations
        self.O[0, 0] = [0.85, 0.10, 0.05]  # Victim on Left
        self.O[0, 1] = [0.10, 0.85, 0.05]  # Victim on Right
        
        # Search actions
        self.O[1, 0] = [0.0, 0.0, 1.0]
        self.O[1, 1] = [0.0, 0.0, 1.0]
        self.O[2, 0] = [0.0, 0.0, 1.0]
        self.O[2, 1] = [0.0, 0.0, 1.0]
        
        # Rewards: R(s, a)
        self.R = np.zeros((self.n_states, self.n_actions))
        self.R[:, 0] = -1.0    # Listen costs -1
        self.R[0, 1] = 50.0    # Search Left when victim is Left
        self.R[1, 1] = -30.0   # Search Left when victim is Right
        self.R[0, 2] = -30.0   # Search Right when victim is Left
        self.R[1, 2] = 50.0    # Search Right when victim is Right
        
    def update_belief(self, belief, action, observation):
        """Bayesian Belief Update: b'(s') = O(o | s', a) * sum(T(s' | s, a) * b(s)) / P(o | b, a)"""
        new_belief = np.zeros(self.n_states)
        
        for s_prime in range(self.n_states):
            prior = sum(self.T[s, s_prime] * belief[s] for s in range(self.n_states))
            new_belief[s_prime] = self.O[action, s_prime, observation] * prior
            
        prob_obs = np.sum(new_belief)
        if prob_obs > 0:
            new_belief /= prob_obs
        else:
            new_belief = belief.copy() # fallback
        return new_belief, prob_obs

    def get_action_value(self, belief, Q_values=None):
        """Compute expected reward of each action based on current belief state"""
        values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            # Immediate expected reward
            values[a] = sum(belief[s] * self.R[s, a] for s in range(self.n_states))
        return values

if __name__ == "__main__":
    pomdp = SearchRescuePOMDP()
    
    # Track belief over steps starting at uniform prior
    belief = np.array([0.5, 0.5])
    
    # Simulate true state: Victim is in Right Room (1)
    true_state = 1
    
    belief_history = [belief[1]]  # probability of being in Right Room
    action_history = []
    
    steps = 0
    done = False
    
    print("Starting Search and Rescue POMDP Agent Simulation...")
    print(f"Initial Belief (Victim is in Right Room): {belief[1]:.2f}")
    
    while not done and steps < 10:
        steps += 1
        q_vals = pomdp.get_action_value(belief)
        
        # Decide action: if search has positive expected reward, search. Otherwise listen.
        if max(q_vals[1], q_vals[2]) > q_vals[0]:
            action = 1 if q_vals[1] > q_vals[2] else 2
            done = True
        else:
            action = 0 # Listen
            
        action_history.append(action)
        
        # Generate observation
        probs_obs = pomdp.O[action, true_state]
        obs = np.random.choice(pomdp.n_obs, p=probs_obs)
        
        # Update belief
        belief, _ = pomdp.update_belief(belief, action, obs)
        belief_history.append(belief[1])
        
        obs_names = ["Beep Left", "Beep Right", "Silence"]
        act_names = ["LISTEN", "SEARCH LEFT", "SEARCH RIGHT"]
        print(f"Step {steps}: Action={act_names[action]}, Obs={obs_names[obs]} -> New Belief (P_Right)={belief[1]:.4f}")
        
    # Plot belief update progression
    plt.figure(figsize=(8, 4))
    plt.plot(belief_history, color='orange', marker='o', label="P(Victim in Right Room)")
    plt.axhline(y=1.0, color='r', linestyle='--', label="True Position (Right)")
    plt.xlabel("Step")
    plt.ylabel("Belief Probability")
    plt.title("POMDP Belief State Evolution during Search and Rescue")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_20_search_and_rescue_pomdp.png")
    plt.close()
    
    print(f"Simulation ended. Active path analysis saved to exp_20_search_and_rescue_pomdp.png")
