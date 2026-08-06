"""
Experiment 38: POMDP Robot Navigation under Partial Observability
Objective: Enable a robot to localize itself in a 5-cell corridor with noisy wall sensors and navigate to the goal.
Method: POMDP Belief State Tracking and Value-based Action Selection
"""

import numpy as np
import matplotlib.pyplot as plt

class CorridorPOMDP:
    def __init__(self):
        # 5 States: cell 0, 1, 2, 3, 4 (Goal)
        self.n_states = 5
        # 2 Actions: 0: Move Left, 1: Move Right
        self.n_actions = 2
        # 2 Observations: 0: Corridor, 1: Wall (located at cells 0 and 4)
        self.n_obs = 2
        
        # Transition probabilities: T(s' | s, a)
        # 90% success rate, 10% stays in place
        self.T = np.zeros((self.n_actions, self.n_states, self.n_states))
        for s in range(self.n_states):
            # Move Left
            self.T[0, s, max(0, s-1)] = 0.9
            self.T[0, s, s] = 0.1
            # Move Right
            self.T[1, s, min(self.n_states-1, s+1)] = 0.9
            self.T[1, s, s] = 0.1
            
        # Observation probabilities: O(o | s, a)
        # Wall at cells 0 and 4. Corridor at 1, 2, 3.
        # Sensor has 15% error rate
        self.O = np.zeros((self.n_actions, self.n_states, self.n_obs))
        for a in range(self.n_actions):
            self.O[a, 0] = [0.15, 0.85]  # Wall
            self.O[a, 1] = [0.85, 0.15]  # Corridor
            self.O[a, 2] = [0.85, 0.15]  # Corridor
            self.O[a, 3] = [0.85, 0.15]  # Corridor
            self.O[a, 4] = [0.15, 0.85]  # Wall
            
        # Rewards: R(s, a)
        self.R = np.zeros((self.n_states, self.n_actions))
        self.R[:, 0] = -1.0  # step cost
        self.R[:, 1] = -1.0
        self.R[3, 1] = 100.0 # moving right from cell 3 hits goal cell 4!

    def update_belief(self, belief, action, observation):
        new_belief = np.zeros(self.n_states)
        for s_prime in range(self.n_states):
            prior = sum(self.T[action, s, s_prime] * belief[s] for s in range(self.n_states))
            new_belief[s_prime] = self.O[action, s_prime, observation] * prior
            
        norm = np.sum(new_belief)
        if norm > 0:
            new_belief /= norm
        else:
            new_belief = belief.copy()
        return new_belief

    def get_action_value(self, belief):
        # Value of each action = expected value
        values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            values[a] = sum(belief[s] * self.R[s, a] for s in range(self.n_states))
        return values

if __name__ == "__main__":
    pomdp = CorridorPOMDP()
    
    # Starting belief: Uniform uncertainty across cells 0 to 3 (goal is 4)
    belief = np.array([0.25, 0.25, 0.25, 0.25, 0.0])
    
    # True state starts stochastically at cell 1
    true_state = 1
    
    belief_history = []
    true_states = [true_state]
    
    steps = 0
    done = False
    
    print("Starting Robot Navigation under Corridor POMDP...")
    
    while not done and steps < 15:
        steps += 1
        
        # Decide action based on expected values of belief
        # If expected reward of moving right is higher, move right. Otherwise move left to localize first.
        val_left = sum(belief[s] * (pomdp.R[s, 0] + 0.9 * (s-1)) for s in range(5)) # estimated heuristic value
        val_right = sum(belief[s] * (pomdp.R[s, 1] + 0.9 * (s+1)) for s in range(5))
        
        action = 1 if val_right >= val_left else 0
        
        # Environmental transition
        probs_trans = pomdp.T[action, true_state]
        true_state = np.random.choice(pomdp.n_states, p=probs_trans)
        true_states.append(true_state)
        
        # Generate observation
        probs_obs = pomdp.O[action, true_state]
        obs = np.random.choice(pomdp.n_obs, p=probs_obs)
        
        # Update Belief
        belief = pomdp.update_belief(belief, action, obs)
        belief_history.append(belief.copy())
        
        obs_name = "Wall Detected" if obs == 1 else "Corridor Detected"
        act_name = "Move Right" if action == 1 else "Move Left"
        print(f"Step {steps}: Action={act_name}, Obs={obs_name} | True State={true_state} | Max belief at Cell {np.argmax(belief)}")
        
        if true_state == 4:
            print("Target reached successfully!")
            done = True
            
    belief_history = np.array(belief_history)
    
    # Save a visualization of belief states over time
    plt.figure(figsize=(10, 4))
    plt.imshow(belief_history.T, aspect='auto', cmap='Blues', origin='lower')
    plt.colorbar(label="Belief Probability")
    plt.xlabel("Interaction Step")
    plt.ylabel("Cell Index (State)")
    plt.title("Robot Belief Localization & Path Tracking")
    plt.yticks(range(5))
    
    # Draw path of true states
    plt.plot(range(len(true_states)-1), true_states[1:], color='red', marker='x', label='True Robot Position')
    plt.legend()
    plt.savefig("exp_38_pomdp_robot_partial_observability.png")
    plt.close()
    
    print("Simulation analysis saved as exp_38_pomdp_robot_partial_observability.png")
