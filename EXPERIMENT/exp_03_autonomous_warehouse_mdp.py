"""
Experiment 3: Autonomous Warehouse Robot MDP
Objective: Design and solve a stochastic MDP for an autonomous warehouse robot retrieving packages.
States: (row, col, carrying_package)
Actions: Up, Down, Left, Right, Pick, Drop
Method: Policy Iteration
"""

import numpy as np
import matplotlib.pyplot as plt

class WarehouseMDP:
    def __init__(self, width=4, height=4):
        self.w = width
        self.h = height
        self.pickup_loc = (0, 3)
        self.dropoff_loc = (3, 0)
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Pick, 5=Drop
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']
        
        # State space: (r, c, carrying) where carrying is 0 or 1
        self.states = [(r, c, car) for r in range(height) for c in range(width) for car in [0, 1]]
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        
        self.gamma = 0.95
        self.theta = 1e-4
        
        # Initialize Value function and Policy
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)  # index of action
        
    def step(self, state, action):
        """
        Returns a list of tuples: (probability, next_state, reward)
        Representing stochastic transitions.
        """
        r, c, carrying = state
        action_name = self.actions[action]
        
        # If terminal state: package is dropped at dropoff_loc while carrying
        if r == self.dropoff_loc[0] and c == self.dropoff_loc[1] and carrying == 1 and action_name == 'DROP':
            return [(1.0, state, 0)]  # Once successfully delivered, stay in terminal state
            
        transitions = []
        
        if action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            # Movement action with 80% success rate, 10% drift left, 10% drift right of direction
            moves = {
                'UP': ((-1, 0), (0, -1), (0, 1)),
                'DOWN': ((1, 0), (0, 1), (0, -1)),
                'LEFT': ((0, -1), (1, 0), (-1, 0)),
                'RIGHT': ((0, 1), (-1, 0), (1, 0))
            }
            primary, drift1, drift2 = moves[action_name]
            
            for move_dir, prob in [(primary, 0.8), (drift1, 0.1), (drift2, 0.1)]:
                nr, nc = r + move_dir[0], c + move_dir[1]
                if 0 <= nr < self.h and 0 <= nc < self.w:
                    next_s = (nr, nc, carrying)
                    transitions.append((prob, next_s, -1)) # step cost
                else:
                    # Bump into wall, stay in state
                    transitions.append((prob, state, -2)) # collision penalty
                    
        elif action_name == 'PICK':
            # Pick package from pickup_loc
            if (r, c) == self.pickup_loc and carrying == 0:
                transitions.append((1.0, (r, c, 1), 10))
            else:
                transitions.append((1.0, state, -5)) # invalid pick penalty
                
        elif action_name == 'DROP':
            # Drop package at dropoff_loc
            if (r, c) == self.dropoff_loc and carrying == 1:
                transitions.append((1.0, (r, c, 0), 100)) # big delivery reward
            else:
                transitions.append((1.0, state, -5)) # invalid drop penalty
                
        return transitions

    def policy_evaluation(self):
        while True:
            delta = 0
            for s_idx, state in enumerate(self.states):
                v_old = self.V[s_idx]
                action = self.policy[s_idx]
                v_new = 0
                transitions = self.step(state, action)
                for prob, next_state, reward in transitions:
                    ns_idx = self.states.index(next_state)
                    v_new += prob * (reward + self.gamma * self.V[ns_idx])
                self.V[s_idx] = v_new
                delta = max(delta, abs(v_old - v_new))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for s_idx, state in enumerate(self.states):
            old_action = self.policy[s_idx]
            
            best_action = None
            best_val = -float('inf')
            
            for a_idx in range(self.n_actions):
                val = 0
                transitions = self.step(state, a_idx)
                for prob, next_state, reward in transitions:
                    ns_idx = self.states.index(next_state)
                    val += prob * (reward + self.gamma * self.V[ns_idx])
                if val > best_val:
                    best_val = val
                    best_action = a_idx
                    
            self.policy[s_idx] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable

    def solve(self):
        steps = 0
        while True:
            self.policy_evaluation()
            stable = self.policy_improvement()
            steps += 1
            if stable or steps > 100:
                break
        print(f"Policy Iteration converged in {steps} iterations.")

if __name__ == "__main__":
    mdp = WarehouseMDP()
    mdp.solve()
    
    # Save a visualization of the Value functions
    V_empty = np.zeros((mdp.h, mdp.w))
    V_carrying = np.zeros((mdp.h, mdp.w))
    
    for idx, (r, c, car) in enumerate(mdp.states):
        if car == 0:
            V_empty[r, c] = mdp.V[idx]
        else:
            V_carrying[r, c] = mdp.V[idx]
            
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im1 = axes[0].imshow(V_empty, cmap='viridis')
    axes[0].set_title("V(s) - Empty Hands (Go to Pick)")
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(V_carrying, cmap='plasma')
    axes[1].set_title("V(s) - Carrying Package (Go to Drop)")
    fig.colorbar(im2, ax=axes[1])
    
    plt.savefig("exp_03_autonomous_warehouse_mdp.png")
    plt.close()
    
    print("Warehouse MDP Solved. Value map plot saved as exp_03_autonomous_warehouse_mdp.png")
