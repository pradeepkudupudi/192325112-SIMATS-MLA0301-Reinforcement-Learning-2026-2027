"""
Experiment 7: Dynamic Programming for Autonomous Taxi Routing
Objective: Implement Dynamic Programming (Value Iteration and Policy Iteration) for an autonomous taxi routing grid.
States: (taxi_row, taxi_col, passenger_status) where status is 0: at source, 1: in taxi, 2: delivered.
Method: Value Iteration vs Policy Iteration comparison
"""

import numpy as np
import matplotlib.pyplot as plt

class TaxiGridEnv:
    def __init__(self, size=5):
        self.size = size
        self.src = (0, 0)      # Red
        self.dest = (4, 3)     # Blue
        
        # State space: (r, c, status)
        # status: 0=passenger at src, 1=passenger in taxi, 2=delivered (terminal)
        self.states = [(r, c, status) for r in range(size) for c in range(size) for status in [0, 1, 2]]
        self.n_states = len(self.states)
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right, 4: Pick, 5: Drop
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1), 'PICK', 'DROP']
        self.n_actions = len(self.actions)
        self.gamma = 0.9
        
    def step(self, state, action_idx):
        r, c, status = state
        
        # If already delivered (terminal state)
        if status == 2:
            return [(1.0, state, 0)]
            
        action = self.actions[action_idx]
        
        if isinstance(action, tuple):
            # Move taxi
            nr = max(0, min(self.size - 1, r + action[0]))
            nc = max(0, min(self.size - 1, c + action[1]))
            return [(1.0, (nr, nc, status), -1)] # movement cost
            
        elif action == 'PICK':
            # Pickup passenger
            if (r, c) == self.src and status == 0:
                return [(1.0, (r, c, 1), 10)] # reward for pickup
            else:
                return [(1.0, state, -10)] # penalty for illegal pickup
                
        elif action == 'DROP':
            # Drop passenger
            if (r, c) == self.dest and status == 1:
                return [(1.0, (r, c, 2), 100)] # large delivery reward
            else:
                return [(1.0, state, -10)] # penalty for illegal drop
                
def value_iteration(env, theta=1e-4):
    V = np.zeros(env.n_states)
    policy = np.zeros(env.n_states, dtype=int)
    iterations = 0
    
    while True:
        delta = 0
        new_V = np.copy(V)
        for s_idx, state in enumerate(env.states):
            if state[2] == 2:
                continue
                
            best_val = -float('inf')
            for a_idx in range(env.n_actions):
                transitions = env.step(state, a_idx)
                val = 0
                for prob, next_state, reward in transitions:
                    ns_idx = env.states.index(next_state)
                    val += prob * (reward + env.gamma * V[ns_idx])
                if val > best_val:
                    best_val = val
                    
            delta = max(delta, abs(new_V[s_idx] - best_val))
            new_V[s_idx] = best_val
            
        V = new_V
        iterations += 1
        if delta < theta:
            break
            
    # Extract Policy
    for s_idx, state in enumerate(env.states):
        best_val = -float('inf')
        best_act = 0
        for a_idx in range(env.n_actions):
            transitions = env.step(state, a_idx)
            val = sum(prob * (reward + env.gamma * V[env.states.index(next_state)]) for prob, next_state, reward in transitions)
            if val > best_val:
                best_val = val
                best_act = a_idx
        policy[s_idx] = best_act
        
    return V, policy, iterations

def policy_iteration(env, theta=1e-4):
    V = np.zeros(env.n_states)
    policy = np.zeros(env.n_states, dtype=int)
    pi_iterations = 0
    
    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for s_idx, state in enumerate(env.states):
                if state[2] == 2:
                    continue
                v_old = V[s_idx]
                action = policy[s_idx]
                transitions = env.step(state, action)
                v_new = sum(prob * (reward + env.gamma * V[env.states.index(next_state)]) for prob, next_state, reward in transitions)
                V[s_idx] = v_new
                delta = max(delta, abs(v_old - v_new))
            if delta < theta:
                break
                
        # Policy Improvement
        policy_stable = True
        for s_idx, state in enumerate(env.states):
            if state[2] == 2:
                continue
            old_action = policy[s_idx]
            
            best_val = -float('inf')
            best_act = 0
            for a_idx in range(env.n_actions):
                transitions = env.step(state, a_idx)
                val = sum(prob * (reward + env.gamma * V[env.states.index(next_state)]) for prob, next_state, reward in transitions)
                if val > best_val:
                    best_val = val
                    best_act = a_idx
            policy[s_idx] = best_act
            if old_action != best_act:
                policy_stable = False
                
        pi_iterations += 1
        if policy_stable:
            break
            
    return V, policy, pi_iterations

if __name__ == "__main__":
    env = TaxiGridEnv()
    
    V_vi, pol_vi, iter_vi = value_iteration(env)
    V_pi, pol_pi, iter_pi = policy_iteration(env)
    
    print(f"Value Iteration Converged in {iter_vi} iterations.")
    print(f"Policy Iteration Converged in {iter_pi} outer loops.")
    
    # Save a comparison bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(["Value Iteration", "Policy Iteration"], [iter_vi, iter_pi], color=['tomato', 'dodgerblue'])
    plt.ylabel("Iterations to Converge")
    plt.title("DP Algorithm Convergence Speed Comparison")
    plt.savefig("exp_07_taxi_routing_dp.png")
    plt.close()
    
    print("Optimal route visualization values saved to exp_07_taxi_routing_dp.png")
