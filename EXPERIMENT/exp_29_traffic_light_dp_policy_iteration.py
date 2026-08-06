"""
Experiment 29: Dynamic Programming for Traffic Light Timing Optimization
Objective: Minimize vehicle wait times at an intersection using Policy Iteration DP.
States: (queue_NS, queue_EW) where queues are 0..4
Actions: 0: Green North-South (Red EW), 1: Green East-West (Red NS)
Method: Policy Iteration Dynamic Programming
"""

import numpy as np
import matplotlib.pyplot as plt

class TrafficLightDP:
    def __init__(self, max_queue=4):
        self.max_queue = max_queue
        # States: (q_ns, q_ew)
        self.states = [(q_ns, q_ew) for q_ns in range(max_queue + 1) for q_ew in range(max_queue + 1)]
        self.n_states = len(self.states)
        self.n_actions = 2  # 0: Green NS, 1: Green EW
        
        self.gamma = 0.9
        self.theta = 1e-4
        
        # Initialize Value and Policy
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)
        
    def step(self, state, action):
        """
        Stochastic transition. Returns a list of (probability, next_state, reward) tuples.
        """
        q_ns, q_ew = state
        
        # Arrival probabilities:
        # P(1 arrival NS) = 0.4, P(0 arrival NS) = 0.6
        # P(1 arrival EW) = 0.3, P(0 arrival EW) = 0.7
        arrivals = [(0, 0, 0.6 * 0.7), (1, 0, 0.4 * 0.7), (0, 1, 0.6 * 0.3), (1, 1, 0.4 * 0.3)]
        
        # Drain rate:
        # Green light drains 2 vehicles, Red light drains 0
        drain_ns = 2 if action == 0 else 0
        drain_ew = 2 if action == 1 else 0
        
        transitions = []
        for arr_ns, arr_ew, prob in arrivals:
            # New queue = old_queue - drain + arrival
            nq_ns = max(0, min(self.max_queue, q_ns - drain_ns + arr_ns))
            nq_ew = max(0, min(self.max_queue, q_ew - drain_ew + arr_ew))
            
            next_state = (nq_ns, nq_ew)
            # Reward: negative sum of queues (waiting time)
            reward = -float(nq_ns + nq_ew)
            transitions.append((prob, next_state, reward))
            
        return transitions

    def policy_evaluation(self):
        while True:
            delta = 0
            for s_idx, state in enumerate(self.states):
                v_old = self.V[s_idx]
                action = self.policy[s_idx]
                v_new = 0
                for prob, next_state, reward in self.step(state, action):
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
                for prob, next_state, reward in self.step(state, a_idx):
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
        print(f"Policy Iteration converged in {steps} loops.")

if __name__ == "__main__":
    dp = TrafficLightDP()
    dp.solve()
    
    # Reshape policy to 5x5 grid
    policy_grid = np.zeros((dp.max_queue + 1, dp.max_queue + 1))
    for idx, (q_ns, q_ew) in enumerate(dp.states):
        policy_grid[q_ns, q_ew] = dp.policy[idx]
        
    plt.figure(figsize=(7, 6))
    plt.imshow(policy_grid, cmap='bwr', origin='lower')
    plt.colorbar(ticks=[0, 1], label="Optimal Light: Green NS (Blue/0) vs Green EW (Red/1)")
    plt.xlabel("East-West Queue Length")
    plt.ylabel("North-South Queue Length")
    plt.title("Traffic Light DP Optimal Policy Map")
    plt.xticks(range(5))
    plt.yticks(range(5))
    plt.grid(True)
    plt.savefig("exp_29_traffic_light_dp_policy_iteration.png")
    plt.close()
    
    print("Optimal policy derived. Map grid saved to exp_29_traffic_light_dp_policy_iteration.png")
