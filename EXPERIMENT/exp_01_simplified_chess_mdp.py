"""
Experiment 1: Simplified Chess MDP
Objective: Design and implement an MDP for a simplified chess game where an intelligent agent (Knight)
learns the optimal sequence of moves to capture the enemy King while avoiding threat zones (guarded by pawns).
Method: Value Iteration (Dynamic Programming)
"""

import numpy as np
import matplotlib.pyplot as plt

class ChessBoardMDP:
    def __init__(self, size=8):
        self.size = size
        self.states = [(r, c) for r in range(size) for c in range(size)]
        
        # Define Knight moves (L-shapes)
        self.actions = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        # Setup positions
        self.king_pos = (7, 7)
        self.pawn_threats = [(3, 3), (4, 4), (5, 5)]
        
        self.gamma = 0.9  # Discount factor
        self.theta = 1e-4 # Convergence threshold
        
        # Initialize Value function
        self.V = np.zeros((size, size))
        
    def step(self, state, action):
        """
        Transition function: P(s' | s, a)
        Returns next state and reward.
        """
        next_r = state[0] + action[0]
        next_c = state[1] + action[1]
        
        # Out of bounds
        if not (0 <= next_r < self.size and 0 <= next_c < self.size):
            return state, -10  # Heavy penalty for illegal move
            
        next_state = (next_r, next_c)
        
        # Rewards
        if next_state == self.king_pos:
            return next_state, 100
        elif next_state in self.pawn_threats:
            return next_state, -20
        else:
            return next_state, -1  # Step penalty
            
    def value_iteration(self):
        iterations = 0
        while True:
            delta = 0
            new_V = np.copy(self.V)
            for r in range(self.size):
                for c in range(self.size):
                    state = (r, c)
                    if state == self.king_pos:
                        continue
                    
                    action_values = []
                    for action in self.actions:
                        next_state, reward = self.step(state, action)
                        val = reward + self.gamma * self.V[next_state[0], next_state[1]]
                        action_values.append(val)
                    
                    new_val = max(action_values)
                    delta = max(delta, abs(new_val - self.V[r, c]))
                    new_V[r, c] = new_val
            
            self.V = new_V
            iterations += 1
            if delta < self.theta or iterations > 1000:
                break
        print(f"Value Iteration converged in {iterations} iterations.")
        
    def get_optimal_policy(self):
        policy = {}
        for r in range(self.size):
            for c in range(self.size):
                state = (r, c)
                if state == self.king_pos:
                    policy[state] = None
                    continue
                
                best_action = None
                best_val = -float('inf')
                for idx, action in enumerate(self.actions):
                    next_state, reward = self.step(state, action)
                    val = reward + self.gamma * self.V[next_state[0], next_state[1]]
                    if val > best_val:
                        best_val = val
                        best_action = action
                policy[state] = best_action
        return policy

    def plot_optimal_value_map(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.V, cmap="hot", interpolation="nearest")
        plt.colorbar(label="State Value")
        plt.title("Chess MDP - Knight Value Map")
        
        # Label coordinates
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) == self.king_pos:
                    plt.text(c, r, 'King', color='blue', ha='center', va='center', fontweight='bold')
                elif (r, c) in self.pawn_threats:
                    plt.text(c, r, 'Threat', color='cyan', ha='center', va='center', fontweight='bold')
                else:
                    plt.text(c, r, f"{self.V[r,c]:.1f}", color='green' if self.V[r,c] > 0 else 'white',
                             ha='center', va='center', fontsize=8)
                             
        plt.savefig("exp_01_simplified_chess_mdp.png")
        plt.close()

if __name__ == "__main__":
    env = ChessBoardMDP()
    env.value_iteration()
    policy = env.get_optimal_policy()
    env.plot_optimal_value_map()
    print("Optimal policy generated. Visualization saved to exp_01_simplified_chess_mdp.png")
    
    # Showcase path from start position (0, 0)
    current = (0, 0)
    path = [current]
    steps = 0
    while current != env.king_pos and steps < 20:
        action = policy[current]
        if action is None:
            break
        current, _ = env.step(current, action)
        path.append(current)
        steps += 1
    print("Knight's Optimal Path from (0,0) to King:", path)
