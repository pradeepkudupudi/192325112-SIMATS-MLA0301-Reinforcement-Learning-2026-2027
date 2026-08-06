"""
Experiment 37: Tabular Dyna-Q and Dyna-Q+ for a Shortcut Gridworld
Objective: Study planning vs learning trade-offs and dynamic environment exploration using Dyna-Q and Dyna-Q+.
Method: Dyna-Q and Dyna-Q+ with simulated environment model updates
"""

import numpy as np
import matplotlib.pyplot as plt

class ShortcutMaze:
    def __init__(self):
        self.height = 6
        self.width = 9
        self.start = (5, 3)
        self.goal = (0, 8)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        self.n_actions = 4
        self.reset()
        
    def reset(self):
        self.pos = self.start
        return self.pos
        
    def step(self, action, shortcut_open=False):
        move = self.actions[action]
        nr = self.pos[0] + move[0]
        nc = self.pos[1] + move[1]
        
        # Check boundary
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return self.pos, 0.0, False
            
        # Check wall
        # Wall is at row 3, cols 1 to 8. If shortcut is open, col 8 is open.
        is_wall = False
        if nr == 3:
            if shortcut_open:
                if 1 <= nc <= 7:
                    is_wall = True
            else:
                if 1 <= nc <= 8:
                    is_wall = True
                    
        if is_wall:
            return self.pos, 0.0, False
            
        self.pos = (nr, nc)
        if self.pos == self.goal:
            return self.pos, 1.0, True
        return self.pos, 0.0, False

def dyna_q(env, episodes=200, planning_steps=50, dyna_plus=False, kappa=1e-4):
    Q = {}
    model = {} # model[state][action] = (next_state, reward)
    time_since_visited = {} # For Dyna-Q+
    
    # Initialize values
    for r in range(env.height):
        for c in range(env.width):
            s = (r, c)
            Q[s] = np.zeros(env.n_actions)
            time_since_visited[s] = np.zeros(env.n_actions)
            
    steps_history = []
    total_steps = 0
    
    shortcut_open = False
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_steps = 0
        
        # Open shortcut at total step threshold
        if total_steps > 3000:
            shortcut_open = True
            
        while not done:
            ep_steps += 1
            total_steps += 1
            
            # Select action e-greedy
            if np.random.rand() < 0.1:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])
                
            # Environment step
            next_state, reward, done = env.step(action, shortcut_open)
            
            # Direct Q-update
            best_next = np.argmax(Q[next_state])
            Q[state][action] += 0.1 * (reward + 0.95 * Q[next_state][best_next] - Q[state][action])
            
            # Model update
            if state not in model:
                model[state] = {}
            model[state][action] = (next_state, reward)
            
            # Update time since visited
            for s_key in time_since_visited:
                time_since_visited[s_key] += 1
            time_since_visited[state][action] = 0
            
            # Planning phase
            for _ in range(planning_steps):
                # Pick a previously visited state
                p_state = list(model.keys())[np.random.randint(len(model))]
                # Pick an action (either visited or any action for Dyna-Q+)
                if dyna_plus:
                    p_action = np.random.randint(env.n_actions)
                    if p_action in model[p_state]:
                        p_next, p_reward = model[p_state][p_action]
                    else:
                        p_next, p_reward = p_state, 0.0 # assume staying in state
                        
                    # Add exploration bonus: R + kappa * sqrt(tau)
                    tau = time_since_visited[p_state][p_action]
                    p_reward += kappa * np.sqrt(tau)
                else:
                    p_action = list(model[p_state].keys())[np.random.randint(len(model[p_state]))]
                    p_next, p_reward = model[p_state][p_action]
                    
                best_plan_next = np.argmax(Q[p_next])
                Q[p_state][p_action] += 0.1 * (p_reward + 0.95 * Q[p_next][best_plan_next] - Q[p_state][p_action])
                
            state = next_state
            
        steps_history.append(ep_steps)
        
    return steps_history

if __name__ == "__main__":
    env = ShortcutMaze()
    
    print("Running standard Dyna-Q...")
    dq_steps = dyna_q(env, episodes=150, planning_steps=50, dyna_plus=False)
    
    print("Running Dyna-Q+ (exploration bonus)...")
    dq_plus_steps = dyna_q(env, episodes=150, planning_steps=50, dyna_plus=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(dq_steps), range(len(dq_steps)), label="Dyna-Q", color="red")
    plt.plot(np.cumsum(dq_plus_steps), range(len(dq_plus_steps)), label="Dyna-Q+ (Exploration Bonus)", color="green")
    plt.axvline(x=3000, color="gray", linestyle="--", label="Shortcut Opens (3000 steps)")
    
    plt.xlabel("Cumulative Steps")
    plt.ylabel("Episodes Completed")
    plt.title("Dyna-Q vs Dyna-Q+ in Shortcut Maze")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_37_dynaq_planning_learning.png")
    plt.close()
    
    print("Dyna-Q planning experiment completed. Saved results comparison to exp_37_dynaq_planning_learning.png")
