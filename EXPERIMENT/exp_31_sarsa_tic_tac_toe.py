"""
Experiment 31: SARSA Algorithm for Tic-Tac-Toe Board Game
Objective: Develop a SARSA agent to learn Tic-Tac-Toe against a random opponent.
State representation: 9-character string representing board (' ' or 'X' or 'O')
Method: Tabular SARSA
"""

import numpy as np
import matplotlib.pyplot as plt

class TicTacToeEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.board = [' '] * 9
        return self.get_state_str()
        
    def get_state_str(self):
        return "".join(self.board)
        
    def get_available_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == ' ']
        
    def check_winner(self):
        # Winning lines
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # cols
            [0, 4, 8], [2, 4, 6]             # diags
        ]
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] and self.board[line[0]] != ' ':
                return self.board[line[0]]
        if ' ' not in self.board:
            return 'Draw'
        return None
        
    def step(self, action, player='X'):
        # Play player's move
        self.board[action] = player
        winner = self.check_winner()
        
        if winner == player:
            return self.get_state_str(), 10.0, True  # win
        elif winner == 'Draw':
            return self.get_state_str(), 2.0, True   # draw
            
        # Opponent's turn (Random)
        opponent = 'O'
        avail = self.get_available_actions()
        opp_action = np.random.choice(avail)
        self.board[opp_action] = opponent
        
        winner = self.check_winner()
        if winner == opponent:
            return self.get_state_str(), -10.0, True  # lose
        elif winner == 'Draw':
            return self.get_state_str(), 2.0, True    # draw
            
        return self.get_state_str(), -0.5, False      # move cost

class SarsaAgent:
    def __init__(self):
        self.Q = {}  # Q-values dictionary: Q[state_str] = np.zeros(9)
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2
        
    def get_q_values(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(9)
        return self.Q[state]
        
    def choose_action(self, state, available_actions):
        q_vals = self.get_q_values(state)
        # Filter available actions
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            best_val = -float('inf')
            best_action = available_actions[0]
            for a in available_actions:
                if q_vals[a] > best_val:
                    best_val = q_vals[a]
                    best_action = a
            return best_action

def train_sarsa_tictactoe(episodes=2000):
    env = TicTacToeEnv()
    agent = SarsaAgent()
    win_history = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        avail = env.get_available_actions()
        action = agent.choose_action(state, avail)
        
        total_r = 0
        while not done:
            next_state, reward, done = env.step(action)
            total_r += reward
            
            if not done:
                next_avail = env.get_available_actions()
                next_action = agent.choose_action(next_state, next_avail)
                
                # SARSA update
                q_vals = agent.get_q_values(state)
                next_q_vals = agent.get_q_values(next_state)
                q_vals[action] += agent.lr * (reward + agent.gamma * next_q_vals[next_action] - q_vals[action])
                
                state = next_state
                action = next_action
            else:
                # Terminal update
                q_vals = agent.get_q_values(state)
                q_vals[action] += agent.lr * (reward - q_vals[action])
                
        win_history.append(1 if reward == 10.0 else (0 if reward == -10.0 else 0.5))
        
    return agent, win_history

if __name__ == "__main__":
    agent, history = train_sarsa_tictactoe(episodes=1500)
    
    # Smooth win rate
    window = 100
    win_rate = np.convolve(history, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(8, 4))
    plt.plot(win_rate, color='indigo')
    plt.xlabel("Episode")
    plt.ylabel("Win Rate against Random Opponent")
    plt.title("SARSA Tic-Tac-Toe Agent Win-Rate Convergence")
    plt.grid(True)
    plt.savefig("exp_31_sarsa_tic_tac_toe.png")
    plt.close()
    
    print("SARSA Tic-Tac-Toe agent trained. Saved plot to exp_31_sarsa_tic_tac_toe.png")
    print(f"Final 100-episode win-rate: {np.mean(history[-100:]):.2%}")
