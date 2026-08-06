"""
Experiment 35: Value-Equivalence Prediction Model for Portfolios
Objective: Estimate long-term portfolio returns of Stock-heavy, Bond-heavy, and Balanced allocation policies.
Method: Temporal Difference (TD) Value Prediction
"""

import numpy as np
import matplotlib.pyplot as plt

class FinancialMarketEnv:
    def __init__(self):
        # Market states: 0: Bear Market, 1: Normal Market, 2: Bull Market
        self.n_states = 3
        # Transitions between market phases
        self.T = np.array([
            [0.4, 0.4, 0.2],  # From Bear
            [0.15, 0.6, 0.25], # From Normal
            [0.1, 0.3, 0.6]   # From Bull
        ])
        
        # Returns for each asset in [Bear, Normal, Bull] states
        # Asset 0: Stocks, Asset 1: Bonds
        self.asset_returns = np.array([
            [-0.10, 0.08, 0.22],  # Stocks
            [ 0.04, 0.04, 0.02]   # Bonds (stable)
        ])
        self.reset()
        
    def reset(self):
        self.state = 1  # Start in Normal market
        return self.state
        
    def step(self, allocation):
        # allocation: [weight_stocks, weight_bonds]
        next_state = np.random.choice(self.n_states, p=self.T[self.state])
        
        # Calculate return
        returns = self.asset_returns[0, self.state] * allocation[0] + self.asset_returns[1, self.state] * allocation[1]
        
        self.state = next_state
        return next_state, returns

def estimate_portfolio_value(env, allocation, episodes=500, steps_per_ep=30, lr=0.05, gamma=0.95):
    """Estimate portfolio value function V(s) using TD(0) prediction"""
    V = np.zeros(env.n_states)
    
    for ep in range(episodes):
        state = env.reset()
        for step in range(steps_per_ep):
            next_state, reward = env.step(allocation)
            
            # TD update: V(S) <- V(S) + alpha * [R + gamma * V(S') - V(S)]
            V[state] += lr * (reward + gamma * V[next_state] - V[state])
            state = next_state
            
    return V

if __name__ == "__main__":
    env = FinancialMarketEnv()
    
    # Portfolio allocations
    portfolios = {
        "Stock-Heavy (90/10)": [0.9, 0.1],
        "Bond-Heavy (10/90)": [0.1, 0.9],
        "Balanced (50/50)": [0.5, 0.5]
    }
    
    results = {}
    for name, alloc in portfolios.items():
        V_est = estimate_portfolio_value(env, alloc)
        results[name] = V_est
        print(f"Portfolio {name} Estimated Long-Term Values V(s):")
        print(f"  Bear Market:   {V_est[0]:.4f}")
        print(f"  Normal Market: {V_est[1]:.4f}")
        print(f"  Bull Market:   {V_est[2]:.4f}\n")
        
    # Save a comparison bar chart
    categories = ["Bear State", "Normal State", "Bull State"]
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width, [results["Stock-Heavy (90/10)"][i] for i in range(3)], width, label="Stock-Heavy", color="crimson")
    rects2 = ax.bar(x, [results["Balanced (50/50)"][i] for i in range(3)], width, label="Balanced", color="gold")
    rects3 = ax.bar(x + width, [results["Bond-Heavy (10/90)"][i] for i in range(3)], width, label="Bond-Heavy", color="dodgerblue")
    
    ax.set_ylabel("Estimated Portfolios Long-Term Value V*(s)")
    ax.set_title("Investment Portfolio Value-Equivalence Analysis (TD-Prediction)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, linestyle=":")
    plt.savefig("exp_35_portfolio_value_equivalence.png")
    plt.close()
    
    print("Portfolio comparison completed. Comparison saved as exp_35_portfolio_value_equivalence.png")
