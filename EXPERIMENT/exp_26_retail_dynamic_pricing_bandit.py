"""
Experiment 26: Dynamic Retail Pricing via Multi-Armed Bandits
Objective: Simulate dynamic pricing strategies using Epsilon-Greedy, UCB, and Thompson Sampling to maximize revenue.
Pricing Choices: $10, $15, $20, $25, $30
Method: Multi-Armed Bandit Revenue Comparison
"""

import numpy as np
import matplotlib.pyplot as plt

class DynamicPricingEnvironment:
    def __init__(self):
        self.prices = [10.0, 15.0, 20.0, 25.0, 30.0]
        # P(buy | price)
        self.buy_probabilities = [0.80, 0.65, 0.55, 0.35, 0.20]
        # Expected revenues: [8.0, 9.75, 11.0, 8.75, 6.0] -> $20.0 is the revenue-maximizing price
        self.n_arms = len(self.prices)
        
    def purchase_simulation(self, price_idx):
        prob = self.buy_probabilities[price_idx]
        bought = 1 if np.random.rand() < prob else 0
        revenue = bought * self.prices[price_idx]
        return revenue

def run_eps_greedy_pricing(env, epsilon=0.1, steps=2000):
    estimates = np.zeros(env.n_arms)
    counts = np.zeros(env.n_arms)
    total_revenue = 0
    history = []
    
    for step in range(1, steps + 1):
        if np.random.rand() < epsilon:
            choice = np.random.randint(env.n_arms)
        else:
            choice = np.argmax(estimates)
            
        revenue = env.purchase_simulation(choice)
        counts[choice] += 1
        estimates[choice] += (revenue - estimates[choice]) / counts[choice]
        total_revenue += revenue
        history.append(total_revenue)
        
    return history

def run_ucb_pricing(env, c=8.0, steps=2000):
    estimates = np.zeros(env.n_arms)
    counts = np.zeros(env.n_arms)
    total_revenue = 0
    history = []
    
    # Try each once
    for step in range(1, env.n_arms + 1):
        choice = step - 1
        revenue = env.purchase_simulation(choice)
        counts[choice] += 1
        estimates[choice] = revenue
        total_revenue += revenue
        history.append(total_revenue)
        
    for step in range(env.n_arms + 1, steps + 1):
        ucb_values = estimates + c * np.sqrt(np.log(step) / (counts + 1e-8))
        choice = np.argmax(ucb_values)
        
        revenue = env.purchase_simulation(choice)
        counts[choice] += 1
        estimates[choice] += (revenue - estimates[choice]) / counts[choice]
        total_revenue += revenue
        history.append(total_revenue)
        
    return history

def run_thompson_pricing(env, steps=2000):
    # To use Beta distribution Thompson Sampling for continuous/scaled rewards:
    # We maintain alpha/beta of binary outcomes (buyer bought or did not buy)
    alphas = np.ones(env.n_arms)
    betas = np.ones(env.n_arms)
    total_revenue = 0
    history = []
    
    for step in range(1, steps + 1):
        # Sample probability of buying and calculate expected revenue
        samples = [np.random.beta(alphas[i], betas[i]) * env.prices[i] for i in range(env.n_arms)]
        choice = np.argmax(samples)
        
        revenue = env.purchase_simulation(choice)
        
        # Binary feedback update: was there a purchase?
        purchase_occurred = 1 if revenue > 0 else 0
        if purchase_occurred == 1:
            alphas[choice] += 1
        else:
            betas[choice] += 1
            
        total_revenue += revenue
        history.append(total_revenue)
        
    return history

if __name__ == "__main__":
    env = DynamicPricingEnvironment()
    steps = 2500
    
    eps_rev = run_eps_greedy_pricing(env, epsilon=0.1, steps=steps)
    ucb_rev = run_ucb_pricing(env, c=15.0, steps=steps)
    th_rev = run_thompson_pricing(env, steps=steps)
    
    plt.figure(figsize=(10, 5))
    plt.plot(eps_rev, label="Epsilon-Greedy (e=0.1)", color="blue")
    plt.plot(ucb_rev, label="UCB (c=15)", color="orange")
    plt.plot(th_rev, label="Thompson Sampling", color="green")
    
    # Perfect pricing baseline (always pick price index 2: $20, exp revenue = $11)
    plt.plot([11.0 * i for i in range(steps)], label="Theoretical Optimal Revenue", color="red", linestyle=":")
    
    plt.xlabel("Customer Interaction")
    plt.ylabel("Cumulative Revenue ($)")
    plt.title("Dynamic Pricing Bandit Strategies Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_26_retail_dynamic_pricing_bandit.png")
    plt.close()
    
    print("Pricing simulation complete. Comparison graph saved to exp_26_retail_dynamic_pricing_bandit.png")
    print(f"Total Cumulative Revenue realized:\n  Epsilon-Greedy: ${eps_rev[-1]:.2f}\n  UCB: ${ucb_rev[-1]:.2f}\n  Thompson Sampling: ${th_rev[-1]:.2f}")
