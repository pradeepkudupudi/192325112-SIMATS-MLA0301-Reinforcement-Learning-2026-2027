"""
Experiment 25: Ad CTR Optimization - Epsilon-Greedy, UCB, and Thompson Sampling
Objective: Implement and compare Epsilon-Greedy, UCB1, and Thompson Sampling for online ad selection.
Method: Multi-Armed Bandit Algorithms Comparison
"""

import numpy as np
import matplotlib.pyplot as plt

class BanditAdPlatform:
    def __init__(self, true_ctrs):
        self.true_ctrs = true_ctrs
        self.n_arms = len(true_ctrs)
        
    def pull(self, arm):
        return 1 if np.random.rand() < self.true_ctrs[arm] else 0

def run_epsilon_greedy(platform, epsilon=0.1, steps=2000):
    estimates = np.zeros(platform.n_arms)
    counts = np.zeros(platform.n_arms)
    clicks = 0
    history = []
    
    for step in range(1, steps + 1):
        if np.random.rand() < epsilon:
            arm = np.random.randint(platform.n_arms)
        else:
            arm = np.argmax(estimates)
            
        reward = platform.pull(arm)
        counts[arm] += 1
        estimates[arm] += (reward - estimates[arm]) / counts[arm]
        clicks += reward
        history.append(clicks / step)
    return history

def run_ucb(platform, c=1.5, steps=2000):
    estimates = np.zeros(platform.n_arms)
    counts = np.zeros(platform.n_arms)
    clicks = 0
    history = []
    
    # Try each arm once initially to avoid division by zero
    for step in range(1, platform.n_arms + 1):
        arm = step - 1
        reward = platform.pull(arm)
        counts[arm] += 1
        estimates[arm] = reward
        clicks += reward
        history.append(clicks / step)
        
    for step in range(platform.n_arms + 1, steps + 1):
        # UCB Choice: argmax [ Q(a) + c * sqrt(ln(t) / N(a)) ]
        ucb_values = estimates + c * np.sqrt(np.log(step) / (counts + 1e-8))
        arm = np.argmax(ucb_values)
        
        reward = platform.pull(arm)
        counts[arm] += 1
        estimates[arm] += (reward - estimates[arm]) / counts[arm]
        clicks += reward
        history.append(clicks / step)
    return history

def run_thompson_sampling(platform, steps=2000):
    # Beta distribution priors: alpha=1, beta=1
    alphas = np.ones(platform.n_arms)
    betas = np.ones(platform.n_arms)
    clicks = 0
    history = []
    
    for step in range(1, steps + 1):
        # Sample from Beta posterior distribution for each arm
        samples = [np.random.beta(alphas[i], betas[i]) for i in range(platform.n_arms)]
        arm = np.argmax(samples)
        
        reward = platform.pull(arm)
        
        # Update Beta parameters
        if reward == 1:
            alphas[arm] += 1
        else:
            betas[arm] += 1
            
        clicks += reward
        history.append(clicks / step)
    return history

if __name__ == "__main__":
    true_ctrs = [0.04, 0.09, 0.22, 0.11, 0.15]  # Arm 2 (0.22) is optimal
    platform = BanditAdPlatform(true_ctrs)
    steps = 3000
    
    eps_hist = run_epsilon_greedy(platform, epsilon=0.1, steps=steps)
    ucb_hist = run_ucb(platform, c=1.5, steps=steps)
    th_hist = run_thompson_sampling(platform, steps=steps)
    
    plt.figure(figsize=(10, 5))
    plt.plot(eps_hist, label="Epsilon-Greedy (e=0.1)", color="blue")
    plt.plot(ucb_hist, label="UCB1 (c=1.5)", color="orange")
    plt.plot(th_hist, label="Thompson Sampling", color="green")
    plt.axhline(y=max(true_ctrs), color="red", linestyle="--", label="Optimal CTR (0.22)")
    
    plt.xlabel("Interaction Step")
    plt.ylabel("Cumulative Click-Through Rate")
    plt.title("Online Ad Bandits: Epsilon-Greedy vs UCB vs Thompson Sampling")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_25_ad_bandit_algorithms.png")
    plt.close()
    
    print("Bandit comparison complete. Saved chart to exp_25_ad_bandit_algorithms.png")
    print(f"Final CTRs:\n  Epsilon-Greedy: {eps_hist[-1]:.4f}\n  UCB1: {ucb_hist[-1]:.4f}\n  Thompson Sampling: {th_hist[-1]:.4f}")
