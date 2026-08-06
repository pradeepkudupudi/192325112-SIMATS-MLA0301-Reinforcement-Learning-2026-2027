"""
Experiment 5: Epsilon-Greedy Multi-Armed Bandit for Ad Recommendation
Objective: Implement epsilon-greedy multi-armed bandit algorithm for ad click-through rate (CTR) optimization.
Method: Epsilon-Greedy Bandit Strategy
"""

import numpy as np
import matplotlib.pyplot as plt

class AdRecommendationSystem:
    def __init__(self, true_ctrs):
        self.true_ctrs = true_ctrs
        self.n_ads = len(true_ctrs)
        
    def show_ad(self, ad_idx):
        # Pull arm: Returns 1 (click) or 0 (no click) based on true CTR
        prob = self.true_ctrs[ad_idx]
        return 1 if np.random.rand() < prob else 0

def run_epsilon_greedy(system, epsilon=0.1, steps=2000):
    estimates = np.zeros(system.n_ads)
    counts = np.zeros(system.n_ads)
    
    total_clicks = 0
    clicks_history = []
    ctr_history = []
    
    for step in range(1, steps + 1):
        if np.random.rand() < epsilon:
            # Explore
            choice = np.random.randint(system.n_ads)
        else:
            # Exploit
            choice = np.argmax(estimates)
            
        reward = system.show_ad(choice)
        
        # Update estimates (Incremental implementation)
        counts[choice] += 1
        estimates[choice] += (reward - estimates[choice]) / counts[choice]
        
        total_clicks += reward
        clicks_history.append(total_clicks)
        ctr_history.append(total_clicks / step)
        
    return estimates, counts, ctr_history

if __name__ == "__main__":
    # 5 Ads with their true conversion probabilities (CTRs)
    true_ctrs = [0.03, 0.08, 0.18, 0.05, 0.12]  # Ad 2 has the highest CTR (0.18)
    system = AdRecommendationSystem(true_ctrs)
    
    steps = 5000
    epsilons = [0.01, 0.1, 0.2, 0.5]
    
    plt.figure(figsize=(10, 5))
    
    for eps in epsilons:
        est, counts, ctr_hist = run_epsilon_greedy(system, epsilon=eps, steps=steps)
        print(f"Epsilon = {eps}:")
        print(f"  True CTRs: {true_ctrs}")
        print(f"  Estimated CTRs: {est}")
        print(f"  Selection Counts: {counts}")
        plt.plot(ctr_hist, label=f"Epsilon = {eps}")
        
    plt.axhline(y=max(true_ctrs), color='r', linestyle='--', label="Optimal CTR")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative CTR")
    plt.title("Epsilon-Greedy Ad Recommendation CTR Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_05_ad_recommendation_bandit.png")
    plt.close()
    
    print("\nComparison complete. Plot saved as exp_05_ad_recommendation_bandit.png")
