"""
Experiment 4: Bellman Equations for Delivery Robot
Objective: Implement Bellman expectation and optimality equations for an autonomous delivery robot routing on a graph.
Method: Analytical Bellman Equation Solver using Value Iteration (Minimizing Cost)
"""

import numpy as np
import matplotlib.pyplot as plt

class DeliveryGraph:
    def __init__(self):
        # Nodes/states: A, B, C, D, E, F, Goal(G)
        self.nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.n_states = len(self.nodes)
        
        # Travel costs (directed edges)
        # edge: (from, to) -> cost
        self.edges = {
            ('A', 'B'): 2, ('A', 'C'): 5,
            ('B', 'D'): 3, ('B', 'E'): 6,
            ('C', 'E'): 2, ('C', 'F'): 4,
            ('D', 'G'): 8,
            ('E', 'G'): 3,
            ('F', 'G'): 5,
        }
        
        self.gamma = 1.0  # Undiscounted for shortest path / minimum cost routing
        self.theta = 1e-6
        self.V = np.zeros(self.n_states)
        # Goal state G has value 0
        self.goal_idx = self.node_to_idx['G']
        
    def solve_bellman_optimality(self):
        """
        Bellman Optimality: V*(s) = min_a [ cost(s, a) + gamma * V*(s') ]
        """
        iterations = 0
        while True:
            delta = 0
            new_V = np.copy(self.V)
            
            for state_name in self.nodes:
                s_idx = self.node_to_idx[state_name]
                if state_name == 'G':
                    new_V[s_idx] = 0  # Goal state always has cost 0
                    continue
                
                # Find all outgoing actions/edges
                costs = []
                for edge, cost in self.edges.items():
                    if edge[0] == state_name:
                        next_state = edge[1]
                        ns_idx = self.node_to_idx[next_state]
                        costs.append(cost + self.gamma * self.V[ns_idx])
                        
                if len(costs) > 0:
                    new_val = min(costs)
                    delta = max(delta, abs(new_val - self.V[s_idx]))
                    new_V[s_idx] = new_val
                else:
                    new_V[s_idx] = float('inf')  # Dead end
                    
            self.V = new_V
            iterations += 1
            if delta < self.theta or iterations > 1000:
                break
        print(f"Bellman optimality solver converged in {iterations} iterations.")

    def get_optimal_path(self, start='A'):
        path = [start]
        curr = start
        total_cost = 0
        while curr != 'G':
            best_next = None
            best_val = float('inf')
            best_cost = 0
            for edge, cost in self.edges.items():
                if edge[0] == curr:
                    next_node = edge[1]
                    ns_idx = self.node_to_idx[next_node]
                    val = cost + self.gamma * self.V[ns_idx]
                    if val < best_val:
                        best_val = val
                        best_next = next_node
                        best_cost = cost
            if best_next is None:
                print("No path found!")
                break
            curr = best_next
            path.append(curr)
            total_cost += best_cost
        return path, total_cost

if __name__ == "__main__":
    solver = DeliveryGraph()
    solver.solve_bellman_optimality()
    
    print("\nState Values (Minimum cost to reach G):")
    for n in solver.nodes:
        idx = solver.node_to_idx[n]
        print(f"Node {n}: {solver.V[idx]:.2f}")
        
    path, cost = solver.get_optimal_path('A')
    print(f"\nOptimal Routing Path from A to G: {' -> '.join(path)}")
    print(f"Total Path Cost: {cost}")
    
    # Save a visualization of the path values
    plt.figure(figsize=(8, 4))
    plt.bar(solver.nodes[:-1], [solver.V[solver.node_to_idx[n]] for n in solver.nodes[:-1]], color='skyblue')
    plt.ylabel("Min Cost to Reach Goal (G)")
    plt.xlabel("Nodes")
    plt.title("Bellman Optimality - Node Minimum Cost Values")
    plt.savefig("exp_04_delivery_robot_bellman.png")
    plt.close()
