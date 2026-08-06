"""
Experiment 11: DQN, Double DQN, Dueling DQN, and PER for Traffic Signal Control
Objective: Compare Q-learning variants for an intersection traffic signal optimizer to minimize waiting times.
Method: Deep RL (DQN, DDQN, Dueling DQN, PER) using TensorFlow/Keras
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class TrafficSignalEnv:
    def __init__(self):
        self.state_dim = 4  # Queue lengths in North, South, East, West directions
        self.action_dim = 2  # 0: Green N-S (Red E-W), 1: Green E-W (Red N-S)
        self.max_steps = 50
        self.reset()
        
    def reset(self):
        self.queues = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        self.steps = 0
        return self.queues.copy()
        
    def step(self, action):
        self.steps += 1
        
        # New vehicles arrive randomly
        arrivals = np.random.poisson(lam=1.5, size=4)
        self.queues += arrivals
        
        # Green light drains queues in active direction
        drain_rate = 4.0
        if action == 0:  # N-S gets green
            self.queues[0] = max(0.0, self.queues[0] - drain_rate)
            self.queues[1] = max(0.0, self.queues[1] - drain_rate)
        else:            # E-W gets green
            self.queues[2] = max(0.0, self.queues[2] - drain_rate)
            self.queues[3] = max(0.0, self.queues[3] - drain_rate)
            
        # Reward is negative sum of wait times (queue lengths)
        reward = -float(np.sum(self.queues))
        done = self.steps >= self.max_steps
        
        return self.queues.copy(), reward, done

def build_q_network(state_dim, action_dim, architecture='standard'):
    """Helper to build Standard vs Dueling architectures"""
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    
    if architecture == 'dueling':
        # Split into Value and Advantage streams
        val_stream = layers.Dense(1, activation='linear')(x)
        adv_stream = layers.Dense(action_dim, activation='linear')(x)
        # Combine: Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
        outputs = layers.Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - tf.reduce_mean(val_adv[1], axis=1, keepdims=True)))([val_stream, adv_stream])
    else:
        outputs = layers.Dense(action_dim, activation='linear')(x)
        
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    return model

class TrafficAgent:
    def __init__(self, state_dim, action_dim, variant='DQN'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.variant = variant
        self.memory = []
        self.priorities = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9
        
        # Build primary and target networks
        arch = 'dueling' if 'Dueling' in self.variant else 'standard'
        self.model = build_q_network(state_dim, action_dim, arch)
        self.target_model = build_q_network(state_dim, action_dim, arch)
        self.update_target_network()
        
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(10.0)  # Max initial priority for new transitions
        if len(self.memory) > 1000:
            self.memory.pop(0)
            self.priorities.pop(0)
            
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        return np.argmax(self.model.predict(state[np.newaxis, :], verbose=0)[0])
        
    def replay(self, batch_size=16):
        if len(self.memory) < batch_size:
            return
            
        # Sample indexes
        if self.variant == 'PER':
            probs = np.array(self.priorities) ** 0.6
            probs /= probs.sum()
            indices = np.random.choice(len(self.memory), batch_size, p=probs)
        else:
            indices = random.sample(range(len(self.memory)), batch_size)
            
        states, targets = [], []
        
        for idx in indices:
            state, action, reward, next_state, done = self.memory[idx]
            
            if done:
                target = reward
            else:
                if self.variant == 'Double_DQN':
                    # Decoupled action selection and evaluation
                    best_action = np.argmax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
                    target = reward + self.gamma * self.target_model.predict(next_state[np.newaxis, :], verbose=0)[0][best_action]
                else:
                    # Standard DQN/Dueling DQN
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state[np.newaxis, :], verbose=0)[0])
                    
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)
            
            # Update priority for PER
            if self.variant == 'PER':
                td_error = abs(target - target_f[0][action])
                self.priorities[idx] = td_error + 1e-5
                
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f[0])
            
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = TrafficSignalEnv()
    variants = ['DQN', 'Double_DQN', 'Dueling_DQN', 'PER']
    episodes = 8
    results = {}
    
    for var in variants:
        print(f"Training agent variant: {var}...")
        agent = TrafficAgent(env.state_dim, env.action_dim, variant=var)
        history = []
        
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
            agent.replay(16)
            if e % 2 == 0:
                agent.update_target_network()
            history.append(total_reward)
            
        results[var] = history
        print(f"  {var} final episode waiting index (reward): {history[-1]:.2f}")
        
    # Plot Comparison
    plt.figure(figsize=(10, 5))
    for var, history in results.items():
        plt.plot(history, label=var, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Waiting Score (Higher is Better / Less Waiting)")
    plt.title("Traffic Light Control: DQN Variants Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_11_traffic_signal_dqn_variants.png")
    plt.close()
    
    print("\nVariant comparison complete. Output saved as exp_11_traffic_signal_dqn_variants.png")
stream = None
