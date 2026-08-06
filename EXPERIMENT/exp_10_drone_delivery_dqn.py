"""
Experiment 10: Deep Q-Network (DQN) for Drone Delivery under Battery Constraints
Objective: Develop a DQN to optimize drone delivery routes under battery capacity constraints.
Method: Deep Q-Network (DQN) using TensorFlow/Keras
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

class DroneDeliveryEnv:
    def __init__(self):
        self.state_dim = 4  # [x, y, battery, cargo_weight]
        self.action_dim = 4  # 0: Up, 1: Down, 2: Left, 3: Right
        self.max_steps = 100
        self.reset()
        
    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.battery = 1.0  # Normalized: starts at 100%
        self.cargo_weight = 0.5  # Weight of delivery package
        self.goal_x = 0.8
        self.goal_y = 0.8
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.x, self.y, self.battery, self.cargo_weight], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        step_cost = 0.08 + (0.04 * self.cargo_weight)  # Battery consumption rate based on cargo
        self.battery -= step_cost
        
        step_dist = 0.15
        if action == 0:   # UP
            self.y = min(1.0, self.y + step_dist)
        elif action == 1: # DOWN
            self.y = max(0.0, self.y - step_dist)
        elif action == 2: # LEFT
            self.x = max(0.0, self.x - step_dist)
        elif action == 3: # RIGHT
            self.x = min(1.0, self.x - step_dist)
            
        dist = np.sqrt((self.x - self.goal_x)**2 + (self.y - self.goal_y)**2)
        
        # Check terminals
        done = False
        reward = -dist  # proximity reward
        
        if dist < 0.1:
            reward = 100.0  # Large delivery success reward
            done = True
        elif self.battery <= 0:
            reward = -100.0 # Drone crashed / ran out of battery
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9
        self.lr = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)
            
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        act_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(act_values[0])
        
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)
            target_f[0][action] = target
            
            states.append(state)
            targets.append(target_f[0])
            
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = DroneDeliveryEnv()
    agent = DQNAgent(env.state_dim, env.action_dim)
    episodes = 20
    batch_size = 16
    rewards_history = []
    
    print("Training Drone Delivery DQN Agent under battery constraints...")
    for e in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {e}/{episodes}, reward: {total_reward:.2f}, steps: {env.steps}, remaining battery: {env.battery:.2f}")
                break
                
        agent.replay(batch_size)
        rewards_history.append(total_reward)
        
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, color='blue', marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Drone Delivery DQN Training Curve")
    plt.grid(True)
    plt.savefig("exp_10_drone_delivery_dqn.png")
    plt.close()
    
    print("Training finished. Progress saved to exp_10_drone_delivery_dqn.png")
