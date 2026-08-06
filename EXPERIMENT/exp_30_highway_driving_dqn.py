"""
Experiment 30: Deep Q-Network (DQN) for Highway Driving
Objective: Train an autonomous vehicle to maintain safety and speed in simulated traffic.
State: [ego_speed, distance_to_lead, lead_speed]
Actions: 0: Coast (Hold Speed), 1: Accelerate, 2: Decelerate
Method: DQN using TensorFlow/Keras
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

class HighwayDriveEnv:
    def __init__(self):
        self.state_dim = 3
        self.action_dim = 3
        self.max_steps = 100
        self.reset()
        
    def reset(self):
        self.ego_speed = 20.0  # m/s
        self.lead_dist = 50.0  # m
        self.lead_speed = 18.0 # m/s
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.ego_speed / 30.0, self.lead_dist / 100.0, self.lead_speed / 30.0], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        
        # Apply action
        accel = 0.0
        if action == 1:   # Accelerate
            accel = 1.5
        elif action == 2: # Decelerate
            accel = -2.5
            
        # Ego physics update
        self.ego_speed = max(0.0, min(30.0, self.ego_speed + accel * 0.2))
        
        # Lead vehicle speed variation
        self.lead_speed = max(10.0, min(25.0, self.lead_speed + np.random.uniform(-1.0, 1.0)))
        
        # Distance delta
        self.lead_dist += (self.lead_speed - self.ego_speed) * 0.2
        
        # Reward design
        # Target safety distance is 20-30 meters
        if 20.0 <= self.lead_dist <= 40.0:
            reward = 2.0
        else:
            reward = -0.05 * abs(self.lead_dist - 30.0) # penalty for being too far or close
            
        # Speed reward
        reward += self.ego_speed * 0.1
        
        done = False
        if self.lead_dist < 5.0:
            reward = -100.0 # Rear-end collision
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

class DQNDriver:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.1
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)
            
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        preds = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(preds[0])
        
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
    env = HighwayDriveEnv()
    agent = DQNDriver(env.state_dim, env.action_dim)
    
    episodes = 20
    batch_size = 16
    rewards_history = []
    
    print("Training DQN Driving Agent...")
    for e in range(1, episodes + 1):
        state = env.reset()
        total_r = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_r += reward
            
        agent.replay(batch_size)
        rewards_history.append(total_r)
        print(f"Episode: {e}/{episodes}, reward: {total_r:.2f}, remaining dist: {env.lead_dist:.2f}m")
        
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, color='teal', marker='^')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Autonomous Driving Training Curve")
    plt.grid(True)
    plt.savefig("exp_30_highway_driving_dqn.png")
    plt.close()
    
    print("Driving agent training finished. Saved model metrics to exp_30_highway_driving_dqn.png")
