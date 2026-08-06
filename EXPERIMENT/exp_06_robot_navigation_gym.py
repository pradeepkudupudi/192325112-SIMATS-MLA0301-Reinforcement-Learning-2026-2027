"""
Experiment 6: Deep Reinforcement Learning for Autonomous Robot Navigation
Objective: Train a Deep Q-Network (DQN) for a robot navigating a continuous 2D area with obstacles.
Environment: Custom Gym-like environment.
Framework: TensorFlow / Keras
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

class RobotGymEnv:
    """A custom self-contained OpenAI Gym-like environment for Robot Navigation"""
    def __init__(self):
        self.state_dim = 4  # [robot_x, robot_y, goal_x, goal_y]
        self.action_dim = 4  # 0: Up, 1: Down, 2: Left, 3: Right
        self.max_steps = 100
        self.reset()
        
    def reset(self):
        self.robot_x = 0.1
        self.robot_y = 0.1
        self.goal_x = 0.9
        self.goal_y = 0.9
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.robot_x, self.robot_y, self.goal_x, self.goal_y], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        step_size = 0.1
        
        # Take action
        if action == 0:   # UP
            self.robot_y = min(1.0, self.robot_y + step_size)
        elif action == 1: # DOWN
            self.robot_y = max(0.0, self.robot_y - step_size)
        elif action == 2: # LEFT
            self.robot_x = max(0.0, self.robot_x - step_size)
        elif action == 3: # RIGHT
            self.robot_x = min(1.0, self.robot_x + step_size)
            
        # Distance to goal
        dist = np.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)
        
        # Reward function
        reward = -dist  # Penalize distance to goal
        done = False
        
        if dist < 0.15:
            reward = 20.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95
        self.lr = 0.005
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
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
    env = RobotGymEnv()
    agent = DQNAgent(env.state_dim, env.action_dim)
    episodes = 25
    batch_size = 16
    reward_history = []
    
    print("Training DQN Autonomous Robot Navigation Agent...")
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
                print(f"Episode: {e}/{episodes}, score: {total_reward:.2f}, epsilon: {agent.epsilon:.2f}")
                break
                
        agent.replay(batch_size)
        reward_history.append(total_reward)
        
    # Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(reward_history, color='orange', marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Robot Navigation DQN Training Progress")
    plt.grid(True)
    plt.savefig("exp_06_robot_navigation_gym.png")
    plt.close()
    
    print("DQN Training complete. Progress plot saved to exp_06_robot_navigation_gym.png")
