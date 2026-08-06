"""
Experiment 32: Dueling DQN vs Standard DQN in Gridworld
Objective: Implement Dueling DQN architecture and compare navigation performance with Standard DQN in a 5x5 gridworld.
Method: Deep Q-Network (DQN) Comparison using TensorFlow/Keras
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class GridworldEnv:
    def __init__(self, size=5):
        self.size = size
        self.state_dim = 4 # [agent_r, agent_c, goal_r, goal_c]
        self.action_dim = 4 # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.max_steps = 30
        self.reset()
        
    def reset(self):
        self.agent_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.agent_pos[0]/4.0, self.agent_pos[1]/4.0, self.goal_pos[0]/4.0, self.goal_pos[1]/4.0], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        move = self.actions[action]
        nr = max(0, min(self.size - 1, self.agent_pos[0] + move[0]))
        nc = max(0, min(self.size - 1, self.agent_pos[1] + move[1]))
        self.agent_pos = (nr, nc)
        
        dist = np.sqrt((self.agent_pos[0]-self.goal_pos[0])**2 + (self.agent_pos[1]-self.goal_pos[1])**2)
        reward = -dist * 0.1 - 0.1 # step cost
        done = False
        
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

def build_network(state_dim, action_dim, is_dueling=True):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    
    if is_dueling:
        # Split into Value and Advantage streams
        val = layers.Dense(1, activation='linear')(x)
        adv = layers.Dense(action_dim, activation='linear')(x)
        # Combine
        outputs = layers.Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - tf.reduce_mean(val_adv[1], axis=1, keepdims=True)))([val, adv])
    else:
        outputs = layers.Dense(action_dim, activation='linear')(x)
        
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    return model

class DQNAgent:
    def __init__(self, state_dim, action_dim, is_dueling=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_dueling = is_dueling
        self.memory = []
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.85
        self.epsilon_min = 0.1
        self.model = build_network(state_dim, action_dim, is_dueling)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)
            
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        preds = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(preds[0])
        
    def replay(self, batch_size=16):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        
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
    env = GridworldEnv()
    
    # Train Standard DQN
    print("Training Standard DQN Agent...")
    std_agent = DQNAgent(env.state_dim, env.action_dim, is_dueling=False)
    std_rewards = []
    
    # Train Dueling DQN
    print("Training Dueling DQN Agent...")
    duel_agent = DQNAgent(env.state_dim, env.action_dim, is_dueling=True)
    duel_rewards = []
    
    episodes = 20
    
    # Train Loop for both
    for e in range(episodes):
        # Standard DQN
        state = env.reset()
        done = False
        total_r = 0
        while not done:
            action = std_agent.act(state)
            next_state, reward, done = env.step(action)
            std_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_r += reward
        std_agent.replay(16)
        std_rewards.append(total_r)
        
        # Dueling DQN
        state = env.reset()
        done = False
        total_r = 0
        while not done:
            action = duel_agent.act(state)
            next_state, reward, done = env.step(action)
            duel_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_r += reward
        duel_agent.replay(16)
        duel_rewards.append(total_r)
        
        print(f"Episode: {e+1}/{episodes} | Std Reward: {std_rewards[-1]:.2f} | Dueling Reward: {duel_rewards[-1]:.2f}")
        
    plt.figure(figsize=(8, 4))
    plt.plot(std_rewards, label="Standard DQN", color='magenta', marker='o')
    plt.plot(duel_rewards, label="Dueling DQN", color='cyan', marker='s')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Dueling DQN vs Standard DQN Navigation Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_32_dueling_dqn_gridworld.png")
    plt.close()
    
    print("Dueling comparison finished. Chart saved to exp_32_dueling_dqn_gridworld.png")
