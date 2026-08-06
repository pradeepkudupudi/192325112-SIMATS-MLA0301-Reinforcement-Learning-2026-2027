"""
Experiment 33: DDPG for Real-Time Strategy Game Optimization
Objective: Train an RTS agent to gather resources and build units using continuous control DDPG.
State: [resource_stock, army_strength, enemy_closeness]
Actions: [Gathering Rate, Building Rate] (Continuous actions in [-1.0, 1.0])
Method: Deep Deterministic Policy Gradient (DDPG) using TensorFlow/Keras
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class RTSEnv:
    def __init__(self):
        self.state_dim = 3
        self.action_dim = 2
        self.max_steps = 40
        self.reset()
        
    def reset(self):
        self.resources = 10.0
        self.army = 1.0
        self.enemy_dist = 50.0  # distance of enemy base
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.resources / 100.0, self.army / 20.0, self.enemy_dist / 100.0], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        
        # Scale continuous actions from [-1, 1] to positive rates
        gather_rate = max(0.0, float(action[0] + 1.0) * 5.0)
        build_rate = max(0.0, float(action[1] + 1.0) * 2.0)
        
        # Resources gathered
        self.resources += gather_rate
        
        # Build army units (costs 3.0 resources per unit)
        units_built = min(build_rate, self.resources / 3.0)
        self.resources -= units_built * 3.0
        self.army += units_built
        
        # Enemy approaches base stochastically
        self.enemy_dist -= np.random.uniform(1.0, 3.0)
        
        reward = 0.0
        done = False
        
        # End of match: fight or get invaded
        if self.enemy_dist <= 0:
            done = True
            if self.army >= 10.0:
                reward = 100.0  # Victory
            else:
                reward = -50.0  # Defeat
        else:
            reward = 0.1 * self.resources + 0.5 * self.army - 0.2 * (50.0 - self.enemy_dist)
            
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.gamma = 0.95
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_opt = keras.optimizers.Adam(learning_rate=0.002)
        self.critic_opt = keras.optimizers.Adam(learning_rate=0.005)
        
    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='tanh')(x)
        return Model(inputs=inputs, outputs=outputs)
        
    def _build_critic(self):
        state_in = layers.Input(shape=(self.state_dim,))
        action_in = layers.Input(shape=(self.action_dim,))
        
        x = layers.Concatenate()([state_in, action_in])
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        return Model(inputs=[state_in, action_in], outputs=outputs)
        
    def act(self, state, noise=0.1):
        action = self.actor(state[np.newaxis, :]).numpy()[0]
        # Add exploration noise (OU-like process or gaussian)
        action += np.random.normal(0, noise, size=self.action_dim)
        return np.clip(action, -1.0, 1.0)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)
            
    def train(self, batch_size=16):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch], dtype=np.float32)
        actions = np.array([x[1] for x in minibatch], dtype=np.float32)
        rewards = np.array([x[2] for x in minibatch], dtype=np.float32)
        next_states = np.array([x[3] for x in minibatch], dtype=np.float32)
        dones = np.array([x[4] for x in minibatch], dtype=np.float32)
        
        # Train Critic
        with tf.GradientTape() as tape:
            next_actions = self.actor(next_states)
            target_q = rewards + self.gamma * tf.squeeze(self.critic([next_states, next_actions])) * (1.0 - dones)
            q_values = tf.squeeze(self.critic([states, actions]))
            critic_loss = tf.math.reduce_mean(tf.math.square(target_q - q_values))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        
        # Train Actor
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            actor_loss = -tf.math.reduce_mean(self.critic([states, new_actions]))
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

if __name__ == "__main__":
    env = RTSEnv()
    agent = DDPGAgent(env.state_dim, env.action_dim)
    
    episodes = 25
    rewards_history = []
    
    print("Training RTS DDPG continuous control agent...")
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
            
            agent.train(16)
            
        rewards_history.append(total_r)
        print(f"Episode: {e}/{episodes}, reward: {total_r:.2f}, Army Strength: {env.army:.2f}, Resources: {env.resources:.2f}")
        
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, color='orange', marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Match Score")
    plt.title("RTS Game DDPG continuous training progress")
    plt.grid(True)
    plt.savefig("exp_33_rts_game_ddpg.png")
    plt.close()
    
    print("RTS game simulation finished. Saved performance log to exp_33_rts_game_ddpg.png")
