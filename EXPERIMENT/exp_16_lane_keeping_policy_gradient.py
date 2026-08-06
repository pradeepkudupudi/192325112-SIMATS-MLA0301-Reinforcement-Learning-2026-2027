"""
Experiment 16: Policy Gradient for Autonomous Lane Keeping
Objective: Compare Policy Gradient (REINFORCE vs A2C) for autonomous lane-keeping.
State: [lateral_deviation, heading_error]
Actions: 0: Steer Left, 1: Steer Right, 2: Keep Straight
Method: Policy Gradient Comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class LaneKeepingEnv:
    def __init__(self):
        self.state_dim = 2  # [lateral_deviation, heading_error]
        self.action_dim = 3  # 0: Left, 1: Right, 2: Straight
        self.max_steps = 60
        self.reset()
        
    def reset(self):
        self.y = 0.5  # lateral deviation (center is 0)
        self.theta = 0.1  # heading error in rad
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.y, self.theta], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        
        # Action updates heading
        steer = 0.05
        if action == 0:
            self.theta -= steer
        elif action == 1:
            self.theta += steer
            
        # Vehicle updates position
        speed = 0.2
        self.y += speed * np.sin(self.theta)
        
        # Penalize deviation and heading angle error
        reward = -(10.0 * abs(self.y) + 2.0 * abs(self.theta))
        done = False
        
        # Check boundary/crashed
        if abs(self.y) > 2.0:
            reward = -100.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

def build_pg_model(state_dim, action_dim):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(action_dim, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005))
    return model

def build_critic_model(state_dim):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    outputs = layers.Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

def train_reinforce(env, episodes=25, gamma=0.95):
    agent = build_pg_model(env.state_dim, env.action_dim)
    history = []
    
    for e in range(episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        
        while not done:
            probs = agent(state[np.newaxis, :]).numpy()[0]
            action = np.random.choice(env.action_dim, p=probs)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            
        # Update policy
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        if np.std(returns) > 0:
            returns = (returns - np.mean(returns)) / np.std(returns)
            
        # Manual backprop
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            probs = agent(states_tensor)
            masks = tf.one_hot(actions_tensor, env.action_dim)
            chosen_probs = tf.reduce_sum(probs * masks, axis=1)
            loss = -tf.reduce_sum(tf.math.log(chosen_probs + 1e-8) * returns_tensor)
            
        grads = tape.gradient(loss, agent.trainable_variables)
        agent.optimizer.apply_gradients(zip(grads, agent.trainable_variables))
        history.append(sum(rewards))
        
    return history

def train_actor_critic(env, episodes=25, gamma=0.95):
    actor = build_pg_model(env.state_dim, env.action_dim)
    critic = build_critic_model(env.state_dim)
    history = []
    
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            probs = actor(state[np.newaxis, :]).numpy()[0]
            action = np.random.choice(env.action_dim, p=probs)
            next_state, reward, done = env.step(action)
            
            # Critic estimate
            state_val = critic(state[np.newaxis, :])[0][0]
            next_state_val = critic(next_state[np.newaxis, :])[0][0]
            
            # TD update target
            target = reward if done else reward + gamma * next_state_val
            advantage = target - state_val
            
            # Train Critic
            critic.fit(state[np.newaxis, :], np.array([[target]]), epochs=1, verbose=0)
            
            # Train Actor manually
            state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
            with tf.GradientTape() as tape:
                probs = actor(state_tensor)
                loss = -tf.math.log(probs[0][action] + 1e-8) * advantage
            grads = tape.gradient(loss, actor.trainable_variables)
            actor.optimizer.apply_gradients(zip(grads, actor.trainable_variables))
            
            state = next_state
            total_reward += reward
            
        history.append(total_reward)
        
    return history

if __name__ == "__main__":
    env = LaneKeepingEnv()
    print("Training REINFORCE agent...")
    reinforce_hist = train_reinforce(env)
    print("Training Actor-Critic (A2C) agent...")
    ac_hist = train_actor_critic(env)
    
    # Save comparison chart
    plt.figure(figsize=(10, 5))
    plt.plot(reinforce_hist, label="REINFORCE (Monte Carlo)", color='blue')
    plt.plot(ac_hist, label="Actor-Critic (A2C)", color='green')
    plt.xlabel("Episode")
    plt.ylabel("Lane Keeping Score")
    plt.title("Lane-Keeping Performance: REINFORCE vs A2C")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_16_lane_keeping_policy_gradient.png")
    plt.close()
    
    print("Comparison complete. Plot saved as exp_16_lane_keeping_policy_gradient.png")
