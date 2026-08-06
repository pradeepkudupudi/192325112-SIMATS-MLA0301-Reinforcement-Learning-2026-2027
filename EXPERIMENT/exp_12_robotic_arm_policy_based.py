"""
Experiment 12: Policy-Based Reinforcement Learning for Industrial Robotic Arm
Objective: Train a robotic arm to perform efficient pick-and-place operations using the REINFORCE Policy Gradient algorithm.
Method: REINFORCE (Monte Carlo Policy Gradient) using TensorFlow/Keras
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class RoboticArmEnv:
    def __init__(self):
        # State: [theta1, theta2, target_x, target_y]
        self.state_dim = 4
        # Actions: 0: theta1+, 1: theta1-, 2: theta2+, 3: theta2-
        self.action_dim = 4
        
        # Link lengths
        self.l1 = 1.0
        self.l2 = 1.0
        
        # Target position
        self.target = (1.2, 0.5)
        self.max_steps = 60
        self.reset()
        
    def reset(self):
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.theta1, self.theta2, self.target[0], self.target[1]], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        delta = 0.05
        
        if action == 0:
            self.theta1 = min(np.pi, self.theta1 + delta)
        elif action == 1:
            self.theta1 = max(-np.pi, self.theta1 - delta)
        elif action == 2:
            self.theta2 = min(np.pi, self.theta2 + delta)
        elif action == 3:
            self.theta2 = max(-np.pi, self.theta2 - delta)
            
        # Kinematics: find end-effector position
        x = self.l1 * np.cos(self.theta1) + self.l2 * np.cos(self.theta1 + self.theta2)
        y = self.l1 * np.sin(self.theta1) + self.l2 * np.sin(self.theta1 + self.theta2)
        
        dist = np.sqrt((x - self.target[0])**2 + (y - self.target[1])**2)
        
        # Reward
        reward = -dist  # Penalize distance to target
        done = False
        
        if dist < 0.15:
            reward = 100.0  # Pick successfully reached
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

class ReinforceAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.model = self._build_model()
        
    def _build_model(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='softmax')(x)
        return Model(inputs=inputs, outputs=outputs)
        
    def get_action(self, state):
        probs = self.model(state[np.newaxis, :]).numpy()[0]
        return np.random.choice(self.action_dim, p=probs)
        
    def train_step(self, states, actions, returns):
        # Minimize -log(pi(a|s))*G
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            action_probs = self.model(states_tensor)
            # Gather probability of chosen actions
            action_masks = tf.one_hot(actions_tensor, self.action_dim)
            chosen_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            loss = -tf.reduce_sum(tf.math.log(chosen_probs + 1e-8) * returns_tensor)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

if __name__ == "__main__":
    env = RoboticArmEnv()
    agent = ReinforceAgent(env.state_dim, env.action_dim)
    
    episodes = 35
    gamma = 0.95
    rewards_history = []
    
    print("Training Robotic Arm with REINFORCE Policy Gradient...")
    for e in range(1, episodes + 1):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            
        total_reward = sum(rewards)
        rewards_history.append(total_reward)
        
        # Calculate returns G_t
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
            
        # Standardize returns to stabilize gradients
        returns = np.array(returns)
        if len(returns) > 1 and np.std(returns) > 0:
            returns = (returns - np.mean(returns)) / np.std(returns)
            
        agent.train_step(states, actions, returns)
        print(f"Episode: {e}/{episodes}, Total Reward: {total_reward:.2f}")
        
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, color='magenta', marker='x')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("REINFORCE Pick-and-Place Robotic Arm Training Progress")
    plt.grid(True)
    plt.savefig("exp_12_robotic_arm_policy_based.png")
    plt.close()
    
    print("Pick-and-place arm simulation completed. Convergence plot saved as exp_12_robotic_arm_policy_based.png")
