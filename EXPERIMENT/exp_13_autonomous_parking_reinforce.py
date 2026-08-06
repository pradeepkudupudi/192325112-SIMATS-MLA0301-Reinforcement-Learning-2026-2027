"""
Experiment 13: REINFORCE for Autonomous Parking System
Objective: Train a vehicle to park in a designated parking slot using REINFORCE with a baseline.
State: [car_x, car_y, car_theta, target_x, target_y]
Actions: 0: Go straight, 1: Steer Left, 2: Steer Right, 3: Reverse
Method: REINFORCE with Baseline (Value network) using TensorFlow/Keras
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class ParkingEnv:
    def __init__(self):
        self.state_dim = 5
        self.action_dim = 4
        self.target = (0.8, 0.8)
        self.max_steps = 50
        self.reset()
        
    def reset(self):
        self.x = 0.2
        self.y = 0.2
        self.theta = 0.0  # Angle in radians
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.x, self.y, self.theta, self.target[0], self.target[1]], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        speed = 0.1
        turn_angle = 0.2  # steering sensitivity
        
        # Actions: 0: Straight, 1: Left-turn, 2: Right-turn, 3: Reverse
        if action == 0:
            self.x += speed * np.cos(self.theta)
            self.y += speed * np.sin(self.theta)
        elif action == 1:
            self.theta += turn_angle
            self.x += speed * np.cos(self.theta)
            self.y += speed * np.sin(self.theta)
        elif action == 2:
            self.theta -= turn_angle
            self.x += speed * np.cos(self.theta)
            self.y += speed * np.sin(self.theta)
        elif action == 3:
            self.x -= speed * np.cos(self.theta)
            self.y -= speed * np.sin(self.theta)
            
        # Bound coords to (0,1)
        self.x = max(0.0, min(1.0, self.x))
        self.y = max(0.0, min(1.0, self.y))
        
        # Calculate distance to parking spot
        dist = np.sqrt((self.x - self.target[0])**2 + (self.y - self.target[1])**2)
        
        reward = -dist
        done = False
        
        if dist < 0.12:
            reward = 150.0  # Successfully parked
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

class ReinforceBaselineAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=0.005)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=0.01)
        
        # Networks
        self.policy = self._build_policy()
        self.value_net = self._build_value_net()
        
    def _build_policy(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='softmax')(x)
        return Model(inputs=inputs, outputs=outputs)
        
    def _build_value_net(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(32, activation='relu')(inputs)
        outputs = layers.Dense(1, activation='linear')(x)
        return Model(inputs=inputs, outputs=outputs)
        
    def get_action(self, state):
        probs = self.policy(state[np.newaxis, :]).numpy()[0]
        return np.random.choice(self.action_dim, p=probs)
        
    def train(self, states, actions, returns):
        states_t = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns_t = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # 1. Update Value Network (Baseline)
        with tf.GradientTape() as tape_val:
            state_values = tf.squeeze(self.value_net(states_t))
            val_loss = keras.losses.MSE(returns_t, state_values)
        grads_val = tape_val.gradient(val_loss, self.value_net.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads_val, self.value_net.trainable_variables))
        
        # 2. Update Policy Network using Advantage: (G - V(s))
        with tf.GradientTape() as tape_pol:
            state_values = tf.squeeze(self.value_net(states_t))
            advantages = returns_t - tf.stop_gradient(state_values)
            
            action_probs = self.policy(states_t)
            masks = tf.one_hot(actions_t, self.action_dim)
            chosen_probs = tf.reduce_sum(action_probs * masks, axis=1)
            
            pol_loss = -tf.reduce_sum(tf.math.log(chosen_probs + 1e-8) * advantages)
            
        grads_pol = tape_pol.gradient(pol_loss, self.policy.trainable_variables)
        self.policy.optimizer = self.policy_optimizer
        self.policy_optimizer.apply_gradients(zip(grads_pol, self.policy.trainable_variables))

if __name__ == "__main__":
    env = ParkingEnv()
    agent = ReinforceBaselineAgent(env.state_dim, env.action_dim)
    
    episodes = 35
    gamma = 0.98
    rewards_history = []
    
    print("Training Autonomous Parking agent using REINFORCE with baseline...")
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
        
        # Calculate returns G
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        
        agent.train(states, actions, returns)
        print(f"Episode: {e}/{episodes}, score: {total_reward:.2f}")
        
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, color='brown', marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("REINFORCE Parking Agent Learning Curve")
    plt.grid(True)
    plt.savefig("exp_13_autonomous_parking_reinforce.png")
    plt.close()
    
    print("Autonomous parking training finished. Visual saved to exp_13_autonomous_parking_reinforce.png")
