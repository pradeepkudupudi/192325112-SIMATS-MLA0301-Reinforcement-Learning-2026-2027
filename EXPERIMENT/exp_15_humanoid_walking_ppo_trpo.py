"""
Experiment 15: PPO and TRPO for Humanoid Walking & Balance
Objective: Implement Proximal Policy Optimization (PPO) clip objective for a balancing/walking simulator.
State: [height, forward_velocity, angle, angular_velocity]
Actions: 0: Lean Left, 1: Lean Right, 2: Do Nothing
Method: PPO with Clipped Policy Objective using TensorFlow/Keras
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class HumanoidBalanceEnv:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 3
        self.max_steps = 80
        self.reset()
        
    def reset(self):
        self.height = 1.0
        self.vel = 0.0
        self.angle = 0.0  # Angle in radians, 0 is vertical balance
        self.ang_vel = 0.0
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.height, self.vel, self.angle, self.ang_vel], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        
        # Apply force based on action
        force = 0.0
        if action == 0:
            force = -0.1
        elif action == 1:
            force = 0.1
            
        # Physics update
        g = 9.8
        dt = 0.05
        
        # Angular acceleration: depends on gravity + torque/force
        ang_acc = g * np.sin(self.angle) + force
        self.ang_vel += ang_acc * dt
        self.angle += self.ang_vel * dt
        
        # Height changes based on angle (taller when vertical, falls down as angle increases)
        self.height = max(0.0, np.cos(self.angle))
        self.vel += 0.01 * np.cos(self.angle)
        
        # Reward: reward height and vertical posture, penalize large angles
        reward = self.height - 0.5 * abs(self.angle)
        done = False
        
        # Termination conditions
        if self.height < 0.5 or abs(self.angle) > 0.8:
            reward = -20.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_ratio = 0.2
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=0.005)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=0.01)
        
        # Models
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='softmax')(x)
        return Model(inputs=inputs, outputs=outputs)
        
    def _build_critic(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(32, activation='relu')(inputs)
        outputs = layers.Dense(1, activation='linear')(x)
        return Model(inputs=inputs, outputs=outputs)
        
    def get_action_and_prob(self, state):
        probs = self.actor(state[np.newaxis, :]).numpy()[0]
        action = np.random.choice(self.action_dim, p=probs)
        return action, probs[action]
        
    def train(self, states, actions, old_probs, returns):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        old_probs = np.array(old_probs, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        # Value Update
        with tf.GradientTape() as tape_v:
            vals = tf.squeeze(self.critic(states))
            v_loss = keras.losses.MSE(returns, vals)
        grads_v = tape_v.gradient(v_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads_v, self.critic.trainable_variables))
        
        # Policy Update (PPO Clipped Objective)
        with tf.GradientTape() as tape_p:
            vals = tf.squeeze(self.critic(states))
            advantages = returns - tf.stop_gradient(vals)
            # normalize advantages
            if np.std(advantages) > 0:
                advantages = (advantages - np.mean(advantages)) / np.std(advantages)
                
            probs = self.actor(states)
            action_masks = tf.one_hot(actions, self.action_dim)
            new_probs = tf.reduce_sum(probs * action_masks, axis=1)
            
            # Ratio r_t = pi_new / pi_old
            ratio = new_probs / (old_probs + 1e-8)
            
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # PPO clip loss
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
        grads_p = tape_p.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads_p, self.actor.trainable_variables))

if __name__ == "__main__":
    env = HumanoidBalanceEnv()
    agent = PPOAgent(env.state_dim, env.action_dim)
    
    episodes = 25
    gamma = 0.98
    rewards_history = []
    
    print("Training Humanoid Balancing Agent with PPO Clipped Objective...")
    for e in range(1, episodes + 1):
        states, actions, old_probs, rewards = [], [], [], []
        state = env.reset()
        done = False
        
        while not done:
            action, prob = agent.get_action_and_prob(state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            old_probs.append(prob)
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
            
        agent.train(states, actions, old_probs, returns)
        print(f"Episode: {e}/{episodes}, Balance Score: {total_reward:.2f}, steps: {env.steps}")
        
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, color='coral', marker='d')
    plt.xlabel("Episode")
    plt.ylabel("Total Balance Score")
    plt.title("PPO Humanoid Walking/Balance Training curve")
    plt.grid(True)
    plt.savefig("exp_15_humanoid_walking_ppo_trpo.png")
    plt.close()
    
    print("Humanoid balance model training completed. Saved plot to exp_15_humanoid_walking_ppo_trpo.png")
