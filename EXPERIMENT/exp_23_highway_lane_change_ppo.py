"""
Experiment 23: PPO for Highway Lane-Changing Decisions
Objective: Train an autonomous vehicle to change lanes to avoid slow traffic using Proximal Policy Optimization (PPO).
State: [current_lane, current_speed, distance_to_lead, lead_speed]
Actions: 0: Keep Lane, 1: Change Lane Left, 2: Change Lane Right
Method: PPO with Clipped Policy Gradient using TensorFlow/Keras
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class HighwayLaneEnv:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 3
        self.max_steps = 50
        self.reset()
        
    def reset(self):
        self.lane = 1  # lanes are 0, 1, 2
        self.speed = 25.0  # m/s
        self.lead_dist = 60.0  # meters
        self.lead_speed = 15.0  # m/s
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.lane, self.speed, self.lead_dist, self.lead_speed], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        
        # Slower traffic logic
        self.lead_dist += (self.lead_speed - self.speed) * 0.2  # update distance
        
        # Process actions
        lane_change_cost = 0.0
        if action == 1:   # change left
            if self.lane > 0:
                self.lane -= 1
                self.lead_dist = np.random.uniform(50.0, 100.0) # free lane ahead
                self.lead_speed = np.random.uniform(15.0, 30.0)
                lane_change_cost = -0.5
            else:
                lane_change_cost = -5.0  # Went off road / illegal change
        elif action == 2: # change right
            if self.lane < 2:
                self.lane += 1
                self.lead_dist = np.random.uniform(50.0, 100.0)
                self.lead_speed = np.random.uniform(15.0, 30.0)
                lane_change_cost = -0.5
            else:
                lane_change_cost = -5.0
                
        # Speed dynamics
        if self.lead_dist < 15.0:
            # decelerate to match lead speed to avoid crash
            self.speed = min(self.speed, self.lead_speed)
            reward_speed = -5.0 # penalized for being blocked
        else:
            self.speed = min(30.0, self.speed + 1.0)  # accelerate up to speed limit
            reward_speed = self.speed * 0.1
            
        # Reward
        reward = reward_speed + lane_change_cost
        done = False
        
        if self.lead_dist < 2.0:
            reward = -50.0  # rear-end collision
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

class HighwayPPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_val = 0.2
        self.actor_opt = keras.optimizers.Adam(learning_rate=0.005)
        self.critic_opt = keras.optimizers.Adam(learning_rate=0.01)
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
        
    def act(self, state):
        probs = self.actor(state[np.newaxis, :]).numpy()[0]
        action = np.random.choice(self.action_dim, p=probs)
        return action, probs[action]
        
    def train(self, states, actions, old_probs, returns):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        old_probs = np.array(old_probs, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        # Critic optimization
        with tf.GradientTape() as tape_v:
            vals = tf.squeeze(self.critic(states))
            v_loss = keras.losses.MSE(returns, vals)
        grads_v = tape_v.gradient(v_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads_v, self.critic.trainable_variables))
        
        # Actor optimization
        with tf.GradientTape() as tape_p:
            vals = tf.squeeze(self.critic(states))
            advantages = returns - tf.stop_gradient(vals)
            if np.std(advantages) > 0:
                advantages = (advantages - np.mean(advantages)) / np.std(advantages)
                
            probs = self.actor(states)
            masks = tf.one_hot(actions, self.action_dim)
            new_probs = tf.reduce_sum(probs * masks, axis=1)
            
            ratio = new_probs / (old_probs + 1e-8)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_val, 1.0 + self.clip_val) * advantages
            
            p_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
        grads_p = tape_p.gradient(p_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads_p, self.actor.trainable_variables))

if __name__ == "__main__":
    env = HighwayLaneEnv()
    agent = HighwayPPOAgent(env.state_dim, env.action_dim)
    
    episodes = 25
    gamma = 0.95
    rewards_history = []
    
    print("Training Highway Lane-Changing PPO Agent...")
    for e in range(1, episodes + 1):
        states, actions, old_probs, rewards = [], [], [], []
        state = env.reset()
        done = False
        
        while not done:
            action, prob = agent.act(state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            old_probs.append(prob)
            rewards.append(reward)
            
            state = next_state
            
        total_r = sum(rewards)
        rewards_history.append(total_r)
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
            
        agent.train(states, actions, old_probs, returns)
        print(f"Episode: {e}/{episodes}, score: {total_r:.2f}, avg speed: {np.mean([s[1] for s in states]):.2f}")
        
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, color='blue', marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("PPO Highway Lane Changing Training curve")
    plt.grid(True)
    plt.savefig("exp_23_highway_lane_change_ppo.png")
    plt.close()
    
    print("Highway PPO training finished. Saved performance log to exp_23_highway_lane_change_ppo.png")
