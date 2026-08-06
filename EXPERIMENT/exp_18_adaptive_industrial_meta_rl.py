"""
Experiment 18: Meta-Reinforcement Learning for Adaptive Industrial Robot
Objective: Develop a Meta-RL model (Context-Based) that adapts a robotic gripper to varying payload friction weights.
State: [gripper_position, payload_mass, applied_force]
Method: Context-Based Meta-RL with Policy Gradient
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class IndustrialGripperEnv:
    def __init__(self):
        self.state_dim = 3  # [position, mass, velocity]
        self.action_dim = 2  # 0: Apply Low Force, 1: Apply High Force
        self.max_steps = 30
        
    def sample_task(self):
        # The meta-task is defined by the weight/mass of the object
        # which is unknown to the agent at start but must be inferred from context
        self.mass = np.random.uniform(0.5, 4.0)
        self.pos = 0.0
        self.vel = 0.0
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.pos, self.vel, 0.0], dtype=np.float32)  # mass is hidden/partially observable
        
    def step(self, action):
        self.steps += 1
        
        # Apply force
        force = 4.0 if action == 1 else 1.0
        
        # Physics: acceleration = force / mass
        acc = force / self.mass
        self.vel += acc * 0.1
        self.pos += self.vel * 0.1
        
        # Reward is negative distance to target position 1.0
        target = 1.0
        dist = abs(self.pos - target)
        reward = -dist
        done = False
        
        if dist < 0.1:
            reward = 50.0  # Successfully held
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

class MetaRLAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Context model learns from previous trajectory transitions to estimate a context vector
        # Policy model takes state and estimated context vector
        self.policy = self._build_meta_policy()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.005)
        
    def _build_meta_policy(self):
        # Input layer takes state (3) + context representation (2) = 5 features
        inputs = layers.Input(shape=(self.state_dim + 2,))
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='softmax')(x)
        return Model(inputs=inputs, outputs=outputs)
        
    def get_context(self, history):
        """Processes previous steps to infer friction/mass context"""
        if len(history) == 0:
            return np.array([0.0, 0.0]) # default starting context
        # Simple heuristic context encoder: mean of previous actions and rewards
        mean_act = np.mean([x[1] for x in history])
        mean_rew = np.mean([x[2] for x in history])
        return np.array([mean_act, mean_rew])
        
    def get_action(self, state, context):
        inputs = np.concatenate([state, context])
        probs = self.policy(inputs[np.newaxis, :]).numpy()[0]
        return np.random.choice(self.action_dim, p=probs)
        
    def train(self, trajectories):
        # trajectories: list of (states, actions, contexts, returns)
        all_inputs = []
        all_actions = []
        all_returns = []
        
        for states, actions, contexts, returns in trajectories:
            for s, a, c, r in zip(states, actions, contexts, returns):
                all_inputs.append(np.concatenate([s, c]))
                all_actions.append(a)
                all_returns.append(r)
                
        all_inputs = np.array(all_inputs, dtype=np.float32)
        all_actions = np.array(all_actions, dtype=np.int32)
        all_returns = np.array(all_returns, dtype=np.float32)
        
        if len(all_returns) > 1 and np.std(all_returns) > 0:
            all_returns = (all_returns - np.mean(all_returns)) / np.std(all_returns)
            
        with tf.GradientTape() as tape:
            probs = self.policy(all_inputs)
            masks = tf.one_hot(all_actions, self.action_dim)
            chosen_probs = tf.reduce_sum(probs * masks, axis=1)
            loss = -tf.reduce_sum(tf.math.log(chosen_probs + 1e-8) * all_returns)
            
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

if __name__ == "__main__":
    env = IndustrialGripperEnv()
    agent = MetaRLAgent(env.state_dim, env.action_dim)
    
    meta_iterations = 20
    tasks_per_iter = 4
    gamma = 0.95
    adaptation_history = []
    
    print("Training Adaptive Meta-RL Agent...")
    for iter_idx in range(meta_iterations):
        trajectories = []
        iter_rewards = []
        
        for task_idx in range(tasks_per_iter):
            state = env.sample_task()
            done = False
            history = []
            
            states, actions, contexts, rewards = [], [], [], []
            
            while not done:
                context = agent.get_context(history)
                action = agent.get_action(state, context)
                next_state, reward, done = env.step(action)
                
                states.append(state)
                actions.append(action)
                contexts.append(context)
                rewards.append(reward)
                
                history.append((state, action, reward, next_state))
                state = next_state
                
            total_r = sum(rewards)
            iter_rewards.append(total_r)
            
            # Calculate returns G
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
                
            trajectories.append((states, actions, contexts, returns))
            
        agent.train(trajectories)
        avg_rew = np.mean(iter_rewards)
        adaptation_history.append(avg_rew)
        print(f"Meta Iteration: {iter_idx + 1}/{meta_iterations}, Avg Task Reward: {avg_rew:.2f}")
        
    plt.figure(figsize=(8, 4))
    plt.plot(adaptation_history, color='navy', marker='o')
    plt.xlabel("Meta-Training Iteration")
    plt.ylabel("Average Reward across Gripper Tasks")
    plt.title("Meta-RL Task Adaptation Learning Curve")
    plt.grid(True)
    plt.savefig("exp_18_adaptive_industrial_meta_rl.png")
    plt.close()
    
    print("Meta-RL script completed. Saved plot to exp_18_adaptive_industrial_meta_rl.png")
