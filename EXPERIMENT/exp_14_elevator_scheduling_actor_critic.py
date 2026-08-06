"""
Experiment 14: Actor-Critic (A2C) for Smart Elevator Scheduling
Objective: Train a scheduling policy for a multi-floor elevator to minimize passenger wait times.
State: [elevator_floor, num_passengers_inside, waiting_f0, waiting_f1, waiting_f2, waiting_f3, waiting_f4]
Actions: 0: Go Up, 1: Go Down, 2: Open Door (Serve)
Method: Advantage Actor-Critic (A2C) using TensorFlow/Keras
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class ElevatorEnv:
    def __init__(self, num_floors=5):
        self.num_floors = num_floors
        self.state_dim = 2 + num_floors  # floor, passengers inside, queues for each floor
        self.action_dim = 3  # 0: UP, 1: DOWN, 2: SERVE (open door)
        self.max_steps = 60
        self.reset()
        
    def reset(self):
        self.floor = 0
        self.inside = 0
        self.queues = np.array([2.0, 1.0, 3.0, 0.0, 2.0], dtype=np.float32)
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.concatenate([[self.floor, self.inside], self.queues], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        
        # New passengers arrive
        arrivals = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1], size=self.num_floors)
        self.queues += arrivals
        
        step_cost = -1.0 # time step penalty
        
        if action == 0:  # UP
            if self.floor < self.num_floors - 1:
                self.floor += 1
            else:
                step_cost -= 5.0  # Boundary penalty
        elif action == 1:  # DOWN
            if self.floor > 0:
                self.floor -= 1
            else:
                step_cost -= 5.0  # Boundary penalty
        elif action == 2:  # SERVE
            # Pick up all waiting passengers at current floor
            picked = self.queues[self.floor]
            if picked > 0:
                self.inside += picked
                self.queues[self.floor] = 0
                step_cost += 10.0 * picked  # Reward for servicing floor
            # Simulate drops randomly
            dropped = np.random.randint(0, int(self.inside) + 1)
            self.inside -= dropped
            step_cost += 15.0 * dropped # Reward for dropping passengers off
            
        # Waiting queue penalties
        total_waiting = np.sum(self.queues)
        reward = step_cost - (2.0 * total_waiting) - (0.5 * self.inside)
        
        done = self.steps >= self.max_steps
        return self._get_obs(), reward, done

class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.95
        
        # Network
        self.actor, self.critic = self._build_networks()
        self.actor_opt = keras.optimizers.Adam(learning_rate=0.005)
        self.critic_opt = keras.optimizers.Adam(learning_rate=0.01)
        
    def _build_networks(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        
        # Dual outputs
        actor_output = layers.Dense(self.action_dim, activation='softmax')(x)
        critic_output = layers.Dense(1, activation='linear')(x)
        
        actor = Model(inputs=inputs, outputs=actor_output)
        critic = Model(inputs=inputs, outputs=critic_output)
        return actor, critic
        
    def get_action(self, state):
        probs = self.actor(state[np.newaxis, :]).numpy()[0]
        return np.random.choice(self.action_dim, p=probs)
        
    def train(self, state, action, reward, next_state, done):
        state_t = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        next_state_t = tf.convert_to_tensor(next_state[np.newaxis, :], dtype=tf.float32)
        
        with tf.GradientTape() as tape_act, tf.GradientTape() as tape_crit:
            val_s = self.critic(state_t)[0][0]
            val_ns = self.critic(next_state_t)[0][0]
            
            # TD Target & Advantage
            target = reward if done else reward + self.gamma * val_ns
            td_error = target - val_s
            
            # Critic loss (MSE)
            critic_loss = tf.math.square(td_error)
            
            # Actor loss -log(pi(a|s))*Advantage
            probs = self.actor(state_t)
            action_prob = probs[0][action]
            actor_loss = -tf.math.log(action_prob + 1e-8) * tf.stop_gradient(td_error)
            
        grads_act = tape_act.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads_act, self.actor.trainable_variables))
        
        grads_crit = tape_crit.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads_crit, self.critic.trainable_variables))

if __name__ == "__main__":
    env = ElevatorEnv()
    agent = A2CAgent(env.state_dim, env.action_dim)
    
    episodes = 25
    rewards_history = []
    
    print("Training Smart Elevator Agent with A2C...")
    for e in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        print(f"Episode: {e}/{episodes}, Total Reward: {total_reward:.2f}")
        
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, color='teal', marker='s')
    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward")
    plt.title("A2C Smart Elevator Scheduling Training Curve")
    plt.grid(True)
    plt.savefig("exp_14_elevator_scheduling_actor_critic.png")
    plt.close()
    
    print("Elevator schedule model saved to exp_14_elevator_scheduling_actor_critic.png")
