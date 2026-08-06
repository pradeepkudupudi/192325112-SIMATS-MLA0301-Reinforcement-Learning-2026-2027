"""
Experiment 34: HVAC Temperature Adjuster via REINFORCE
Objective: Train an HVAC temperature controller to balance user comfort and energy costs using REINFORCE.
State: [outdoor_temp, indoor_temp, target_temp, electricity_price]
Actions: 0: Cool (AC), 1: Heat, 2: Idle
Method: REINFORCE Policy Gradient using TensorFlow/Keras
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class HvacEnv:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 3 # AC, Heat, Idle
        self.max_steps = 24  # hourly control for a day
        self.reset()
        
    def reset(self):
        self.outdoor_temp = 32.0  # Hot day
        self.indoor_temp = 26.0
        self.target_temp = 22.0
        self.electricity_price = 0.15 # $/kWh
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.array([self.outdoor_temp/40.0, self.indoor_temp/40.0, self.target_temp/40.0, self.electricity_price], dtype=np.float32)
        
    def step(self, action):
        self.steps += 1
        
        # Outdoor temp varies stochastically over the day
        self.outdoor_temp += np.random.uniform(-0.5, 0.5)
        
        # Room heat transfer dynamics (towards outdoor temperature)
        self.indoor_temp += 0.1 * (self.outdoor_temp - self.indoor_temp)
        
        power_consumed = 0.0
        # HVAC system actions
        if action == 0:    # AC cooling
            self.indoor_temp -= 1.5
            power_consumed = 2.0  # kW
        elif action == 1:  # Heating
            self.indoor_temp += 1.5
            power_consumed = 2.5  # kW
        # Idle (2) consumes 0 power
        
        # Comfort penalty (deviation from setpoint 22C)
        comfort_loss = -1.5 * ((self.indoor_temp - self.target_temp) ** 2)
        
        # Cost of energy
        energy_cost = - power_consumed * self.electricity_price
        
        reward = comfort_loss + energy_cost
        done = self.steps >= self.max_steps
        
        return self._get_obs(), reward, done

class HvacAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.optimizer = keras.optimizers.Adam(learning_rate=0.005)
        self.model = self._build_model()
        
    def _build_model(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='softmax')(x)
        return Model(inputs=inputs, outputs=outputs)
        
    def act(self, state):
        probs = self.model(state[np.newaxis, :]).numpy()[0]
        return np.random.choice(self.action_dim, p=probs)
        
    def train(self, states, actions, returns):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        
        if np.std(returns) > 0:
            returns = (returns - np.mean(returns)) / np.std(returns)
            
        with tf.GradientTape() as tape:
            probs = self.model(states)
            masks = tf.one_hot(actions, self.action_dim)
            chosen_probs = tf.reduce_sum(probs * masks, axis=1)
            loss = -tf.reduce_sum(tf.math.log(chosen_probs + 1e-8) * returns)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

if __name__ == "__main__":
    env = HvacEnv()
    agent = HvacAgent(env.state_dim, env.action_dim)
    
    episodes = 30
    gamma = 0.95
    rewards_history = []
    
    print("Training HVAC controller with REINFORCE policy gradient...")
    for e in range(1, episodes + 1):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            
        total_r = sum(rewards)
        rewards_history.append(total_r)
        
        # Calculate returns G
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
            
        agent.train(states, actions, returns)
        print(f"Episode: {e}/{episodes}, reward: {total_r:.2f}, final temp: {env.indoor_temp:.2f}C")
        
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, color='crimson', marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Comfort & Energy Score")
    plt.title("REINFORCE HVAC Smart Temperature Control Training")
    plt.grid(True)
    plt.savefig("exp_34_hvac_temperature_reinforce.png")
    plt.close()
    
    print("HVAC thermostat training completed. Saved plot to exp_34_hvac_temperature_reinforce.png")
stream = None
