"""
Experiment 24: Automated Trading via REINFORCE Policy Gradient
Objective: Develop an automated trading system that maximizes profit while managing risk using REINFORCE.
State: [current_price, price_change_rate, holding_position]
Actions: 0: Hold, 1: Buy, 2: Sell
Method: REINFORCE Policy Gradient using TensorFlow/Keras
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class TradingEnv:
    def __init__(self):
        self.state_dim = 3
        self.action_dim = 3  # 0: Hold, 1: Buy, 2: Sell
        # Generate synthetic price series (Sine wave + random noise + upward drift)
        steps = 100
        t = np.linspace(0, 10, steps)
        self.prices = 100.0 + 15.0 * np.sin(t) + np.random.normal(0, 1.0, steps) + 0.5 * t
        self.max_steps = len(self.prices) - 2
        self.reset()
        
    def reset(self):
        self.curr_step = 0
        self.holding = 0.0  # 0: None, 1: Holding stock
        self.buy_price = 0.0
        self.capital = 1000.0
        return self._get_obs()
        
    def _get_obs(self):
        price = self.prices[self.curr_step]
        prev_price = self.prices[max(0, self.curr_step - 1)]
        change = (price - prev_price) / prev_price
        return np.array([price / 150.0, change, float(self.holding)], dtype=np.float32) # normalize price
        
    def step(self, action):
        self.curr_step += 1
        price = self.prices[self.curr_step]
        reward = 0.0
        
        transaction_cost = 0.5
        
        if action == 1:   # BUY
            if self.holding == 0.0:
                self.holding = 1.0
                self.buy_price = price
                reward = -transaction_cost
            else:
                reward = -2.0  # penalty for buying when already holding
        elif action == 2: # SELL
            if self.holding == 1.0:
                self.holding = 0.0
                profit = price - self.buy_price
                self.capital += profit - transaction_cost
                
                # Risk management: reward profit, penalize loss heavily (risk aversion)
                if profit > 0:
                    reward = profit * 1.5
                else:
                    reward = profit * 3.0  # heavy loss aversion
                reward -= transaction_cost
            else:
                reward = -2.0  # penalty for selling when empty
        else: # HOLD (0)
            if self.holding == 1.0:
                # Unrealized daily change reward/penalty
                day_change = price - self.prices[self.curr_step - 1]
                reward = day_change * 0.1
                
        done = self.curr_step >= self.max_steps
        return self._get_obs(), reward, done

class ReinforceTrader:
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
    env = TradingEnv()
    agent = ReinforceTrader(env.state_dim, env.action_dim)
    
    episodes = 25
    gamma = 0.98
    rewards_history = []
    capital_history = []
    
    print("Training REINFORCE Automated Trader...")
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
        capital_history.append(env.capital)
        
        # Calculate returns G
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
            
        agent.train(states, actions, returns)
        print(f"Episode: {e}/{episodes}, Return: {total_r:.2f}, Final Capital: ${env.capital:.2f}")
        
    plt.figure(figsize=(10, 4))
    plt.plot(capital_history, color='green', marker='^', label='Final Capital')
    plt.axhline(y=1000.0, color='red', linestyle='--', label='Initial Capital ($1000)')
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value ($)")
    plt.title("REINFORCE Trader: Capital Growth Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_24_automated_trading_reinforce.png")
    plt.close()
    
    print("Trading simulation finished. Saved capital growth graph to exp_24_automated_trading_reinforce.png")
