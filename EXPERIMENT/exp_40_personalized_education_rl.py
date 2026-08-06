"""
Experiment 40: Reinforcement Learning for Personalized Education
Objective: Develop a tutoring system that sequences lessons and quizzes dynamically based on student knowledge scores.
State: [knowledge_A, knowledge_B, knowledge_C] (each can be 0: Novice, 1: Intermediate, 2: Master)
Actions: 0: Teach A, 1: Teach B, 2: Teach C, 3: Conduct Quiz
Method: Q-Learning Adaptive Tutoring
"""

import numpy as np
import matplotlib.pyplot as plt

class StudentTutoringEnv:
    def __init__(self):
        # 3 Topics: A, B, C. Values: 0, 1, 2.
        # State space size = 3 * 3 * 3 = 27 states
        self.n_states = 27
        self.n_actions = 4  # 0: Teach A, 1: Teach B, 2: Teach C, 3: Quiz
        self.reset()
        
    def reset(self):
        self.knowledge = np.array([0, 0, 0]) # novice in all topics
        return self._get_obs()
        
    def _get_obs(self):
        return (self.knowledge[0] * 9) + (self.knowledge[1] * 3) + self.knowledge[2]
        
    def step(self, action):
        reward = 0
        done = False
        
        if action in [0, 1, 2]:
            topic = action
            # Teach Lesson: has a probability of increasing student knowledge level
            if self.knowledge[topic] < 2:
                prob = 0.6 if self.knowledge[topic] == 0 else 0.45
                if np.random.rand() < prob:
                    self.knowledge[topic] += 1
                    reward = 5.0
                else:
                    reward = 0.5  # positive effort reward
            else:
                reward = -5.0  # teaching mastered topic causes boredom
                
        elif action == 3:
            # Conduct Quiz: tests the student and consolidates knowledge
            avg_knowledge = np.mean(self.knowledge)
            reward = float(avg_knowledge * 10.0) # reward based on score
            
        # Completion check: mastered all subjects
        if np.all(self.knowledge == 2):
            reward += 100.0
            done = True
            
        return self._get_obs(), reward, done

def train_tutoring_system(env, episodes=1000, lr=0.1, gamma=0.95, epsilon=0.15):
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        steps = 0
        
        while not done and steps < 40:
            steps += 1
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])
                
            next_state, reward, done = env.step(action)
            
            # Q-update
            best_next = np.argmax(Q[next_state])
            Q[state, action] += lr * (reward + gamma * Q[next_state, best_next] - Q[state, action])
            
            state = next_state
            total_r += reward
            
        rewards_history.append(total_r)
        
    return Q, rewards_history

if __name__ == "__main__":
    env = StudentTutoringEnv()
    Q, history = train_tutoring_system(env, episodes=800)
    
    # Smooth history
    window = 30
    smooth_history = np.convolve(history, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(8, 4))
    plt.plot(smooth_history, color='dodgerblue')
    plt.xlabel("Episode")
    plt.ylabel("Tutor Performance Score")
    plt.title("Adaptive Tutoring RL System Training Progression")
    plt.grid(True)
    plt.savefig("exp_40_personalized_education_rl.png")
    plt.close()
    
    print("Tutoring tutoring script finished. Saved chart to exp_40_personalized_education_rl.png")
