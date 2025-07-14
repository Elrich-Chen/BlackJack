import numpy as np
from env.blackjackEnv import BlackjackEnv
from agent.dqn_agent import DQN_Agent
import torch
import os

os.makedirs("checkpoints", exist_ok=True)

num_episodes = 50_000
epsilon_start = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
gamma = 0.99
learning_rate = 0.001
buffer_capacity = 10_000
batch_size = 32
max_steps_per_episode = 100

env = BlackjackEnv()

agent = DQN_Agent(epsilon_start=epsilon_start,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    gamma=gamma,
    lr=learning_rate,
    buffer_capacity=buffer_capacity,
    batch_size=batch_size)

for episode in range(num_episodes):
    total_reward = 0
    state = env.reset()

    for step in range(max_steps_per_episode):
        action = agent.select_action(state)
        next_state, reward, done, msg = env.step(action)

        agent.store_experience(state, action, reward, next_state, done)

        agent.train()

        state = next_state
        total_reward += reward

        if done:
            break
    
    if (episode+1 % 100 == 0):
         print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

torch.save(agent.q_network.state_dict(), "checkpoints/blackjack_dqn.pth")
print("✅ Model saved to checkpoints/blackjack_dqn.pth")