import numpy as np
from env.blackjackEnv import BlackjackEnv
from agent.dqn_agent import DQN_Agent
import torch
import os
from utils.shaping import shaping_bonus
from eval.ev_by_true_count import run_once as ev_by_TC

os.makedirs("checkpoints/count_weighted", exist_ok=True)

num_episodes = 100_000    #this is where results are plateuing
epsilon_start = 0.20
epsilon_min = 0.05
epsilon_decay = 0.9999      #
gamma = 0.95
learning_rate = 1e-4
buffer_capacity = 150_000
batch_size = 32
max_steps_per_episode = 100
PRINT_FREQ = 10_000          # Print stats every X episodes

env = BlackjackEnv()
CKPT_FREQ = 50_000

agent = DQN_Agent(epsilon_start=epsilon_start,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    gamma=gamma,
    lr=learning_rate,
    buffer_capacity=buffer_capacity,
    batch_size=batch_size)

total_reward = []
ckpt = "checkpoints/count_aware/blackjack_dqn_ep400000.pth"
agent.q_network.load_state_dict(torch.load(ckpt, map_location="cpu"))

for episode in range(num_episodes):
    state = env.reset()
    env.set_bet(10)

    for step in range(max_steps_per_episode):
        legal = env.legal_actions()
        if step == 0 and env.dealer_hand == set(env.BLACKJACK):
            next_state, reward, done, msg =  env.compare_hands()
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()  
            break
        else:
            action = agent.select_action(state, legal_actions=legal)
            next_state, reward, done, msg = env.step(action)
            if step > 3:
                reward -= 0.2

            agent.store_experience(state, action, reward, next_state, done)
        
            #if len(agent.replay_buffer) >= 10_000 and len(agent.replay_buffer) < 20_000:
                #for _ in range(2):
                    #agent.train()
            #elif len(agent.replay_buffer) >= 20_000:
                #for _ in range(3):
                    #agent.train()
            if len(agent.replay_buffer) > 1500:
                agent.train()  # Only 1 step until buffer fills

            state = next_state

            if done:
                break
    
    total_reward.append(reward)

    agent.decay_eps_episode()
    
    if (episode+1) % PRINT_FREQ == 0 and episode > 0:
        avg_recent_reward = np.mean(total_reward[-PRINT_FREQ:])
        print(f"Episode {episode+1}:, Average Reward = {avg_recent_reward:.3f}, Epsilon = {agent.epsilon:.3f}")
    
    if (episode+1) % CKPT_FREQ == 0 and episode > 0:
        path = f"checkpoints/count_weighted/blackjack_dqn_ep{episode+1:06d}.pth"
        torch.save(agent.q_network.state_dict(), path)
        print(f"\n=== EVAL after {episode+1} episodes ===")
        ev_by_TC(n_hands=200_000, ckpt=path)  # fixed $10, pre-deal TC, ε=0
        agent.replay_buffer.composition()


#torch.save(agent.q_network.state_dict(), "checkpoints/1_million/blackjack_dqn_pt3.pth")
#print("✅ Model saved to checkpoints/1/million/blackjack_dqn_pt3.pth")
agent.replay_buffer.composition()