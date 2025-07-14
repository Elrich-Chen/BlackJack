import torch
from env.blackjackEnv import BlackjackEnv
from agent.dqn_agent import DQN_Agent

env = BlackjackEnv()
agent = DQN_Agent(epsilon_start=0.0)

checkpoint_path = "checkpoints/blackjack_dqn.pth"
agent.q_network.load_state_dict(torch.load(checkpoint_path))
agent.q_network.eval()

num_episodes = 10_000
wins = 0
losses = 0
draws = 0
total_reward = 0
actions_count = {0: 0, 1:0 , 2:0, 3:0}

for episode in range(num_episodes):
    state = env.reset()

    while not env.done:
        action = agent.select_action(state)
        actions_count[action] += 1  
        next_state, reward, done, msg = env.step(action)
        state = next_state
    
    total_reward += reward
    if reward >0:
        wins += 1
    elif reward < 0:
        losses+=1
    else:
        draws += 1

# === Results ===
print("\n=== Evaluation Results ===")
print(f"Episodes Played:  {num_episodes}")
print(f"Wins:             {wins}")
print(f"Losses:           {losses}")
print(f"Draws:            {draws}")
print(f"Win Rate:         {wins / num_episodes:.2%}")
print(f"Average Reward:   {total_reward / num_episodes:.2f}")
print(f"Actions Count: {actions_count}")
