import torch
from env.blackjackEnv import BlackjackEnv
from agent.dqn_agent import DQN_Agent
import numpy as np

env = BlackjackEnv()
agent = DQN_Agent(epsilon_start=0.0)

CKPT = "checkpoints/count_aware/blackjack_dqn_ep350000.pth"
checkpoint_path = "checkpoints/1_million/blackjack_dqn_pt2.pth"
checkpoint_path = CKPT
print(f"evaluating {checkpoint_path}")
agent.q_network.load_state_dict(torch.load(checkpoint_path))
agent.q_network.eval()

total_rewards = []

num_episodes = 100_000
wins = 0
losses = 0
draws = 0
total_reward = 0
max_steps_per_episode = 50
steps = 0
hands= 0 
actions_count = {0: 0, 1:0 , 2:0, 3:0}

for episode in range(num_episodes):
    state = env.reset()
    steps = 0
    env.set_bet(10)
    previous_hand_index = env.current_hand_index

    while not env.done and steps < max_steps_per_episode:
        action = agent.select_action(state)
        legal = env.legal_actions()
        a = agent.select_action(state, legal_actions=legal)
        if a not in legal:
            print("HEEEEEEEEEEEELOOOOOO \n\n\n\n")
        assert a in legal, f"Picked illegal {a} with legal={legal}, state[:6]={state}"
        actions_count[action] += 1  
        next_state, reward, done, msg = env.step(action)
        state = next_state
        if env.current_hand_index != previous_hand_index:
            previous_hand_index = env.current_hand_index
            hands += 1
    
    total_reward += reward
    info = env.payout_tracker.get_info()
    raw_reward = info["net_result"]
    total_rewards.append(raw_reward)
    
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
print(f"Loss Rate:         {losses / num_episodes:.2%}")
print(f"Draw Rate:         {draws / num_episodes:.2%}")
print(f"Average Reward:   {total_reward / num_episodes:.2f}")
print(f"Actions Count: {actions_count}")

avg_loss_per_episode = np.mean(total_rewards)
avg_hands_per_episode = hands / num_episodes
avg_loss_per_hand = avg_loss_per_episode / avg_hands_per_episode
print(f"Average Loss Per Hand: {avg_loss_per_hand}")

if __name__ == "__main__":
    if len(total_rewards) > 0:
        print(f"Final Training Rewards:")
        print(f"  Mean: {np.mean(total_rewards):.4f}")
        print(f"  Std Dev: {np.std(total_rewards):.4f}")
        print(f"  Min: {np.min(total_rewards):.4f}")
        print(f"  Max: {np.max(total_rewards):.4f}")
        
        # Count positive vs negative
        positive = sum(1 for r in total_rewards if r > 0)
        negative = sum(1 for r in total_rewards if r < 0)
        zero = sum(1 for r in total_rewards if r == 0)
        
        print(f"  Positive rewards: {positive} ({positive/len(total_rewards)*100:.1f}%)")
        print(f"  Negative rewards: {negative} ({negative/len(total_rewards)*100:.1f}%)")
        print(f"  Zero rewards: {zero} ({zero/len(total_rewards)*100:.1f}%)")

