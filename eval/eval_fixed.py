import torch, numpy as np
from env.blackjackEnv import BlackjackEnv
from agent.dqn_agent import DQN_Agent

ACTION_NAMES = {0:"HIT", 1:"STAND", 2:"DOUBLE", 3:"SPLIT"}  # confirm matches env

env = BlackjackEnv()
agent = DQN_Agent(epsilon_start=0.0)     # greedy
CKPT = "checkpoints/iteration3/blackjack_dqn_ep500000.pth"
print(f"evaluating {CKPT}")
agent.q_network.load_state_dict(torch.load(CKPT, map_location="cpu"))
agent.q_network.eval()

num_episodes = 100_000
wins = losses = draws = 0
episode_returns = []      # final $ per episode
actions_count = {0:0, 1:0, 2:0, 3:0}

# Debug toggles
DISABLE_SPLIT_FOR_TEST = True   # set True to A/B SPLIT
track_split_legal = {"legal_yes":0, "legal_no":0, "chosen_when_legal":0}

total_hands = 0
max_steps_per_episode = 50

for _ in range(num_episodes):
    state = env.reset()
    env.set_bet(10)
    ep_return = 0.0
    steps = 0

    while steps < max_steps_per_episode and not getattr(env, "done", False):
        steps += 1
        legal = env.legal_actions()

        # SPLIT A/B test (optional)
        if DISABLE_SPLIT_FOR_TEST and 3 in legal:
            legal = [a for a in legal if a != 3]

        # track split legality
        if 3 in legal: track_split_legal["legal_yes"] += 1
        else:          track_split_legal["legal_no"]  += 1

        # masked greedy
        action = agent.select_action(state, legal_actions=legal)
        actions_count[action] += 1
        if action == 3 and 3 in legal:
            track_split_legal["chosen_when_legal"] += 1

        next_state, reward, done, msg = env.step(action)
        ep_return += reward
        state = next_state

        if done: break

    # Use the environment’s final accounting for $ result and hands played
    info = env.payout_tracker.get_info()
    final_dollars = info.get("net_result", ep_return)
    hands_this_ep = info.get("hands_played", 1)  # fallback if not provided

    episode_returns.append(final_dollars)
    total_hands += hands_this_ep

    # decide win/loss/draw from final dollars (not last step reward)
    if final_dollars > 0:   wins  += 1
    elif final_dollars < 0: losses+= 1
    else:                   draws += 1

# === Results ===
print("\n=== Evaluation Results ===")
print(f"Episodes Played:  {num_episodes}")
print(f"Wins:             {wins}")
print(f"Losses:           {losses}")
print(f"Draws:            {draws}")
print(f"Win Rate:         {wins / num_episodes:.2%}")
print(f"Loss Rate:         {losses / num_episodes:.2%}")
print(f"Draw Rate:         {draws / num_episodes:.2%}")

avg_ep = float(np.mean(episode_returns))      # $ per episode
avg_per_hand = avg_ep if total_hands == 0 else (sum(episode_returns) / total_hands)
print(f"Average $ per Episode: {avg_ep:.3f}")
print(f"Average $ per Hand:    {avg_per_hand:.3f}")

print("Actions Count:", {ACTION_NAMES[k]: v for k,v in actions_count.items()})
if track_split_legal["legal_yes"] + track_split_legal["legal_no"] > 0:
    rate_legal = track_split_legal["legal_yes"] / (track_split_legal["legal_yes"] + track_split_legal["legal_no"])
    rate_taken = (track_split_legal["chosen_when_legal"] / track_split_legal["legal_yes"]) if track_split_legal["legal_yes"] else 0.0
    print(f"SPLIT legal fraction: {rate_legal:.2%}")
    print(f"SPLIT chosen when legal: {rate_taken:.2%}")
