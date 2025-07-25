import torch
import numpy as np
from agent.betting_agent import Betting_Agent
from agent.dqn_agent import DQN_Agent
from env.blackjackEnv import BlackjackEnv
from config import BET_SIZES, INITIAL_BANKROLL  # We'll create this config later

NUM_EPISODES = 50_000   # Total rounds to simulate
TARGET_UPDATE_FREQ = 1000  # How often to sync target network
PRINT_FREQ = 1000          # Print stats every X episodes

def train_betting_agent():
    env= BlackjackEnv()

    player_agent = DQN_Agent.load("checkpoints/blackjack_dqn.pth")
    player_agent.q_network.eval()

    # Initialize betting agent
    bet_agent = Betting_Agent()
    total_rewards = []

    for episode in range(NUM_EPISODES):
        deck_state = env.get_deck_distribution()

        bet_action_index = bet_agent.select_bet(deck_state)
        bet = BET_SIZES[bet_action_index]

        state = env.reset()
        env.set_bet(bet)

        done = False
        while not done:
            action = player_agent.select_action(state)
            next_state, _, done, info = env.step(action)
            state = next_state
        
        reward = info["net_result"]  # Profit/loss for chosen bet
        next_deck_state = env.get_deck_distribution()

        # 6. Store Experience
        bet_agent.store_experience(deck_state, bet_action_index, reward, next_deck_state, done=False)

        # 7. Train Betting Agent
        bet_agent.train_step()
        total_rewards.append(reward)

         # 8. Update Target Network Periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            bet_agent.target_network.load_state_dict(bet_agent.q_network.state_dict())

        # 9. Print Progress
        if episode % PRINT_FREQ == 0 and episode > 0:
            avg_reward = np.mean(total_rewards[-PRINT_FREQ:])
            print(f"Episode {episode}/{NUM_EPISODES}, Avg Reward: {avg_reward:.3f}, Epsilon: {bet_agent.epsilon:.3f}")

    # Save the trained betting agent
    torch.save(bet_agent.q_network.state_dict(), "checkpoints/betting/bet_dqn.pth")
    print("Training complete.")

if __name__ == "__main__":
    train_betting_agent()