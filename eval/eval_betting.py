import torch
import numpy as np
from agent.betting_agent import Betting_Agent
from agent.dqn_agent import DQN_Agent
from env.blackjackEnv import BlackjackEnv
from config import BET_SIZES, UNIT, INITIAL_BANKROLL, EVAL_HANDS, PLAYING_MODEL_PATH, BETTING_MODEL_PATH

def evaluate_betting_agent():
    # --- Load Environment ---
    env = BlackjackEnv()

    # --- Load Playing Agent (Pretrained) ---
    player_agent = DQN_Agent(epsilon_start=0.0)
    player_agent.q_network.load_state_dict(torch.load("checkpoints/1_million/blackjack_dqn_pt2.pth"))
    player_agent.q_network.eval()

    # --- Load Betting Agent (Trained) ---
    betting_agent = Betting_Agent(epsilon_start=0.0)
    betting_agent.q_network.load_state_dict(torch.load(BETTING_MODEL_PATH))
    betting_agent.q_network.eval()

    # --- Evaluation Variables ---
    bankroll = INITIAL_BANKROLL
    total_bet_amount = 0
    total_rewards = []
    bet_counts = {bet: 0 for bet in BET_SIZES}  # Track how often each bet is chosen

    # --- Simulate Hands ---
    for hand in range(1, EVAL_HANDS + 1):
        # 1. Observe current deck state
        deck_state = env.get_deck_distribution(betting=True)

        # 2. Betting agent chooses bet (index → amount)
        action_index = betting_agent.select_bet(deck_state)
        bet_units = BET_SIZES[action_index]
        bet = bet_units * UNIT
        bet_counts[bet_units] += 1

        # 3. Reset game and set bet
        state = env.reset()
        env.set_bet(bet)

        # 4. Play hand with playing agent
        done = False
        while not done:
            action = player_agent.select_action(state)
            next_state, _, done, info = env.step(action)
            state = next_state

        # 5. Update bankroll based on net result
        info = env.payout_tracker.get_info()
        net_profit = info["net_result"] 
        bankroll += net_profit
        total_bet_amount += bet
        total_rewards.append(net_profit)

        # 6. Print bankroll progress every 1000 hands
        if hand % 1000 == 0:
            avg_profit_per_hand = np.mean(total_rewards[-1000:])
            print(f"After {hand} hands: Bankroll=${bankroll:.2f}, "
                  f"Last 1000 Avg Profit=${avg_profit_per_hand:.2f}")

        # 7. Stop early if bankroll depletes
        if bankroll <= 0:
            print(f"Bankroll depleted after {hand} hands.")
            break

    # --- Final Stats ---
    total_profit = bankroll - INITIAL_BANKROLL
    roi = (total_profit / total_bet_amount) * 100 if total_bet_amount > 0 else 0
    avg_bet = total_bet_amount / len(total_rewards)

    print("\n=== Final Evaluation Results ===")
    print(f"Hands Played: {len(total_rewards)}")
    print(f"Final Bankroll: ${bankroll:.2f}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Average Bet: ${avg_bet:.2f}")
    print(f"ROI (Return on Investment): {roi:.2f}%")
    print(f"Bet Selection Frequency: {bet_counts}")

    avg_profit_overall = np.mean(total_rewards)
    print(f"Average Profit per Hand: ${avg_profit_overall:.2f}")

if __name__ == "__main__":
    evaluate_betting_agent()
