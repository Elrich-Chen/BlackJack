import torch
import numpy as np
from agent.betting_agent import Betting_Agent
from agent.dqn_agent import DQN_Agent
from env.blackjackEnv import BlackjackEnv
from config import BET_SIZES, UNIT, INITIAL_BANKROLL
from env.highTCWrapper import HighTCWrapper as TC

NUM_EPISODES = 300_000   # Total rounds to simulate
TARGET_UPDATE_FREQ = 1000  # How often to sync target network
PRINT_FREQ = 10_000          # Print stats every X episodes

def train_betting_agent():
    env = BlackjackEnv()

    player_agent = DQN_Agent()
    player_agent.q_network.load_state_dict(torch.load("checkpoints/blackjack_dqn.pth"))
    player_agent.q_network.eval()

    # Initialize betting agent with VERY conservative settings
    bet_agent = Betting_Agent(
        epsilon_decay=0.99995,  # Slower exploration decay
        gamma=0.95,           # Higher discount factor (more long-term thinking)
        lr=0.0005,           # Slower learning rate
        epsilon_start=1.0,   # Start with less exploration
        epsilon_min=0.01,     # Very low final exploration
        buffer_capacity= 30_000 
    )
    total_rewards = []
    raw_rewards = []

    # Track betting by count
    count_bet_stats = {}

    for episode in range(NUM_EPISODES):
        deck_state = env.get_deck_distribution(betting=True)
        bet_action_index = bet_agent.select_bet(deck_state)
        
        # Simple unit betting - no Kelly Criterion for now
        bet_dollars = BET_SIZES[bet_action_index] * UNIT
        
        state = env.reset()
        env.set_bet(bet_dollars)

        # Play the hand
        done = False
        while not done:
            action = player_agent.select_action(state)
            next_state, reward, done, msg = env.step(action)
            state = next_state
        
        # Get game results
        info = env.payout_tracker.get_info()
        raw_reward = info["net_result"]
        raw_rewards.append(raw_reward)
        
        # Get true count for training reward
        true_count = deck_state[11]

        # EXTREMELY CAUTIOUS reward structure for positive EV
        training_reward = 0

        if true_count >= 4:  # Only bet big on VERY high counts
            if bet_action_index == 4:  # Only 16 units on extremely high counts
                training_reward = +2.0  # Excellent decision
            elif bet_action_index == 3:  # 8 units acceptable
                training_reward = +1.0  # Good decision
            elif bet_action_index <= 2:  # Still betting small on great count
                training_reward = -1.0  # Major missed opportunity
            else:
                training_reward = +0.5  # Medium bet okay
                
        elif true_count >= 3:  # High count - medium to big bets
            if bet_action_index >= 3:  # 8 or 16 units
                training_reward = +1.5  # Good decision
            elif bet_action_index == 2:  # 4 units acceptable
                training_reward = +0.5  # Okay decision
            else:
                training_reward = -0.5  # Too small for this count
                
        elif true_count >= 1:  # Slightly favorable - small to medium bets
            if bet_action_index <= 2:  # 1, 2, or 4 units
                training_reward = +1.0  # Safe decision
            else:
                training_reward = -1.5  # Too aggressive
                
        elif true_count >= -1:  # Neutral - only small bets
            if bet_action_index <= 1:  # 1 or 2 units only
                training_reward = +1.0  # Safe decision
            elif bet_action_index == 2:  # 4 units borderline
                training_reward = +0.0  # Neutral
            else:
                training_reward = -2.0  # Way too aggressive
                
        else:  # true_count < -1 (Unfavorable) - minimum bets only
            if bet_action_index == 0:  # 1 unit only
                training_reward = +2.0  # Excellent caution
            elif bet_action_index == 1:  # 2 units acceptable
                training_reward = +0.5  # Okay
            else:
                training_reward = -3.0  # Terrible decision

        total_rewards.append(training_reward)

        # Debug output every 10,000 episodes
        if episode % PRINT_FREQ == 0 and episode > 0:
            print(f"  Detailed Debug:")
            print(f"    Bet: {BET_SIZES[bet_action_index]} units (${bet_dollars}) - Action Index: {bet_action_index}")
            print(f"    True Count: {true_count:.1f}")
            print(f"    Raw Reward: ${raw_reward}")
            print(f"    Training Reward: {training_reward:.2f}")

        next_deck_state = env.get_deck_distribution(betting=True)

        # Store experience using training reward
        bet_agent.store_experience(deck_state, bet_action_index, training_reward, next_deck_state, done)

        if len(bet_agent.replay_buffer) >= 10_000 and len(bet_agent.replay_buffer) < 20_000:
            for _ in range(2):
                bet_agent.train()
        elif len(bet_agent.replay_buffer) >= 20_000:
            for _ in range(3):
                bet_agent.train()
        elif len(bet_agent.replay_buffer) > 1500:
            bet_agent.train()  # Only 1 step until buffer fills
        bet_agent.train()

        # Update Target Network Periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            bet_agent.target_network.load_state_dict(bet_agent.q_network.state_dict())

        # Print Progress
        if episode % PRINT_FREQ == 0 and episode > 0:
            avg_reward = np.mean(total_rewards[-PRINT_FREQ:])
            print(f"Episode {episode}/{NUM_EPISODES}, Avg Training Reward: {avg_reward:.3f}, Epsilon: {bet_agent.epsilon:.3f}")

        # Track betting behavior by MORE GRANULAR count ranges
        if true_count >= 4:
            count_range = "Very High (+4 or more)"
        elif true_count >= 3:
            count_range = "High (+3 to +4)"
        elif true_count >= 1:
            count_range = "Slightly Favorable (+1 to +3)"
        elif true_count >= -1:
            count_range = "Neutral (-1 to +1)"
        else:
            count_range = "Unfavorable (below -1)"

        if count_range not in count_bet_stats:
            count_bet_stats[count_range] = {1: 0, 2: 0, 4: 0, 8: 0, 16: 0}

        count_bet_stats[count_range][BET_SIZES[bet_action_index]] += 1

    # Save the trained betting agent
    torch.save(bet_agent.q_network.state_dict(), "checkpoints/betting/bet_dqn.pth")
    print("Training complete.")

    # Training Analysis
    print("\n=== Training Analysis ===")
    if len(total_rewards) > 0:
        print(f"Training Rewards:")
        print(f"  Mean: {np.mean(total_rewards):.4f}")
        print(f"  Std Dev: {np.std(total_rewards):.4f}")
        print(f"  Min: {np.min(total_rewards):.4f}")
        print(f"  Max: {np.max(total_rewards):.4f}")
        
        positive = sum(1 for r in total_rewards if r > 0)
        negative = sum(1 for r in total_rewards if r < 0)
        zero = sum(1 for r in total_rewards if r == 0)
        
        print(f"  Positive rewards: {positive} ({positive/len(total_rewards)*100:.1f}%)")
        print(f"  Negative rewards: {negative} ({negative/len(total_rewards)*100:.1f}%)")
        print(f"  Zero rewards: {zero} ({zero/len(total_rewards)*100:.1f}%)")

    if len(raw_rewards) > 0:
        print(f"\nGame Results (Raw Rewards):")
        print(f"  Mean: ${np.mean(raw_rewards):.2f}")
        print(f"  Std Dev: ${np.std(raw_rewards):.2f}")
    
    # Betting Behavior Analysis
    print("\n=== Betting Behavior by Count ===")
    for count_range, bets in count_bet_stats.items():
        total = sum(bets.values())
        print(f"{count_range}:")
        for bet_size, count in bets.items():
            pct = (count/total)*100 if total > 0 else 0
            print(f"  {bet_size} units: {count} times ({pct:.1f}%)")

if __name__ == "__main__":
    train_betting_agent()