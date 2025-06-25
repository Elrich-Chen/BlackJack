from blackjackEnv import BlackjackEnv
from basicStrategy import basic_strategy
import pickle, random

def generate_dataset(max_per_class=20000, save_path="balanced_dataset.pkl"):
    env = BlackjackEnv()
    dataset = []
    counts = {0: 0, 1: 0}  # 0 = hit, 1 = stand
    soft = 0
    soft_target = int(0.4*max_per_class)

    while counts[0] < max_per_class or counts[1] < max_per_class: #check for BOTH {hit} and {stand} hands (50/50)
        env.reset()
        state = env.get_state() #this holds the starting point of the game state, 2 cards with player and one dealer upcard

        score = int(state[0])
        is_soft = int(state[1])
        dealer_upcard = int(state[2])

        key = (score, is_soft, dealer_upcard)
        if key in basic_strategy:  #basically what if the score is <5 or score > 17, many edge cases might be missed
            label = basic_strategy[key]

            if is_soft and soft < soft_target: #if we have enough soft hands, stop adding them and go to next iteration
                soft += 1
            # Only add if we still need more of hit or stand label
            elif counts[label] < max_per_class:
                dataset.append((state, label))
                counts[label] += 1

    print(f"Final label counts: {counts}")
    print(f"Number of soft hands %: {soft/max_per_class}") #check the number of soft hands
    
    with open(save_path, "wb") as f:
        random.shuffle(dataset) #in case we have many hit or stand entries clumped together, it is much better to shuffle them
        pickle.dump(dataset, f)
    print(f"Saved balanced dataset with {len(dataset)} samples to {save_path}")

    return dataset

# Run it directly
if __name__ == "__main__":
    generate_dataset()