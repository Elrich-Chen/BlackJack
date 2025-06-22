from blackjackEnv import BlackjackEnv
from basicStrategy import basic_strategy
import pickle

def generate_dataset(max_per_class=20000, save_path="balanced_dataset.pkl"):
    env = BlackjackEnv()
    dataset = []
    counts = {0: 0, 1: 0}  # 0 = hit, 1 = stand

    while counts[0] < max_per_class or counts[1] < max_per_class:
        env.reset()
        state = env.get_state()

        score = int(state[0])
        is_soft = int(state[1])
        dealer_upcard = int(state[2])

        key = (score, is_soft, dealer_upcard)
        if key in basic_strategy:
            label = basic_strategy[key]

            # Only add if we still need more of this label
            if counts[label] < max_per_class:
                dataset.append((state, label))
                counts[label] += 1

    print(f"Final label counts: {counts}")
    
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved balanced dataset with {len(dataset)} samples to {save_path}")

    return dataset

# Run it directly
if __name__ == "__main__":
    generate_dataset()