from collections import Counter
import pickle

# Load dataset
with open("balanced_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Count how many are 'hit' (0) and 'stand' (1)
labels = [label for _, label in dataset]
counts = Counter(labels)

print("Number of 'hit' labels (0):", counts[0])
print("Number of 'stand' labels (1):", counts[1])