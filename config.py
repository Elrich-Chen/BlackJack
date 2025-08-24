# config.py

# === BETTING PARAMETERS ===
UNIT = 10  # 1 unit = $50
BET_SIZES = [1, 2, 4, 8, 16]  # Units (map to $50, $100, etc.)

# Automatically compute min/max bet
MIN_BET = BET_SIZES[0] * UNIT
MAX_BET = BET_SIZES[-1] * UNIT

# === TRAINING PARAMETERS ===
NUM_EPISODES = 50000      # Number of hands for training betting agent
TARGET_UPDATE_FREQ = 1000 # Steps to update target network
PRINT_FREQ = 1000         # Episodes between printing progress
BATCH_SIZE = 32           # Replay buffer batch size
BUFFER_CAPACITY = 50000   # Replay buffer capacity
GAMMA = 0.99              # Discount factor
LR = 1e-4                 # Learning rate
EPSILON_START = 1.0       # Start exploration rate
EPSILON_END = 0.1         # Min exploration rate
EPSILON_DECAY = 0.995     # Epsilon decay factor

# === EVALUATION PARAMETERS ===
INITIAL_BANKROLL = 100_000  # Starting bankroll in dollars
EVAL_HANDS = 20_000      # Hands to play during evaluation

# === FILE PATHS ===
PLAYING_MODEL_PATH = "checkpoints/playing/dqn_model.pth"
BETTING_MODEL_PATH = "checkpoints/betting/bet_dqn.pth"