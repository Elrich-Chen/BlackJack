import torch
from model import BlackjackNet
from blackjackEnv import BlackjackEnv

# Load the trained model
model = BlackjackNet()
model.load_state_dict(torch.load("basic_strategy_ai.pt"))
model.eval()

# Create a game environment
env = BlackjackEnv()
state = env.reset()

print(f"Initial Player Hand: {env.player_hand}")
print(f"Dealer Upcard: {env.dealer_hand[0]}")

# Play the game
while not env.done:
    # Convert state to tensor
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # Predict action (0 = hit, 1 = stand)
    with torch.no_grad():
        prediction = model(state_tensor)
        action = torch.argmax(prediction).item()

    # Show what the AI is doing
    print(f"AI decision: {'Hit' if action == 0 else 'Stand'}")

    # Take action in the environment
    state, reward, done, _ = env.step(action)

    # Show the updated hand
    print(f"Player hand: {env.player_hand}")
    print(f"Dealer hand: {env.dealer_hand}")
    print("----------")

# Show result
if reward == 1:
    print("🎉 AI WON!")
elif reward == -1:
    print("💀 AI LOST!")
else:
    print("🤝 It's a TIE (push)")
