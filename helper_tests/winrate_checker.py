import os
import sys
print(sys.path) #all the different folders
print("-------------------------------")
print(os.path.abspath(".")) #absolute path of parent folder
print("-------------------------------")
sys.path.append(os.path.abspath("."))
print("-------------------------------")
import torch
from blackjackEnv import BlackjackEnv
from model import BlackjackNet

env = BlackjackEnv()
model = BlackjackNet()
#torch allows us to load tensors and models form the disk ? 
model.load_state_dict(torch.load("basic_strategy_ai.pt"))
model.eval()

trials = 10_000
wins = 0
losses = 0
draws = 0

for trial in range(trials):
    starting_state = env.reset()

    while not env.done:
        state_tensor = torch.tensor(starting_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = model(state_tensor)
            action = torch.argmax(prediction, dim=1).item()
            starting_state, reward, done, _ = env.step(action)
        
    if reward == 1:
        wins += 1
    elif reward == -1:
        losses +=1
    else:
        draws += 1

print(f"the win rate is {wins/10_000 *100} %") #it should be 44.smth percent
print(f"the loss rate is {losses/10_000 *100} %")
print(f"the draw rate is {draws/10_000 *100} %")
