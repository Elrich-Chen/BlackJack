import torch
import torch.nn as nn    #nerual network
import torch.nn.functional as F

class BlackjackNet(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, output_size=2):
        super(BlackjackNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))       # first hidden layer
        x = F.relu(self.fc2(x))       # second hidden layer
        x = self.out(x)               # raw scores for each action
        return x
    