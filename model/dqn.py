import torch
import torch.nn as nn    #nerual network tools
import torch.nn.functional as F    #neural network functions such as ReLu

"""
This defines a new class called BlackjackModel
It inherits from nn.Module — all neural nets in PyTorch must do this
"""
class DQN(nn.Module): #nn.Module is the base class for any model ?
    def __init__(self, input_size=17, hidden_size=128, output_size=4):
        super().__init__()    #calls the super class constructor 
        self.fc1 = nn.Linear(input_size, hidden_size)   #Linear means a layer of neurons with weights & bias. FIRST layer, 13 inputs and 64 outputs
        self.fc2 = nn.Linear(hidden_size, hidden_size)   #Second layer is 64 layers into 64 layers
        #self.fc3 = nn.Linear(hidden_size, hidden_size)   #Second layer is 64 layers into 64 layers
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):             # Defines how data should flow through the model. mandatory method to have defined !
        x = F.relu(self.fc1(x))       # first hidden layer
        x = F.relu(self.fc2(x))       # second hidden layer
        #x = F.relu(self.fc3(x))
        x = self.out(x)               # raw scores for each action
        return x