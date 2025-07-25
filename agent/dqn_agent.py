import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.dqn import DQN
from memory.replay_buffer import Replay_Buffer

class DQN_Agent:
    def __init__(self, buffer_capacity=10000, batch_size=32,
    gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_min=0.1, 
    epsilon_decay=0.995, device="cpu", loss_type="mse"):
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device) 

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  
        self.train_step = 0
        self.target_update_freq = 100  

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError("Unsupported loss type")

        self.replay_buffer = Replay_Buffer(buffer_capacity)
    
    def select_action(self, state):
        """
        Choose an action using epsilon-greedy
        """
        if random.random() < self.epsilon:
            return random.randint(0,3)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
        
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self):
        """
        Sample the buffer, apply Bellman equation, update weights after
        """
        if len(self.replay_buffer) < 32:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions) #gathering values that the ai model has *previously* predicted. it goes by column and grabs the q-value based on the aciton index

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * max_next_q * (~dones) # remember that this is a np array or wtv of target_q values

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
    

        
