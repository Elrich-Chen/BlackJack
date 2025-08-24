import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.betting_dqn import Betting_DQN
from memory.replay_buffer import Replay_Buffer

class Betting_Agent:
    def __init__(self, buffer_capacity=10000, batch_size=32,
    gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_min=0.05, 
    epsilon_decay=0.999, device="cpu", loss_type="mse"):
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.input_dim = 14
        self.action_dim = 5
        self.bet_sizes = [1, 2, 4, 8, 16]

        self.q_network = Betting_DQN(self.input_dim, 64, 5).to(self.device)
        self.target_network = Betting_DQN(self.input_dim, 64, 5).to(self.device) 

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

    def select_bet(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim-1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) #[batch size, input_features]
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        return action
    
    def store_experience(self, state, action_index, reward, next_state, done):
        """
        Store an experience tuple for replay training.
        We store action_index (0-4), not bet amount.
        """
        self.replay_buffer.add(state, action_index, reward, next_state, done)
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_weighted(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) #which action was taken during gameplay

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())