import random
from collections import deque
import numpy as np

class Replay_Buffer:
    def __init__(self, capacity: int):
        """
        Initializes the Replay Buffer. Makes use of the deque data type from collections. 
        Args: 
            capacity (int): Maximum number of experiences to store
        Returns:
            None
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        Args:
            state (np.array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state after the action.
            done (bool): Whether the episode ended after this step.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Sample a random batch of experiences from the buffer.
        Args:
            batch_size (int): Number of experiences to return.
        Returns:
            Tuple of np.arrays: (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_),
        )


    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)
