import random
from collections import deque, namedtuple

import numpy as np
import torch

from device import device


class ReplayBuffer:
    """
    Fixed-size buffer for storing experience tuples.
    (S,A,R,S'): (states, actions, rewards, next states)
    """
    def __init__(self, seed, buffer_size, batch_size):
        """
        :param seed: random seed
        :param buffer_size:  buffer capacity; i.e., maximal number of experiences that can be stored
        :param batch_size:   training min-batch size
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.experience = namedtuple('Experience', field_names=['states', 'actions', 'rewards', 'next_states'])
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states)
        self.buffer.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().\
            to(device)

        return states, actions, rewards, next_states

    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self.buffer)
