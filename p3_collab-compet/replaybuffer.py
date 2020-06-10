from collections import deque, namedtuple
import random
import torch
import numpy as np

from device import device


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        :param action_size: dimension of the action space
        :param buffer_size: capacity of the replay buffer
        :param batch_size: sample size that is retrieved from the replay buffer (matches the mini-batch size used during training)
        :param seed: random number generator seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience (S;A;R;S') into the replay buffer
        :param state: state
        :param action: action
        :param reward: reward
        :param next_state: next state
        :param done: done indicator flag
        :return:
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences (S;A;R;S') from the replay buffer.
        :return: a batch of experiences
        """
        """"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()\
            .to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()\
            .to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()\
            .to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()\
            .to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
