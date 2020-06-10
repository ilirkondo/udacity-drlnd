import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, fc3_units=128):
        """

        :param state_size: dimension of the state space
        :param action_size: dimension of the action space
        :param seed: ramndom number generator seed
        :param fc1_units: number of nodes in the first hidden layer
        :param fc2_units: number of nodes in the second hidden layer
        :param fc3_units: number of nodes in the third hidden layer
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the actor's NN
        :return:
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Calculates actions for given state.
        :param state: state
        :return: actions
        """
        """Build an actor (policy) network that maps states -> actions."""
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128, fc3_units=128):
        """

        :param state_size: state dimension
        :param action_size: action dimension
        :param seed: random number generator seed
        :param fcs1_units: number of nodes in the first hidden layer
        :param fc2_units: number of nodes in the second hidden layer
        :param fc3_units: number of nodes in the third hidden layer
        """

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

        self.reset_parameters()
        print('Critic network:', self.fc_layers)

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Calculate critic's Q value for given state and action.
        :param state: state
        :param action: action
        :return: Q-Value
        """

        s = state.view(-1, 48)
        a = action.view(-1, 4)

        xs = F.leaky_relu(self.fcs1(s))
        x = torch.cat((xs, a), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
