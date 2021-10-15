import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable


class Logger:
    def __init__(self):
        self.reward_list = []  # list of per-episode-reward-sum
        self.loss_list = []  # list of per-episode-mean-losses
        self.e_list = []  # history of e parameters over episodes (decaying)
        self.episode_reward = 0
        self.episode_loss = 0

    def reset(self):
        self.episode_reward = 0
        self.episode_loss = 0

    def plot(self):
        window = 1000

        plt.plot(pd.Series(self.reward_list).rolling(window).mean())
        plt.title('Reward Moving Average ({}-episode window)'.format(window))
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.savefig('reward_plot.png')
        plt.clf()

        plt.plot(pd.Series(self.loss_list).rolling(window).mean())
        plt.title('Loss Moving Average ({}-episode window)'.format(window))
        plt.ylabel('Loss')
        plt.xlabel('Episode')
        plt.savefig('loss_plot.png')


class Agent:
    def __init__(self, gamma, e, state_space, action_space):
        self.gamma = gamma
        self.loss_fn = nn.SmoothL1Loss()
        self.e = e
        # Simple, one layer network is sufficient to solve the problem.
        # I spend a lot of time to find out why it is required to set 'bias = False'
        # Pretty interesting: https://datascience.stackexchange.com/questions/32006/adding-a-bias-makes-q-learning-algorithm-ineffective
        self.model = nn.Sequential(nn.Linear(state_space, action_space, bias=False))

    def take_action(self, state, environment):
        """
        Either take the action according to the current policy (exploitation) or
        take a random action with small probability (exploration)
        """
        Q = self.model(state)
        if np.random.rand() < self.e:
            action = environment.action_space.sample()
        else:
            _, action = torch.max(Q, 1)
            action = int(action.data[0])
        return action, Q

    def update(self, Q, new_state, reward, action, state, observation_space):
        """
        Because the problem is so easy, it is sufficient do to 'on-policy'(ish), update per current sample.
        For more difficult problems, I would probably employ an experience buffer and sample batch of experiences
        from it.
        """
        Q_new = self.model(one_hot_encode(new_state, observation_space))
        max_Q_new, _ = torch.max(Q_new.data, 1)
        target = Q.data
        target[0, action] = reward + self.gamma * max_Q_new

        return self.loss_fn(self.model(state), target)


def one_hot_encode(state, state_space):
    """
    Encode the state into one-hot encoded vector of length state_space.
    E.g. state:int = 3, state_space: int = 5 -> one_hot: Tensor = [0, 0, 0, 1, 0]
    Finally, alter the tensor to be a valid input to the neural network.
    """
    one_hot = torch.zeros(state_space)
    one_hot[state] = 1.0
    return Variable(one_hot.unsqueeze(0))
