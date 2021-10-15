import argparse

import gym
import numpy as np
import torch
import torch.nn as nn

from utils import Agent, Logger, one_hot_encode

import warnings
warnings.filterwarnings("ignore") # Risky, but going to make printing nicer.

parser = argparse.ArgumentParser(description='Training a Q-learning agent for the FrozenLake-v1')
parser.add_argument('--no_episodes', type=int, default=20000, help='Number of episodes in training')
parser.add_argument('--max_no_steps', type=int, default=100, help='Max number of steps per episode')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor in Bellmans equation')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--e', type=float, default=0.1, help='Probability of random action.')
parser.add_argument('--is_slippery', type=bool, default=False, help='Train in the /slippery/ environment or no?')
args = parser.parse_args()
print('Training with following parameters:')
print(args, '\n\n')

if __name__ == '__main__':
    environment = gym.make('FrozenLake8x8-v1', is_slippery=args.is_slippery)
    agent = Agent(
        gamma=args.gamma,
        e=args.e,
        state_space=environment.observation_space.n,
        action_space=environment.action_space.n)

    logger = Logger()

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)

    for episode in range(args.no_episodes):
        state = int(environment.reset())
        done = False
        logger.reset()

        for step in range(args.max_no_steps):
            state = one_hot_encode(state, environment.observation_space.n)
            action, Q = agent.take_action(state, environment)

            new_state, reward, done, _ = environment.step(action)

            if done and not reward:
                reward = -1

            train_loss = agent.update(Q, new_state, reward, action, state, environment.observation_space.n)

            logger.episode_loss += float(train_loss.data)

            agent.model.zero_grad()
            train_loss.backward()
            optimizer.step()

            logger.episode_reward += reward

            state = new_state

            if done:
                if reward > 0:
                    agent.e = 1. / ((episode / 50) + 10)
                break
        logger.loss_list.append(logger.episode_loss / step)
        logger.reward_list.append(logger.episode_reward)
        logger.e_list.append(agent.e)
        if episode % 1000 == 0:
            print(f'------------Epoch {episode}/{args.no_episodes}------------')
            print('Ratio of successful vs failed episodes: {}'.format(
                np.sum(np.array(logger.reward_list) > 0.0) / episode))

    print('\n\n\nTraining finished, saving the training plots...')
    logger.plot()

    print('\n\n\nLet us play one game using the trained agent!')
    state = int(environment.reset())
    done = False

    while True:
        environment.render()
        state = one_hot_encode(state, environment.observation_space.n)
        action, Q = agent.take_action(state, environment)

        new_state, reward, done, _ = environment.step(action)

        state = new_state
        if done == True:
            environment.render()
            print('Done!')
            break
