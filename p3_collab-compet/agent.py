import numpy as np
import random
from collections import deque

from device import device
from model import Actor, Critic
from replaybuffer import ReplayBuffer
from ounoise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


class HParameters:
    buffer_size = int(1e5)  # replay buffer size
    batch_size = 128        # mini-batch size
    gamma = 0.99            # reward discount factor
    tau = 1e-3              # for soft update of target parameters
    lr_actor = 1e-5         # actor's learning rate
    lr_critic = 1e-4        # critic's learning rate
    weight_decay = 0        # NN weight decay (L2 penalty)
    update_every = 10       # number of steps before updating the agent's target networks


class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient Agent that Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, n_agnets, random_seed):
        """
        :param state_size:  state dimension
        :param action_size: action dimension
        :param n_agnets:    number of agents
        :param random_seed: random generator seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.n_agnets = n_agnets
        self.seed = random.seed(random_seed)

        # Actor Network (with Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=HParameters.lr_actor)

        # Critic Network (with Target Network)
        self.critic_local = Critic(state_size * self.n_agnets,
                                   action_size * self.n_agnets,
                                   random_seed).to(device)

        self.critic_target = Critic(state_size * self.n_agnets,
                                    action_size * self.n_agnets,
                                    random_seed).to(device)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=HParameters.lr_critic,
                                           weight_decay=HParameters.weight_decay)

        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        # Ornstein-Uhlenbeck noise process
        self.noise = OUNoise(action_size, random_seed)

        # Experiences Replay buffer
        self.buffer = ReplayBuffer(action_size, HParameters.buffer_size, HParameters.batch_size, random_seed)

        # Step counter
        self.t_step = 0

    def hard_copy_weights(self, target, source):
        """
        Copy weights from source to target network (part of initialization)

        :param target: target NN parameters / weights
        :param source: source NN parameters / weights
        :return:
        """

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, state, action, reward, next_state, done, num_updates=1):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Save experience / reward
        self.buffer.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % HParameters.update_every

        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.buffer) > HParameters.batch_size:
                for _ in range(num_updates):
                    experiences = self.buffer.sample()
                    self.learn(experiences, HParameters.gamma)

    def act(self, state, add_noise=True):
        """
        Evaluates actions using the local actor's NN with the given state; performs a training step and returns the actions.

        :param state: current state
        :param add_noise: add-noise indicator
        :return: actions
        """
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset_noise(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param experiences: (S, A, R, S')
        :param gamma: reward discount factor
        :return:
        """

        states, actions, rewards, next_states, dones = experiences

        # Update critic
        # 1 - Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)

        # 2 - Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # 3 - Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)

        # 4 - Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Update actor
        # 1 - Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # 2 - Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, HParameters.tau)
        self.soft_update(self.actor_local, self.actor_target, HParameters.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model:  model (weights will be copied from)
        :param target_model: model (weights will be copied to)
        :param tau: interpolation parameter
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self, env, n_episodes=2000, print_every=100):
        """
        Perfoms training as per "Deep Deterministic Policy Gradient" - Method (DDPG)
        :param env: training environment (in our case, the Unity Environment Object)
        :param n_episodes: number of episodes
        :param print_every: print score every 'print_every' episodes
        :return:
        """

        scores_window = deque(maxlen=100)  # save last 100 total scores in one episode
        all_scores = []
        avg_scores_window = []
        max_score = 0  # save best score in that run

        for i_episode in range(1, n_episodes + 1):
            brain_name = env.brain_names[0]
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment in training mode
            num_agents = len(env_info.agents)
            states = env_info.vector_observations              # get the current state (for each agent)

            self.reset_noise()
            # initialize the score
            scores = np.zeros(num_agents)

            while True:
                actions = self.act(states)                    # select an action (for each agent)
                env_info = env.step(actions)[brain_name]      # send all actions to tne environment

                rewards = env_info.rewards                    # get reward (for each agent)
                next_state = env_info.vector_observations     # get next state (for each agent)
                dones = env_info.local_done                   # get done value (for each agent)

                self.step(states, actions, rewards, next_state, dones, num_updates=3)  # agent step

                states = next_state
                scores += rewards

                if np.any(dones):                             # see if episode finished
                    break

                    # score for one episode of mean of all agents
            avg_score = np.mean(scores)

            # save last 100 avg_score scores
            scores_window.append(avg_score)

            all_scores.append(avg_score)
            avg_scores_window.append(np.mean(scores_window))

            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)), end="")

            # save agent if 100 perfomance is better, that max_score
            if max_score < np.mean(scores_window):
                torch.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(self.critic_local.state_dict(), 'checkpoint_critic.pth')
                max_score = np.mean(scores_window)

            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.6f}'.format(i_episode, np.mean(scores_window)))

        return all_scores, avg_scores_window

    def plot(self, all_scores, avg_scores_window):
        """
        Plots training results.

        :param all_scores: all scores (rewards) through all episodes
        :param avg_scores_window: moving average scores
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(all_scores) + 1), all_scores)
        plt.plot(np.arange(1, len(avg_scores_window) + 1), avg_scores_window)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
