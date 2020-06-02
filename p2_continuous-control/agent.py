import random
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from device import device
from noise import OUNoise
from replaybuffer import ReplayBuffer


class HParameters:
    batch_size = 128            # training mini batch size
    buffer_size = int(1e5)      # replay buffer size
    gamma = 0.99                # reward discount rate
    lr_actor = 1e-3             # actor's learning rate
    lr_critic = 1e-3            # critic's learning rate
    tau = 1e-3                  # soft update of target NN parameters
    noise_decay = 0.999         # OU process noise decay
    target_episodes = 100       # target number of training episodes for reaching the target training score
    target_score = 30           # target training score
    moving_average_window = 20  # number of past episodes the moving average is applied


class DDPGAgent(nn.Module):

    CHECKPOINT_FILE = 'checkpoint.pt'
    
    def __init__(self, state_size, action_size, random_seed):
        """
        :param state_size:  dimension of each state
        :param action_size:  imension of each action
        :param random_seed:  random seed
        """
        super(DDPGAgent, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        print('DDPG Agent hyper parameters:\n' 
              'batch size:              {:8.0f}\n' 
              'buffer size:             {:8.0f}\n' 
              'reward discount - gamma: {:8.3f}\n' 
              'actor learning rate:     {:8.3f}\n' 
              'critic learning rate:    {:8.3f}\n' 
              'soft update - tau:       {:8.3f}\n' 
              'noise decay rate:        {:8.3f}\n'
              'target episodes:         {:8.0f}\n'
              'target score:            {:8.0f}\n'
              'moving average window:   {:8.0f}\n'
              .format(HParameters.batch_size,
                      HParameters.buffer_size,
                      HParameters.gamma,
                      HParameters.lr_actor,
                      HParameters.lr_critic,
                      HParameters.tau,
                      HParameters.noise_decay,
                      HParameters.target_episodes,
                      HParameters.target_score,
                      HParameters.moving_average_window))
        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed, [400, 300]).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, [400, 300]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=HParameters.lr_actor)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed, [400, 300]).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, [400, 300]).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=HParameters.lr_critic)

        # Initialize target networks weights with the local networks ones
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(random_seed, HParameters.buffer_size, HParameters.batch_size)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_decay = HParameters.noise_decay

    def act(self, state, noise=True):
        """
        Returns actions for given state as per current policy.

        :param state: state object
        :param noise: a flag that indicates OE process noise application on the actor's action
                      ("Exploration vs. Exploitation")
        """

        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).data.cpu().numpy()
        self.actor_local.train()

        if noise:
            # Add noise to the action in order to explore the environment ("Exploration vs. Exploitation")
            action += self.noise_decay * self.noise.sample()
            # Decay the noise process
            self.noise_decay *= self.noise_decay

        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states):
        """
        Save experience in replay buffer, and use random sample from buffer to learn.
        :param states: states of the agents
        :param actions: actions of the agents
        :param rewards: received rewards
        :param next_states: next states of the agents
        """

        # Save experience
        self.replay_buffer.add(states, actions, rewards, next_states)

        # Learn, if enough samples are available in memory
        if len(self.replay_buffer) > HParameters.batch_size:
            experiences = self.replay_buffer.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param experiences: tuple of (S, A, R, S')
        """

        states, actions, rewards, next_states = experiences
        
        # Update critic:
        # 1 - Get predicted next-state actions from actor_target model
        actions_next = self.actor_target(next_states)
        # 2 - Get predicted next-state Q-Values from critic_target model
        q_targets_next = self.critic_target(next_states, actions_next)
        # 3 - Compute Q targets for current states (y_i)
        q_targets = rewards + (HParameters.gamma * q_targets_next)
        # 4 - Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor:
        # 1- Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # 2 - Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
    def soft_update(self, local_model, target_model, tau=HParameters.tau):
        """
        Soft update model parameters.

            θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: the source PyTorch model: weights will be copied from
        :param target_model: the target PyTorch model: weights will be copied to
        :param tau: soft update hyper parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # @staticmethod
    def train(self, env, n_episodes=2000, max_t=1000):
        """
        :param env: Unity environment object
        :param n_episodes: maximum number of training episodes
        :param max_t: maximum number of timesteps per episode
        """

        scores = []      # episodic scores
        moving_avg = []  # moving average over 100 episodes and over all agents

        # Perform n_episodes of training
        time_training_start = time.time()
        for i in range(1, n_episodes+1):
            time_episode_start = time.time()
            self.noise.reset()

            brain_name = env.brain_names[0]
            env_info = env.reset(train_mode=True)[brain_name]
            num_agents = len(env_info.agents)
            states = env_info.vector_observations
            scores_episode = np.zeros(num_agents)           # rewards per episode per each agent

            for t in range(1, max_t+1):
                # Perform a step: S;A;R;S'
                actions = self.act(states)                  # select the next action for each agent
                env_info = env.step(actions)[brain_name]    # send the actions to the environment
                rewards = env_info.rewards                  # get the rewards
                next_states = env_info.vector_observations  # get the next states
                # Send the results to the Agent
                for (state, action, reward, next_state) in zip(states, actions, rewards, next_states):
                    self.step(state, action, reward, next_state)
                # Update the variables for the next iteration
                states = next_states
                scores_episode += rewards

            # Store the rewards and calculate the moving average
            scores.append(scores_episode.tolist())
            #moving_avg.append(np.mean(scores[-HParameters.target_episodes:], axis=0))
            moving_avg.append(np.mean(scores[-HParameters.moving_average_window:], axis=0))

            time_episode = time.time() - time_episode_start
            time_elapsed = time.time() - time_training_start
            time_episode_str = time.strftime('%M:%S', time.gmtime(time_episode))

            print('Episode {:3d} - {} - Score: {:3.2f} (max: {:3.2f} - min: {:3.2f})\n' 
                  'Moving average: {:3.2f} (max: {:3.2f} - min: {:3.2f})\n'
                  .format(i, time_episode_str, scores_episode.mean(),
                          scores_episode.max(), scores_episode.min(),
                          moving_avg[-1].mean(), moving_avg[-1].max(),
                          moving_avg[-1].min()))

            if i % 10 == 0:
                time_elapsed_str = time.strftime('%Hh%Mm%Ss', time.gmtime(time_elapsed))
                self.save_checkpoint(i, moving_avg, scores, time_elapsed_str)

            # Check if the environment has been solved
            if moving_avg[-1].mean() >= HParameters.target_score and i >= HParameters.target_episodes:
                time_elapsed_str = time.strftime('%Hh%Mm%Ss', time.gmtime(time_elapsed))
                print('\nEnvironment solved in {:d} episodes!\t' 'Average Score: {:.2f}\tElapsed time: {}'
                      .format(i, moving_avg[-1].mean(), time_elapsed_str))
                self.save_checkpoint(i, moving_avg, scores, time_elapsed_str)
                break

        return scores, moving_avg

    def save_checkpoint(self, i_episode, moving_avg, scores, time_elapsed_str):
        checkpoint = dict(agent_dict=self.state_dict(),
                          actor_dict=self.actor_local.state_dict(),
                          critic_dict=self.critic_local.state_dict(),
                          episodes=i_episode,
                          elapsed_time=time_elapsed_str,
                          scores=scores,
                          moving_avg=moving_avg)
        print('Saving model ...\n')
        torch.save(checkpoint, self.CHECKPOINT_FILE)

