import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from qnn import QNN
from device import device
from collections import deque

# NN hyper-parameters

BUFFER_SIZE  = int(1e5) # replay buffer size
BATCH_SIZE   = 64       # training batch size
GAMMA        = 0.99     # reward discount factor
TAU          = 1e-3     # for soft update of target parameters
LR           = 5e-4     # learning rate 
UPDATE_EVERY = 4        # how often to update the network

class Agent():
    def __init__(self, state_size, action_size, seed, min_score=13.0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Initialize Q-NN Networks
        self.qnn_local  = QNN(state_size, action_size, seed).to(device)
        self.qnn_target = QNN(state_size, action_size, seed).to(device)
        self.optimizer  = optim.Adam(self.qnn_local.parameters(), lr=LR)

        # Initialize Replay Memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step
        self.t_step = 0
        
        # Initialize minimal score threshold
        self.m_score = min_score
    
    def step(self, state, action, reward, next_state, done):
        # 1 - Save experience in Replay Memory
        self.memory.add(state, action, reward, next_state, done)
        
        # 2 - Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If there are enough samples available in the Replay Memory, then pick a random experiences subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnn_local.eval()
        with torch.no_grad():
            action_values = self.qnn_local(state)
        self.qnn_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) # pick most probalbe action from the Local NN
        else:
            return random.choice(np.arange(self.action_size))  # pick a random action to explore a new experience

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # 1   - Forward Propagation
        
        # 1.1 - Get action index from the Local QNN.
        Q_local_idx = self.qnn_local(next_states).detach().argmax(1).unsqueeze(1)
        
        # 1.2 - Get all action indexes from the Target QNN for the action provided from the Local QNN: action_size x 1  
        Q_targets_next = self.qnn_target(next_states).detach().gather( 1, Q_local_idx)
        
        # 1.3 - Calculate rewards from the Target QNN for the current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 1.4 - Calculate expected rewards from the Local QNN for the states and actions of the given experiences
        Q_expected = self.qnn_local(states).gather(1, actions)

        # 2 - Calculate the loss using MSE as defined in https://pytorch.org/docs/stable/nn.html
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # 3 - Backward Propagation
        
        # 3.1 Clear old gradients from the last step
        #     (Otherwise the gradients from all loss.backward() calls would be accumulated)
        self.optimizer.zero_grad()
        
        # 3.2 Calculate the gradient of the loss based on the parameters of the Local & Target NNs
        loss.backward()
        
        # 3.3 Make a gradient descent step through the optimizer based on the gradient of the loss.
        self.optimizer.step()
        
        # 4 Update Target QNN
        self.soft_update(self.qnn_local, self.qnn_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        # Calculate wighted parameters for the Taget NN and copy them into the Target NN
        # e.g. for TAU = 0.1, then target_weights = target_params*0.9 + local_params*0.1
        # This means that we are updating only 10% of new weights and we use 90% old weights.
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def train(self, env, brain_name, n_episodes=2000, max_t=100000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for t in range(max_t):
                action = self.act(state, eps)
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=self.m_score:
                print('\nEnvironment solved in {:d} episodes.\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(self.qnn_local.state_dict(), 'checkpoint.pth')
                break
        return scores