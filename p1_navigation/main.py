import torch
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from device import device

env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

brain_name = env.brain_names[0]                  # get the default brain
print('Brain name:', brain_name)
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name] # reset the environment

print('Number of agents:', len(env_info.agents))

action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)
      
agent = Agent(state_size=state_size, action_size=action_size, seed=0, min_score=13.0)
scores = agent.train(env=env, brain_name=brain_name, n_episodes=1000)
      
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

input("Press Enter to continue...")

env.close()      
      

      
