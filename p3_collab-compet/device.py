import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------------------------------
# ...
# UserWarning: 
#    Found GPU0 GeForce RTX 2070 with Max-Q Design which requires CUDA_VERSION >= 9000 for
#     optimal performance and fast startup time, but your PyTorch was compiled
#     with CUDA_VERSION 8000. Please install the correct PyTorch binary
#     using instructions from http://pytorch.org
#    
#  warnings.warn(incorrect_binary_warn % (d, name, 9000, CUDA_VERSION))
# Traceback (most recent call last):
#  File "main.py", line 33, in <module>
#    scores = agent.train(env=env, brain_name=brain_name, n_episodes=1000)
#  File "/home/ikondo/deep-reinforcement-learning/p1_navigation/agent.py", line 118, in train
#    action = self.act(state, eps)
#  File "/home/ikondo/deep-reinforcement-learning/p1_navigation/agent.py", line 58, in act
#    action_values = self.qnn_local(state)
#  File "/home/ikondo/.conda/envs/drlnd/lib/python3.6/site-packages/torch/nn/modules/module.py", line 491, in __call__
#    result = self.forward(*input, **kwargs)
#  File "/home/ikondo/deep-reinforcement-learning/p1_navigation/qnn.py", line 26, in forward
#    x = F.relu(self.fc1(state))
#  File "/home/ikondo/.conda/envs/drlnd/lib/python3.6/site-packages/torch/nn/modules/module.py", line 491, in __call__
#    result = self.forward(*input, **kwargs)
#  File "/home/ikondo/.conda/envs/drlnd/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 55, in forward
#    return F.linear(input, self.weight, self.bias)
#  File "/home/ikondo/.conda/envs/drlnd/lib/python3.6/site-packages/torch/nn/functional.py", line 992, in linear
#    return torch.addmm(bias, input, weight.t())
# RuntimeError: cublas runtime error : the GPU program failed to execute at /pytorch/aten/src/THC/THCBlas.cu:249

# This must be due to the fact to the fact that the conda environment is set up with PyTorch 0.4 which is build on an
# earlier CUDA version. To my understanding the older PyTorch version is conditioned from the binary of the Unity
# environment.

device = torch.device('cpu')
