r"""
Utility functions for Variational Optimization.

This module includes a MLP model that parallelize forward calls with different seed.

Say you have a model giving you the output ``y = f(x, W)`` for an input ``x`` and 
parameters ``W``, and have a probability distribution over ``W``.
To sample multiple outputs using the traditional PyTorch model, 
you have to sample ``W``, change the weights and pass ``x`` through the model to get a single sample.
The sampling procedure is essentially sequential.

The ``MLPSampling`` model in this module allows you to sample multiple sets of parameters
and get multiple output with a single ``forward`` call, leveraging the parallel capabilities
of GPUs where possible to speed up sampling.
"""

import torch
from torch.optim import Optimizer
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module

import torchutils

__all__ = [
	'LinearWithSampling', 'MLPWithSampling'
]

class LinearWithSampling(Linear):
	r"""Extension of the ``torch.nn.Linear`` Module with support for sampling.

	See :meth:`torch.nn.Linear`` for a full documentation.
	"""
	def __init__(self, in_features, out_features, bias=True):
		super(LinearWithSampling, self).__init__(in_features, out_features, bias)
		
	def forward(self, input, weight_noise=None, bias_noise=None):
		r"""
		"""
		if weight_noise is None:
			output = input.matmul(self.weight.t())
		else:
			output = input.matmul(self.weight.t() + weight_noise)
			
		if self.bias is not None:
			output += self.bias
			if bias_noise is not None:
				output += bias_noise
				
		return output
		
class MLPWithSampling(Module):
	r"""MultiLayer Perceptron with support for parallel computation of samples
	
	Example:
		Creates a MLP with two hidden layers of size [64, 16], 
		taking 256-valued input and returning a single output.
		
			>>> model = MLP(256, [64, 16], 1)
	
	Arguments:
		input_size (int): Size of the input.
		hidden_sizes (List of int): Size of the hidden layers. 
			Defaults to [] (no hidden layer).
		output_size (int): Size of the output.
			Defaults to 1
		act_func: Activation function (see ``torch.nn.functional``).
			Defaults to ``torch.tanh``.
	"""
	
	def __init__(self, input_size, hidden_sizes=[], output_size=1, act_func=torch.tanh):
		super(MLPWithSampling, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.act = act_func
		
		if len(hidden_sizes) == 0:
			self.hidden_layers = []
			self.output_layer = LinearWithSampling(self.input_size, self.output_size)
		else:
			self.hidden_layers = nn.ModuleList([LinearWithSampling(in_size, out_size) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
			self.output_layer = LinearWithSampling(hidden_sizes[-1], self.output_size)
			
	def forward(self, x, noise=None):
		r"""Forward pass with noisy parameters
		
		Arguments:
			x (Tensor): [n x input_size]
			noise (list of Tensor): 
			
		Returns:
			(Tensor): y [n x output_size]
		"""
		if noise is None:
			y = x
			for layer_id in range(len(self.hidden_layers)):
				y = self.act(self.hidden_layers[layer_id](y))
			y = self.output_layer(y)
		else:
			assert len(noise) == len(list(self.parameters()))
			assert all(noise[0].shape[0] == noise[i].shape[0] for i in range(1,len(noise)))
			
			s = noise[0].shape[0]
			y = x.expand(s, -1, -1)
			
			for layer_id in range(len(self.hidden_layers)):
				y = self.act(self.hidden_layers[layer_id](y, noise[2*layer_id], noise[2*layer_id + 1]))
			y = self.output_layer(y)
		return y

if __name__ == "__main__":
	n = 10 
	s = 10

	mlp = MLPSampling(2, [3], 1)
	noiseScaling = 0.01
	noise = [torch.rand((s, *(reversed(list(p.shape)))), requires_grad=False)*noiseScaling for p in mlp.parameters()]
	X = torch.rand(n,2)
	
	print("params", [p.shape for p in mlp.parameters()])
	
	print("Noise", [eps.shape for eps in noise])
	print("X", X.shape)
	
	y = mlp(X)
	print("y", y.shape)
	
	print(torch.mean(y**2))
	print(torch.autograd.grad(torch.mean(y**2), mlp.parameters()))
	
	y = mlp(X, noise)
	print("y", y.shape)
	
	print(torch.mean(y**2))
	print(torch.autograd.grad(torch.mean(y**2), mlp.parameters()))
	