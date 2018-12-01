r"""
Utility functions to handle parralel computation of individual gradients
"""

import torch
from torch.optim import Optimizer
from torchutils.models import MLP 

__all__ = [
]

class MLPIndGrad(MLP):
	r"""MultiLayer Perceptron with support for individual gradient computation
	"""
	
	def __init__(self, input_size, hidden_sizes=[], output_size=1, act_func=torch.tanh):
		r"""See ``torchutils.models.MLP``
		"""	
		super(MLP, self).__init__(input_size, hidden_sizes, output_size, act_func)

    def forward(self, x):
		r"""
		"""
        activation = x
		
		activations = [activation]
		linearCombs = []

        for layer in self.hidden_layers:
            linearComb = layer(activation)
            activation = self.act(linearComb)

			linearComb.retain_grad()
			linearComb.requires_grad_(True)
			
			linearCombs.append(linearComb)
			activations.append(activation)

        output = self.output_layer(activation)

		output.retain_grad()
		output.requires_grad_(True)
		
		linearComb.append(output)

		return (output, activations, linearCombs)



def goodfellow_grad(G, X)

def goodfellow_backprop(activations, linearCombs, backprop_func, n_out):
	outs = tuple([[] for i in range(n_out)])
    for i in range(len(linearCombs)):
		out = backprop_func(linearCombs[i], activations[i])
		
		for j in range(n_out):
			outs[j].append(out[j])

    return *outs
