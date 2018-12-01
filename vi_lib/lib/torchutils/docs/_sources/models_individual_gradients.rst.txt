Individual Gradient computations
================================================================

PyTorch's autodifferentiation (AD) library is optimized to return gradients of scalar functions, not Jacobians ("gradients" of vectors).
This means that it is not possible to get individual gradients for each example passed through a Neural Network.
If ``pytorch.autograd`` is used to take the derivative of vector, it is assumed that the user wants the gradient of the sum of the elements of that vector.

As users often only want the overall gradient, this makes it possible to optimize the backward calls for that use case
and avoid storing big tensors with a dimension dedicated to the samples.

Goodfellow's trick
------------------

It is however still possible to use part of the automatic differentiation toolkit to get individual gradients 
without *too much* efforts, using Ian Goodfellow's `Efficient Per-Example Gradient Computations <https://arxiv.org/pdf/1510.01799.pdf>`_.
At least for simple models with linear layers.

Notation Setup
^^^^^^^^^^^^^^

A bit of notation; consider the following model, 

.. math::

	A_0 \stackrel{T_1(A_0, W_1)}{\longrightarrow} 
	Z_1 \stackrel{\sigma}{\longrightarrow}
	A_1 \stackrel{T_2(A_1, W_2)}{\longrightarrow}
	... \stackrel{T_L(A_{L-1}, W_L)}{\longrightarrow} 
	Z_L \stackrel{\sigma}{\longrightarrow}
	A_L \stackrel{\mathcal{L}(A_L, y)}{\longrightarrow} e
	
where 

* :math:`A_0` is the input of the network, 
* :math:`T_l(A_{l-1}, W_l)` is the transformation of the :math:`l` th layer parametrized by :math:`W_l`, 
  e.g. a linear layer :math:`T_l(A_{l-1}, W_l) = W_l A_{l-1}`,
* :math:`\sigma` is a parameter-less transformation, e.g. a sigmoid non-linearity,
* :math:`A_L` is the output of the network
* :math:`\mathcal{L}(A_L, y)` is the loss function (where the summing/averaging over examples happens)
* and :math:`e` is the loss.

The typical use of ``autograd`` would use a single function :math:`f(X) = e`, representing the model and loss, 
and give the derivative with respect to :math:`W_1, ..., W_L` as a list of tensors of matching sizes.

Single Layer
^^^^^^^^^^^^

Assume that we are interested in the per-example gradient computation for the :math:`l` layer, :math:`T_l(A_{l-1}, W_l)`, 
where :math:`W_L` is of size :math:`(d_{in}, d_{out})`. 
:math:`A_l` would be of size :math:`(n, d_{in})`, where :math:`n` is the number of examples passed through the network.
We would want the derivatives in a :math:`(n, d_{in}, d_{out})` Tensor ``dW_l``,
where ``dW_l[0,:,:]`` would be the :math:`(d_{in}, d_{out})` gradient for the first example.

Using the chain rule, we can rewrite the gradient of :math:`f` with respect to :math:`W_l` as the gradient of :math:`Z_l` with respect to :math:`W_l` and the gradient of :math:`f` with respect to :math:`Z_l`,

.. math::
	
	\frac{\partial f}{\partial W_l} = \frac{\partial Z_l}{\partial W_l} \frac{\partial f}{\partial Z_l}.

This is useful as ``autograd`` will happily compute :math:`G_l = \frac{\partial f}{\partial Z_l}`, 
the output being unidimensional, giving a :math:`(n, d_{out})` matrix.
The computation of :math:`\frac{\partial Z_l}{\partial W_l}` however is not supported by ``autograd``
- as :math:`Z_l` is not a scalar, the output would be the gradient of the sum of the elements of :math:`Z_l`.

This is also useful as the summation over the example dimension only happens during the computation of that final step, 
taking :math:`\frac{\partial Z_l}{\partial W_l}` and multiplying it with :math:`\frac{\partial f}{\partial Z_l}`.
We can do this step manually in an efficient way - assuming the transformation from :math:`A_{l-1}` to :math:`Z_l` is simple.

Manual Differentiation for the last step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Assuming that :math:`T_l` is a linear transformation, :math:`T_l(A_{l-1}, W_l) = W_l A_{l-1} = Z_l`, 
the derivative of :math:`Z_l` w.r.t :math:`W_l` is simply 

.. math::
	
	\frac{\partial Z_l}{\partial W_l} = A_{l-1}^\top.
	
If :math:`f` involved a summation over the examples, 
the gradient of that sum w.r.t. :math:`W_l` would be given by :math:`A_{l-1}^\top G_l` 
- the multiplication of :math:`(d_{in}, n) \times (n, d_{out})` matrices giving a :math:`(d_{in}, d_{out})` matrix 
where the summation happens in the example dimension.
The gradient with respect to the first example would be given by ``A_lm1[:, 0] @ G_l[0, :]`` (where ``A_lm1`` is :math:`A_{l-1}`)
and a *naive* way to compute the sum of the gradients for all examples would be 

.. code-block:: guess

	grad_w_l = torch.zeros((d_in, d_out))
	for i in range(n):
		grad_w_l += A_lm1[:, i] @ G_l[i, :]
		
To get the result we want - the gradients in a :math:`(n, d_{in}, d_{out})` tensor - we could use

.. code-block:: guess

	ind_grad_w_l = torch.zeros((n, d_in, d_out))
	for i in range(n):
		grad_w_l[i, :, :] = A_lm1[:, i] @ G_l[i, :]

.. warning::
	
	Don't actually use that code - for loops are incredibly inefficient.

To get the benefit of batch matrix computation, we can use ``torch.bmm``, 
where the batch dimension matches the examples.
Given tensors of sizes :math:`(n, d_{in}, 1), (n, 1, d_{out})`, ``bmm`` returns a :math:`(n, d_{in}, d_{out})` tensor 
- basically performing the previous piece of code in batch.
Thus, the following function call gives us the individual gradients,

.. code-block:: guess

	ind_grad_w_l = torch.bmm(G_l.unsqueeze(2), A_lm1.unsqueeze(1))

If the transformation is a linear transformation with a bias term, 
the gradient for the bias term can be computed similarly
and would simply ge ``G_l`` in that case.
	
Putting it together 
^^^^^^^^^^^^^^^^^^^

We'll need:

* A model's ``forward`` function returning the intermediate layers :math:`A_l, Z_l`.

	.. code-block:: python

		class MLP(nn.Module):
			
			def __init__(self, input_size, hidden_sizes):
				super(type(self), self).__init__()
				
				self.input_size = input_size
				self.act = F.relu
				
				self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
				self.output_layer = nn.Linear(hidden_sizes[-1], 1)

			def forward(self, x):
				A = x 
				
				activations = [A]
				linearCombs = []

				for layer in self.hidden_layers:
					Z = layer(A)
					A = self.act(Z)

					# Save the activations and linear combinations from this layer.
					activations.append(A)
					Z.retain_grad()
					Z.requires_grad_(True)
					linearCombs.append(Z)

				y = self.output_layer(A)
					
				# Save the linear combinations from the output
				y.retain_grad()
				y.requires_grad_(True)
				linearCombs.append(y)
				
				return (y, activations, linearCombs)
				
* A function to *manually* compute the last part of the differentiation for each layer

	.. code-block:: python 

		def goodfellow_backprop(activations, linearGrads):
		
			L = len(linearGrads)
			grads = []
			for l in range(L):
				G_l, A_lm1 = linearGrads[l], activations[l]
				
				if len(G.shape) < 2:
					G_l = G_l.unsqueeze(1)
				
				grads.append(torch.bmm(G_l.unsqueeze(2), A_lm1.unsqueeze(1)))
				grads.append(G_l) # Gradient for the bias term

			return grads

* Compute the derivative of the final function with respect to :math:`Z_l`.

	.. code-block:: python

		y_pred, activations, linearCombs = model.forward(X)
		loss = loss_func(y_pred, y)
		
		linearGrads = torch.autograd.grad(loss, linearCombs)
		gradients = goodfellow_backprop(activations, linearGrads)

	