r"""
Utility functions for optimizers that need higher order information (e.g., Hessian).

This module is similar in spirit to Pyro's ``MultiOptimizer`` 
(`link <http://docs.pyro.ai/en/0.2.1-release/optimization.html#pyro.optim.multi.MultiOptimizer>`_),
but differs on which class should know how to differentiate the module.

* in Pyro, the ``step`` function takes a ``loss`` Tensor as an argument.
  The optimizer is then responsible for computing the required derivatives.
* The approach taken is this module is that the ``step`` function takes a ``closure`` as an argument.
  This closure, given data points as an input, computes the required derivates and would return, for example, the loss, gradients and the diagonal of the Hessian.

The reasoning behind that approach is that the computation of higher order derivatives is more often model dependent than for gradients,
and optimizations are possible for some models. 
The ``closure`` acts as a link between the ``model`` and ``optimizer`` classes, which should be independent.

Another advantage of this approach is that it makes it possible to try different approximations of higher order information
without requiring duplication of optimizers, which would be identitcal except in their computation of the second order information,
e.g., a single ``optimizer`` could handle optimization based on the diagonal of the Hessian, Generalized-Gauss Newton and Fisher matrices
with different closures.
"""

import torch

__all__ = [
]
