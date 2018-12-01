# @Author: aaronmishkin
# @Date:   18-11-25
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-11-25

import torch

""" Root finding methods for inverting planar flows. """


# Only implemented for one dimension right now -- this is all that's needed for planar flows.

def newtons_method(fn, grad_fn, max_iters=20):
    """ Newton's method implemented in for one dimensional problems. """

    x_0 = torch.normal(mean=torch.tensor(0.), std=10)
    x = newton_restart(fn, grad_fn, x_0, max_iters=20)

    while x is None:
        x_0 = torch.normal(mean=torch.tensor(0.), std=10)
        x = newton_restart(fn, grad_fn, x_0, max_iters=20)

    return x

def newton_restart(fn, grad_fn, x_0, max_iters=20):
    """ Newton's method implemented in for one dimensional problems. """
    y_i = fn(x_0)
    x_i = x_0
    i = 0
    while torch.abs(y_i) > 1e-6:
        i = i + 1
        grad_i = grad_fn(x_i)
        x_i = x_i - y_i / grad_i
        y_i = fn(x_i)

        if i == max_iters:
            return None

    return x_i
