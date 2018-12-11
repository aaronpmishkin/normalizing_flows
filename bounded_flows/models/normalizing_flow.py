# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca

from functools import partial

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import bounded_flows.transformations.support_transformations as ST
import bounded_flows.transformations.flow_transformations as FT
import torch.nn.functional as F


class NormalizingFlow(nn.Module):

    def __init__(self, initial_dist, dimension, num_layers=5, TransformLayer=None, transform_params=None, support_transform="r", support_bounds=None):
        super(type(self), self).__init__()

        self.initial_dist = initial_dist

        if dimension:
            self.dimension = dimension
        else :
            self.dimension = 1

        self.transform = FT.planar_transform
        self.log_det_jac = FT.planar_log_det_jac
        self.inverse_transform = FT.planar_inverse_transform

        # Setup the transformations:
        self.transform_layers =  nn.ModuleList([TransformLayer(dimension, transform_params[k]) for k in range(num_layers)])
        self.output_layer = SupportTransformLayer(support_transform, bounds=support_bounds)

    # Forward pass implements sampling:
    def forward(self, z=None, num_samples=1):
        if z is None:
            z = self.initial_dist.sample_n(num_samples)

        log_prob = self.initial_dist.log_prob(z)

        for layer in self.transform_layers:
            z, log_prob = layer.forward(z, log_prob)

        y, log_prob = self.output_layer.forward(z, log_prob)

        return y, log_prob

    # Inverse pass computes the log_prob of an example.
    def inverse(self, y):
        z, log_prob = self.output_layer.inverse(y, 0)

        for layer in reversed(self.transform_layers):
            z, log_prob = layer.inverse(z, log_prob)

        log_prob = log_prob + self.initial_dist.log_prob(z)

        return z, log_prob

    def make_kl_fn(prior_dist):

        def kl_fn():
            theta, q_log_prob = self.forward(num_samples=num_mc_samples)
            p_log_prob = prior_dist.log_prob(theta)
            kl_divergence = torch.sum(q_log_prob - p_log_prob) / num_mc_samples

            return kl_divergence

        return kl_fn

class SupportTransformLayer(nn.Module):
    def __init__(self, support_transform, bounds=None):
        super(type(self), self).__init__()

        if support_transform == 'r':
            self.transform = ST.identity_transform
            self.log_det_jac_fn = ST.identity_log_det_jac
            self.inverse_transform = ST.identity_transform
        elif support_transform == 'r_plus':
            self.transform = ST.r_plus_transform
            self.log_det_jac_fn = ST.r_plus_log_det_jac
            self.inverse_transform = ST.r_plus_inverse_transform
        elif support_transform == 'r-bounded':
            c, b = bounds
            # TODO: Support bounds over only a subset of the dimensions.
            self.transform = partial(ST.r_bounded_transform, c=c, b=b)
            self.log_det_jac_fn = partial(ST.r_bounded_log_det_jac, c=c, b=b)
            self.inverse_transform = partial(ST.r_bounded_inverse_transform, c=c, b=b)
        elif support_transform == 'simplex':
            self.transform = ST.simplex_transform
            self.log_det_jac_fn = ST.simplex_log_det_jac
            self.inverse_transform = ST.simplex_inverse_transform
        else:
            raise ValueError("Transform not supported.")

    def forward(self, z, log_prob):
        y = self.transform(z)
        log_prob = log_prob - self.log_det_jac_fn(z)

        return y, log_prob

    def inverse(self, y, log_prob):
        z = self.inverse_transform(y)
        log_prob = log_prob - self.log_det_jac_fn(z)

        return z, log_prob


class CouplingTransformLayer(nn.Module):

    def __init__(self, dimension, params=None):
        super(type(self), self).__init__()

        if params is None:
            raise ValueError("Coupling transform parameters cannot be 'None'")

        self.dimension = dimension

        mask, s_fn, t_fn = params
        self.mask = mask
        self.s_fn = s_fn
        self.t_fn = t_fn

        # register the parameters of the scale and shift
        # operaters. This ensures that fitting the coupling layer
        # fits the parameterizing functions.
        for i, param in enumerate(s_fn.parameters()):
            self.register_parameter("s_fn_" + str(i), param)

        for i, param in enumerate(t_fn.parameters()):
            self.register_parameter("t_fn_" + str(i), param)


    def forward(self, z, log_prob):
        y = FT.coupling_transform(z, self.mask, self.s_fn, self.t_fn)
        log_prob = log_prob - FT.coupling_log_det_jac(z, self.mask, self.s_fn, self.t_fn)

        return y, log_prob

    def inverse(self, y, log_prob):
        z = FT.coupling_inverse_transform(y, self.mask, self.s_fn, self.t_fn)
        log_prob = log_prob - FT.coupling_log_det_jac(z, self.mask, self.s_fn, self.t_fn)

        return z, log_prob



class PlanarTransformLayer(nn.Module):

    def __init__(self, dimension, params=None):
        super(type(self), self).__init__()
        self.w = Parameter(torch.ones(dimension, requires_grad=True))
        self.u = Parameter(torch.zeros(dimension, requires_grad=True))
        self.b = Parameter(torch.tensor(0., requires_grad=True))

    def _u_reparam(self):
        wu = torch.dot(self.w, self.u)
        m = F.softplus(wu) - 1
        w_norm = torch.dot(self.w,self.w)

        u_reparam = self.u + self.w.mul((m - wu) / w_norm)

        return u_reparam

    def forward(self, z, log_prob):
        u_reparam = self._u_reparam()

        y = FT.planar_transform(z, u_reparam, self.w, self.b)
        log_prob = log_prob - FT.planar_log_det_jac(z, u_reparam, self.w, self.b)

        return y, log_prob

    def inverse(self, y, log_prob):
        u_reparam = self._u_reparam()
        z = FT.planar_inverse_transform(y, u_reparam, self.w, self.b)

        log_prob = log_prob - FT.planar_log_det_jac(z, u_reparam, self.w, self.b)

        return z, log_prob






#
