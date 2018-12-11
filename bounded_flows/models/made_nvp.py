# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca


# Adapted from code originally written by Brooks Paige: https://github.com/tbrx/compiled-inference

import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

def sample_mask_indices(D, H, simple=False):
    if simple:
        return np.random.randint(0, D, size=(H,))
    else:
        mk = np.linspace(0,D-1,H)
        ints = np.array(mk, dtype=int)
        ints += (np.random.rand() < mk - ints)
        return ints

def create_mask(D_observed, D_latent, H, num_layers):
    m_input = np.concatenate((np.zeros(D_observed), 1+np.arange(D_latent)))
    m_w = [sample_mask_indices(D_latent, H) for i in range(num_layers)]
    m_v = np.arange(D_latent)
    M_A = (1.0*(np.atleast_2d(m_v).T >= np.atleast_2d(m_input)))
    M_W = [(1.0*(np.atleast_2d(m_w[0]).T >= np.atleast_2d(m_input)))]
    for i in range(1, num_layers):
        M_W.append(1.0*(np.atleast_2d(m_w[i]).T >= np.atleast_2d(m_w[i-1])))
    M_V = (1.0*(np.atleast_2d(m_v).T >= np.atleast_2d(m_w[-1])))

    return M_W, M_V, M_A


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        """ Linear layer, but with a mask.
            Mask should be a tensor of size (out_features, in_features). """
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, input):
        mask = Variable(self.mask, requires_grad=False)
        if self.bias is None:
            return F.linear(input, self.weight*mask)
        else:
            return F.linear(input, self.weight*mask, self.bias)


class AbstractConditionalMADE(nn.Module):
    def __init__(self, D_observed, D_latent, H, num_layers):
        super(AbstractConditionalMADE, self).__init__()
        self.D_in = D_observed + D_latent
        self.D_out = D_latent
        assert num_layers >= 1

        # create masks
        M_W, M_V, M_A = create_mask(D_observed, D_latent, H, num_layers)
        self.M_W = [torch.FloatTensor(M) for M in M_W]
        self.M_V = torch.FloatTensor(M_V)
        self.M_A = torch.FloatTensor(M_A)

        # nonlinearities
        self.relu = nn.ReLU()

    def forward(self, x):
        raise NotImplementedError()

    def sample(self, parents):
        raise NotImplementedError()

    def logpdf(self, parents, values):
        raise NotImplementedError()

    def propose(self, parents, ns=1):
        """ Given a setting of the observed (parent) random variables, sample values of
            the latents; returns a tuple of both the values and the log probability under
            the proposal.

            If ns > 1, each of the tensors has an added first dimension of ns, each
            containing a sample of size [batch_size, D_latent] and [batch_size] """
        original_batch_size = parents.size(0)
        if ns > 1:
            parents = parents.repeat(ns,1)
        values = self.sample(parents)
        ln_q = self.logpdf(parents, values)
        if ns > 1:
            values = values.resize(ns, original_batch_size, self.D_out)
            ln_q = ln_q.resize(ns, original_batch_size)
        return values, ln_q

# TODO: We could setup a ConditionalSimplexMADE, although I don't think it can actually factorize.

# Aaron:
# Combination of MADE and a normalizing flow --> output layers have *many* parameters...
# We might consider adding the inputs (D_observed directly to the normalizing flows and avoiding MADE...)
# None of the this code has been checked yet.
class ConditionalRealValueMADE(AbstractConditionalMADE):
    def __init__(self, D_observed, D_latent, H, num_layers, flows):
        super(ConditionalRealValueMADE, self).__init__(D_observed, D_latent, H, num_layers)

        # Aaron: There must be one flow per latent variable (autoregressive model)
        assert(len(flows) == D_latent)

        self.flows = flows

        layers = [MaskedLinear(self.D_in, H, self.M_W[0])]
        for i in xrange(1,num_layers):
            layers.append(MaskedLinear(H, H, self.M_W[i]))
        self.layers = nn.ModuleList(layers)

        # Aaron: No skip connections for now...
        flow_layers = []
        for flow in flows:
            param_layers = []
            for parameter in flow.parameters():
                out = parameter.size()
                param_layers.append(MaskedLinear(H, out, self.M_V))
            flow_layers.append(ModuleList(param_layers))

        # Aaron: Can we nest modules like this?
        self.flow_layers = ModuleList(flow_layers)

        # initialize parameters
        for param in self.parameters():
            if len(param.size()) == 1:
                init.normal(param, std=0.01)
            else:
                init.uniform(param, a=-0.01, b=0.01)


    # Aaron: This is where we obtain the parameterization of our normalizing flow.
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = self.relu(layer(h))

        # Aaron: compute the parameters for each flow:
        flow_params = []
        for layers in flow_layers:
            parameters = []
            for param_layer in layers:
                # Aaron: append the activations, which are the parameters of the normalizing flow
                parameters.append(parameter_layer(h))
            flow_params.append(parameters)

        return flow_params

    # TODO: Implement sample and logpdf routines:

    def sample(self, parents, ns=1):
        """ Given a setting of the observed (parent) random variables, sample values of the latents.

            If ns > 1, returns a tensor whose first dimension is ns, each containing a sample
            of size [batch_size, D_latent] """
        assert parents.size(1) == self.D_in - self.D_out
        original_batch_size = parents.size(0)
        if ns > 1:
            parents = parents.repeat(ns,1)
        batch_size = parents.size(0)


        # sample noise variables
        FloatTensor = torch.cuda.FloatTensor if parents.is_cuda else torch.FloatTensor
        latent = Variable(torch.zeros(batch_size, self.D_out))
        randvals = Variable(torch.FloatTensor(batch_size, self.D_out))
        torch.randn(batch_size, self.D_out, out=randvals.data);
        # Aaron: Looks like he's using the Gumbel trick to compute discrete random variables?
        gumbel = Variable(torch.rand(batch_size, self.D_out, self.K).log_().mul_(-1).log_().mul_(-1))
        if parents.is_cuda:
            latent = latent.cuda()
            randvals = randvals.cuda()
            gumbel = gumbel.cuda()

        for d in xrange(self.D_out):
            full_input = torch.cat((parents, latent), 1)
            alpha, mu, sigma = self(full_input)
            _, z = torch.max(alpha.log() + gumbel, 2, keepdim=False)
            one_hot = torch.zeros(alpha.size())
            if parents.is_cuda: one_hot = one_hot.cuda()
            one_hot = one_hot.scatter_(2, z.data.unsqueeze(-1), 1).squeeze_().byte()
            tmp = randvals.data * sigma.data[one_hot].view(z.size())
            latent = Variable(tmp + mu.data[one_hot].view(z.size()))
        if ns > 1:
            latent = latent.resize(ns, original_batch_size, self.D_out)
        return latent

    def logpdf(self, parents, values):
        """ Return the conditional log probability `ln p(values|parents)` """
        full_input = torch.cat((parents, values), 1)
        alpha, mu, sigma = self(full_input)
        eps = 1e-6 # need to prevent hard zeros
        alpha = torch.clamp(alpha, eps, 1.0-eps)

        const = sigma.pow(2).mul_(2*np.pi).log().mul_(0.5)
        normpdfs = (values[:,:,None].expand(mu.size()) - mu).div(sigma).pow(2).div_(2).add_(const).mul_(-1)
        lw = normpdfs + alpha.log()
#         print "norm", normpdfs, normpdfs.sum()
#         print "alph", alpha.log(), alpha.log().sum()
        # need to do log-sum-exp along dimension 2
        A, _ = torch.max(lw, 2, keepdim=True)
        weighted_normal = (torch.sum((lw - A.expand(lw.size())).exp(), 2, keepdim=True).log() + A).squeeze(2)
        return torch.sum(weighted_normal, 1, keepdim=True)
