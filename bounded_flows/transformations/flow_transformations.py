# @Author: aaronmishkin
# @Date:   18-11-25
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-11-25

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import bounded_flows.root_finding.newton as newton

# Planar Flow Transformation

def planar_transform(Z, u, w, b):
    """ A planar flow as described in 'Variational Inference with Normalizing FLows, Rezende and Mohamed, ICML, 2015' """
    linear_combs = torch.mv(Z, w) + b

    return Z + u.mul(F.tanh(linear_combs).unsqueeze(1))


def planar_inverse_transform(Y, u, w, b):
    """ Inverse of a planar flow. The parameters of the planar flow must satisfy
            torch.dot(w, u) >= -1
        to be invertible. This function assumes that condition holds.
        Evaluating the inverse requires solving a non-linear root-finding problem.
    """
    Z = []
    wu = torch.dot(w, u)

    for y in Y:
        wy = torch.dot(w, y)
        fn, grad_fn = planar_root_fns(wu, wy, b)
        alpha = newton.newtons_method(fn, grad_fn)

        z_parallel = w.mul(alpha).div(torch.dot(w,w))
        z_orthogonal = y - z_parallel - u.mul(F.tanh(alpha + b))
        z = z_parallel + z_orthogonal

        Z.append(z)

    Z = torch.stack(Z)

    return Z

def planar_log_det_jac(Z, u, w, b):
    linear_combs = torch.mv(Z, w) + b

    linear_combs_exp = torch.exp(2 * linear_combs)
    h_prime = 4 * linear_combs_exp.div((linear_combs_exp + 1).pow(2))

    psi = torch.mul(w, h_prime.unsqueeze(1))
    det = 1 + torch.mv(psi, u)

    return torch.log(torch.abs(det))


# Coupling Transforms: Unchecked



def coupling_transform(Z, mask, s_fn, t_fn):
    Z_masked = Z.mul(mask)
    Y_masked = Z_masked

    Z_unmasked = Z.mul(1 - mask)
    Y_unmasked = Z_unmasked.mul(torch.exp(s_fn(Z_masked))) + t_fn(Z_masked)
    Y = Y_masked + Y_unmasked

    return Y


def coupling_inverse_transform(Y, mask, s_fn, t_fn):
    Y_masked = Y.mul(mask)
    Z_masked = Y_masked

    Y_unmasked = Y.mul(1 - mask)
    Z_unmasked = (Y_unmasked - t_fn(Z_masked)).mul(torch.exp( - s_fn(Z_masked)))

    Z = Z_masked + Z_unmasked

    return Z

def coupling_log_det_jac(Z, mask, s_fn, t_fn):
    Z_masked = Z.mul(mask)
    log_det = torch.sum(s_fn(Z_masked), dim=1)

    return log_det




# Helper Functions:
def planar_root_fns(wu, wy, b):

    fn = lambda alpha: torch.exp(2 * (alpha + b))*(alpha + wu - wy) - (wu + wy - alpha)
    grad_fn = lambda alpha: 1 + torch.exp(2 * (alpha + b)) * (1 + 2 * (alpha + wu - wy))

    return fn, grad_fn
