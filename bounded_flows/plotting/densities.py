# @Author: aaronmishkin
# @Date:   2018-06-23T02:43:22+02:00
# @Email:  aaron.mishkin@riken.jp
# @Last modified by:   aaronmishkin
# @Last modified time: 2018-06-24T20:23:17+02:00


import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib.tri as tri
from matplotlib.patches import Ellipse


def plot_density(ax, w1, w2, density, levels=None, title="", xlim=[-2,20], ylim=[-2,20]):
    ax.set_title(title)
    if levels is None:
        contour = ax.contourf(w1, w2, density)
    else:
        contour = ax.contourf(w1, w2, density, levels=levels)
    ax.set_xlabel("w_1"); ax.set_ylabel("w_2")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return contour

# Adapted from https://gist.github.com/tboggs/8778945


_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])
_midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3]) / 2.0 \
              for i in range(3)]


def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.
    Arguments:
        `xy`: A length-2 sequence containing the x and y value.
    '''
    s = [(_corners[i] - _midpoints[i]).dot(xy - _midpoints[i]) / 0.75 \
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)

def plot_simplex_density(density_fn, nlevels=200, subdiv=8, levels=None, **kwargs):

    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    mesh = [torch.tensor(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    mesh = torch.stack(mesh, dim=0).to(torch.float)
    log_probs = density_fn(mesh)
    log_probs = log_probs.detach().numpy()


    if levels is None:
        v = plt.tricontourf(trimesh, log_probs, nlevels, **kwargs)
    else:
        v = plt.tricontourf(trimesh, log_probs, levels=levels, **kwargs)

    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')

    return v
