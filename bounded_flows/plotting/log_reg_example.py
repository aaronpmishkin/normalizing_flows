# @Author: aaronmishkin
# @Date:   2018-06-23T02:43:22+02:00
# @Email:  aaron.mishkin@riken.jp
# @Last modified by:   aaronmishkin
# @Last modified time: 2018-06-24T20:23:17+02:00


import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def visualize_prior_joint_posterior(w1,w2,prior_density,joint_density,posterior_density):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    prior_joint_levels = np.linspace(0, 0.004, 20)
    posterior_levels = np.linspace(0, 0.016, 20)

    c1 = plot_density(ax1, w1, w2, prior_density, prior_joint_levels, title="Prior", xlim=[-20,20], ylim=[-20,20])
    c2 = plot_density(ax2, w1, w2, joint_density, prior_joint_levels, title="Joint", xlim=[-20,20], ylim=[-20,20])
    c3 = plot_density(ax3, w1, w2, posterior_density, posterior_levels, title="Posterior", xlim=[-20,20], ylim=[-20,20])


    cb = fig.colorbar(c2, ax=ax2, ticks=[0, 0.001, 0.002, 0.003, 0.004])
    cb = fig.colorbar(c3, ax=ax3, ticks=[0, 0.004, 0.008, 0.012, 0.016])

    return


##########################
#### Helper Functions ####
##########################

def plot_objective_history(ax, iterations, objective_history, ylabel):
    ax.plot(list(iterations), objective_history, 'kx', alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Iterations')
    ax.set_title("Training Loss")


def plot_logreg_data(ax, X, y):
    ax.set_title('Synthetic logreg Data')

    class_one = X[y == 0, :]
    class_two = X[y == 1, :]

    ax.plot(class_one[:,0], class_one[:,1], 'bo', alpha=0.5, label='Class One')
    ax.plot(class_two[:,0], class_two[:,1], 'gs', alpha=0.5, label='Class Two')

    ax.set_xlabel("x_1"); ax.set_ylabel("x_2")
    ax.legend()
    ax.set_ylim([-3, 10])

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

def plot_cov_ellipse(ax, mu, Sigma, nstd=1.5, label='', color='r'):

    vals, vecs = np.linalg.eigh(Sigma)
    order = vals.argsort()[::-1]
    vals, vecs =  vals[order], vecs[:,order]

    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=mu, width=width, height=height, angle=theta, fill=False, color=color, lw='4')
    ax.plot(mu[0], mu[1], 'o', color=color, label=label)
    ax.add_artist(ellip)
    return ellip

def plot_map_boundary(ax, w):
    if w[1] == 0:
        boundary = 0
    else:
        boundary = - w[0] / w[1]

    x1 = np.arange(-7,4)

    ax.plot(x1, boundary * x1, 'r', lw=2, alpha=0.5, label='Decision Boundary')
    ax.legend()

    return ax

def plot_decision_boundaries(ax, sample_model, num_samples, X, y, mu, L, title):
    X = X.numpy()
    y = y.numpy()
    class_one = X[y == 0, :]
    class_two = X[y == 1, :]

    ax.plot(class_one[:,0], class_one[:,1], 'bo', alpha=0.5, label='Class One')
    ax.plot(class_two[:,0], class_two[:,1], 'gs', alpha=0.5, label='Class Two')

    ax.set_xlabel("x_1"); ax.set_ylabel("x_2")

    for i in range(num_samples):
        w, _ = sample_model(mu, L)
        plot_map_boundary(ax, w.detach().numpy())

    ax.legend_.remove()
    ax.set_title(title)
    ax.set_ylim([-3, 10])
