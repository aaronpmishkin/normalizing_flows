# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca

# Adapted from code originally written by Brooks Paige: https://github.com/tbrx/compiled-inference

num_points = 10 # number of points in the synthetic dataset we train on

import numpy as np
import torch
from torch.autograd import Variable
import sys, inspect

import pymc as pymc
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Import our model:
from bounded_flows.models.real_nvp import conditional_real_nvp_flow

import pickle

# set the seeds for reproducibility.
np.random.seed(12)
torch.manual_seed(12)

# Helper functions:
def systematic_resample(log_weights):
    A = log_weights.max()
    normalizer = np.log(np.exp(log_weights - A).sum()) + A
    weights = np.exp(log_weights - normalizer)
    ns = len(weights)
    cdf = np.cumsum(weights)
    cutoff = (np.random.rand() + np.arange(ns))/ns
    return np.digitize(cutoff, cdf)

# Model Definition:

def robust_regression(x, t, sigma_0=np.array([10.0, 1.0, .1]), epsilon=1.0):
    """ X: input (NxD matrix)
        t: output (N vector)
        sigma_0: prior std hyperparameter for weights
        epsilon: std hyperparameter for output noise """

    if x is not None:
        N, D = x.shape
        assert D == 1
    else:
        N = num_points
        D = 1

    # assume our input variable is bounded by some constant
    const = 10.0

    x = pymc.Uniform('x', lower=-const, upper=const, value=x, size=(N, D), observed=(x is not None), rseed=42)

    # create design matrix (add intercept)
    @pymc.deterministic(plot=False)
    def X(x=x, N=N):
        return np.hstack((np.ones((N,1)), x, x**2))

    w = pymc.Laplace('w', mu=np.zeros((D+2,)), tau=sigma_0**(-1.0), rseed=42)

    @pymc.deterministic(plot=False, trace=False)
    def mu(X=X, w=w):
        return np.dot(X, w)

    y = pymc.NoncentralT('y', mu=mu, lam=epsilon**(-2.0), nu=4, value=t, observed=(t is not None), rseed=42)

    return locals()

def get_observed(model):
    return np.atleast_2d(np.concatenate((model.x.value.ravel(), model.y.value.ravel())))

def get_latent(model):
    return np.atleast_2d(model.w.value)

def generate_synthetic(model, size=100):
    observed, latent = get_observed(model), get_latent(model)
    # print("size: ", size)
    for i in range(size-1):
        model.draw_from_prior()
        observed = np.vstack((observed, get_observed(model)))
        latent = np.vstack((latent, get_latent(model)))
    return observed, latent

def _iterate_minibatches(inputs, outputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield Variable(torch.FloatTensor(inputs[excerpt])), Variable(torch.FloatTensor(outputs[excerpt]))



# Very Complicated Training Loop... Need to modify this to work with our model:

def training_step(optimizer, model, gen_data, dataset_size, batch_size, max_local_iters=10, misstep_tolerance=0, verbose=False):
    """ Training function for fitting density estimator to simulator output.
        Adapted to use conditional, real NVP density estimator. """
    # Train
    synthetic_ins, synthetic_outs = gen_data(dataset_size)
    validation_size = int(dataset_size/10)
    validation_ins, validation_outs = [Variable(torch.FloatTensor(t)) for t in gen_data(validation_size)]
    missteps = 0
    num_batches = float(dataset_size)/batch_size

    validation_err = -torch.mean(model.inverse(validation_ins, validation_outs)[1]).detach().item()
    for local_iter in range(max_local_iters):
        # print(local_iter)
        train_err = 0
        for inputs, outputs in _iterate_minibatches(synthetic_ins, synthetic_outs, batch_size):
            # print("taking step!")
            optimizer.zero_grad()
            Z, log_probs = model.inverse(inputs, outputs)
            loss = -torch.mean(log_probs)
            print("STEP LOSS: ", loss.detach().item())
            loss.backward()
            optimizer.step()
            train_err += loss.item()/num_batches

        next_validation_err = -torch.mean(model.inverse(validation_ins, validation_outs)[1]).item()
        if next_validation_err > validation_err:
            missteps += 1
        validation_err = next_validation_err
        if missteps > misstep_tolerance:
            break

    if verbose:
        print(train_err, validation_err, "(", local_iter+1, ")")

    return train_err, validation_err, local_iter+1



M_train = pymc.Model(robust_regression(None, None))
gen_data = lambda num_samples: generate_synthetic(M_train, num_samples)

# This is where we insert our real_nvp model:

observed_dim = num_points*2
latent_dim = 3

def gen_example_pair(model):
    model.draw_from_prior()
    data_x = model.X.value
    data_y = model.y.value
    true_w = model.w.value
    return data_x, data_y, true_w

def estimate_MCMC(data_x, data_y, ns, iters=10000, burn=0.5):
    """ MCMC estimate of weight distribution """
    mcmc_est = pymc.MCMC(robust_regression(data_x[:,1:2], data_y))
    mcmc_est.sample(iters, burn=burn*iters, thin=np.ceil(burn*iters/ns))
    trace_w = mcmc_est.trace('w').gettrace()[:ns]
    return trace_w

def estimate_NN(network, data_x, data_y, ns):
    """ NN proposal density for weights """
    nn_input = Variable(torch.FloatTensor(np.concatenate((data_x[:,1], data_y[:]))))
    print(nn_input.size())
    nn_input = nn_input.unsqueeze(0).repeat(ns,1)
    print(nn_input)
    print(nn_input.size())
    values, log_q = network.forward(o=nn_input)
    return values.cpu().detach().numpy(), log_q.squeeze().cpu().detach().numpy()

def sample_prior_proposals(model, ns):
    samples = []
    for n in range(ns):
        model.draw_from_prior()
        samples.append(model.w.value)
    return np.array(samples)

def compare_and_plot(row=0, ns=100, alpha=0.05, data_x=None, data_y=None, true_w=None, seed=12):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = pymc.Model(robust_regression(None, None))
    prior_proposals = sample_prior_proposals(model, ns*10)
    if data_x is None:
        data_x, data_y, true_w = gen_example_pair(model)
    mcmc_trace = estimate_MCMC(data_x, data_y, ns)
    nn_proposals, logq = estimate_NN(flow_model, data_x, data_y, ns*10)
    mcmc_mean = mcmc_trace.mean(0)
    nn_mean = nn_proposals.mean(0)

    print("True (generating) w:", true_w)
    print("MCMC weight mean:", mcmc_mean)
    print("NN weight proposal mean:", nn_mean)

    domain = np.linspace(min(data_x[:,1])-2, max(data_x[:,1])+2, 50)
    plt.subplot(3,4,1 + row*4)

    plt.plot(domain, mcmc_mean[0] + mcmc_mean[1]*domain + mcmc_mean[2]*domain**2, "b--", linewidth=2)
    for i in range(ns):
        plt.plot(domain, mcmc_trace[i,0] + mcmc_trace[i,1]*domain + mcmc_trace[i,2]*domain**2, "b-", linewidth=2, alpha=alpha)
    plt.plot(data_x[:,1], data_y, "k.", markersize=8)
    plt.xlim(np.min(domain),np.max(domain))
    limy = plt.ylim()
    if row == 0:
        plt.title("MH Posterior", fontsize=18)

    ax = plt.subplot(3,4,3 + row*4)
    plt.plot(domain, nn_mean[0] + nn_mean[1]*domain + nn_mean[2]*domain**2, "r--", linewidth=2)
    for i in range(ns):
        plt.plot(domain, nn_proposals[i,0] + nn_proposals[i,1]*domain  + nn_proposals[i,2]*domain**2, "r-", linewidth=2, alpha=alpha)
    plt.plot(data_x[:,1], data_y, "k.", markersize=8)
    if row == 0:
        plt.title("NF Proposal", fontsize=18)

    plt.ylim(limy)
    plt.xlim(min(domain),max(domain));
    ax.yaxis.set_ticklabels([])

    ax = plt.subplot(3,4,2 + row*4)
    prior_samples_mean = prior_proposals.mean(0)
    prior_proposals = prior_proposals[::10]
    plt.plot(domain, prior_samples_mean[0] + prior_samples_mean[1]*domain + prior_samples_mean[2]*domain**2, "c--", linewidth=2)
    for i in range(ns):
        plt.plot(domain, prior_proposals[i,0] + prior_proposals[i,1]*domain  + prior_proposals[i,2]*domain**2, "c-", linewidth=2, alpha=0.15)
    plt.plot(data_x[:,1], data_y, "k.", markersize=8)
    if row == 0:
        plt.title("Prior", fontsize=18)
    plt.ylim(limy)
    plt.xlim(min(domain),max(domain));
    ax.yaxis.set_ticklabels([])

    # compute NN-IS estimate
    logp = []
    nn_test_model = pymc.Model(robust_regression(data_x[:,1:2], data_y))
    for nnp in nn_proposals:
        nn_test_model.w.value = nnp
        try:
            next_logp = nn_test_model.logp
        except:
            next_logp = -np.Inf
        logp.append(next_logp)
    logp = np.array(logp)
    w = np.exp(logp - logq) / np.sum(np.exp(logp - logq))
    nnis_mean = np.sum(w*nn_proposals.T,1)
    print("NN-IS estimated mean:", nnis_mean)
    print("NN-IS ESS:", 1.0/np.sum(w**2), w.shape[0])

    ax = plt.subplot(3,4,4 + row*4)
    plt.plot(domain, nnis_mean[0] + nnis_mean[1]*domain + nnis_mean[2]*domain**2, "g--", linewidth=2)

    nn_resampled = nn_proposals[systematic_resample(np.log(w))][::10]
    for i in range(ns):
        plt.plot(domain, nn_resampled[i,0] + nn_resampled[i,1]*domain  + nn_resampled[i,2]*domain**2, "g-", linewidth=2, alpha=alpha)
    plt.plot(data_x[:,1], data_y, "k.", markersize=8)
    if row == 0:
        plt.title("IS Posterior", fontsize=18)
    plt.ylim(limy)
    plt.xlim(min(domain),max(domain));
    ax.yaxis.set_ticklabels([])

# Load the model:

pickle_in = open("saved_models/conditional_density_est.pkl","rb")
flow_model = pickle.load(pickle_in)
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
# generate four different plots.
f = plt.figure(figsize=(14,9))

compare_and_plot(row=0, seed=12)
compare_and_plot(row=1, seed=13)
compare_and_plot(row=2, seed=14)

plt.tight_layout()
f.savefig("conditional_est.pdf", bbox_inches='tight')

plt.show()
