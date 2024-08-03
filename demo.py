from optimal_psychophysics import optimal_psychophysics as op
import numpy as np
import time
import matplotlib.pyplot as plt

# Create a model
model = op.models.TwoParameterPsychometricModel(10.0, 4.0, 0.5, 3.0)
print(model)


# generate some fake data
def logistic(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))

# true parameters
a = 10.0
b = 0.5

# prior parameters
mu_a, sigma_a = 5.0, 0.5
mu_b, sigma_b = 3.0, 0.5

n_trials = 15

num_draws = 2500
num_tune = 1000

# Create the estimators
eig_estimator = op.eig.EnumeratedMonteCarloEstimator([False,True], 5_0000)
candidates = np.linspace(-0.3, 1.2, 20).reshape(-1, 1)
posterior_estimator = op.posterior.NUTSEstimator(num_draws=num_draws, num_tune=num_tune, num_chains=4)

X = np.empty((0,1))
Y = np.empty((0,)).astype(bool)


for trial in range(n_trials):

    # compute eig for candidate sampling points
    print(f"Model parameters: mu_a = {mu_a}, sigma_a = {sigma_a}, mu_b = {mu_b}, sigma_b = {sigma_b}")
    eig_model = op.models.TwoParameterPsychometricModel(mu_a, sigma_a, mu_b, sigma_b)
    eig = eig_estimator.estimate(eig_model, ["a","b"], candidates)

    # sample the next point
    pX = candidates[np.argmax(eig)].reshape(-1, 1)
    pY = np.random.binomial(1, logistic(pX, a, b)).astype(bool).reshape(-1,)

    print(f"Sampling point {pX}: {pY}")

    X = np.vstack((X, pX))
    Y = np.hstack((Y, pY))

    # estimate the posterior
    res = np.stack(posterior_estimator.estimate(model, X, Y))
    means = res[:, num_tune:].mean(axis=(0,1))
    stds = res[:, num_tune:].std(axis=(0,1))

    # update the the prior (i.e. the posterior becomes the new prior)
    mu_a, sigma_a = means[0], stds[0]
    mu_b, sigma_b = means[1], stds[1]



quit()

X = np.linspace(-0.3, 2.0, 10).reshape(-1, 1)
Y = np.random.binomial(1, logistic(X, a, b)).flatten().astype(bool)

# compute the log likelihood of the data wrt to a and b parameters
aa = np.linspace(0.0, 10.0, 100)
bb = np.linspace(-0.5, 2.0, 100)
AA, BB = np.meshgrid(aa, bb)

log_likelihood = [model.log_likelihood(np.array([a,b]), X, Y) for a, b in zip(AA.ravel(), BB.ravel())]
log_likelihood = np.array(log_likelihood).reshape(AA.shape)

log_posterior = [model.log_posterior(np.array([a,b]), X, Y) for a, b in zip(AA.ravel(), BB.ravel())]
log_posterior = np.array(log_posterior).reshape(AA.shape)

# OED


# Use the estimator to sample the model


t0 = time.time()

t1 = time.time()

# plot the data and the likelihood for the data
fig, axs = plt.subplots(1, 4)
axs[0].plot(X, Y, 'o')
axs[1].contourf(AA, BB, np.exp(log_likelihood), levels=100)
axs[1].set_xlabel('a')
axs[1].set_ylabel('b')
axs[2].contourf(AA, BB, np.exp(log_posterior), levels=100)
axs[2].set_xlabel('a')
axs[2].set_ylabel('b')
axs[3].plot(candidates, eig)


plt.show()



# Create an estimator

estimator = op.posterior.NUTSEstimator(num_draws=num_draws, num_tune=num_tune, num_chains=4)

# Use the estimator to sample the model
t0 = time.time()
res = estimator.estimate(model, X, Y)

t1 = time.time()

# stack the chains
res = np.stack(res)

print(res.shape)

# print mean and std of the posterior (skip the first 400 samples)
print(res[:, num_tune:].mean(axis=(0,1)))
print(res[:, num_tune:].std(axis=(0,1)))

print(f"Posterior estimation took {t1 - t0} seconds")
