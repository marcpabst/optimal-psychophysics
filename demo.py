from optimal_psychophysics import optimal_psychophysics as op
import numpy as np
import time
import matplotlib.pyplot as plt

# Create a model
model = op.models.TwoParameterPsychometricModel(5.0, 3.0, 0.5, 3.0)
print(model)



# generate some fake data
def logistic(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))

def inv_logit(p):
    return np.log(p/(1-p))

a = 4.0
b = 0.5

# X = np.array([[-7.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 7.0]]).T
# Y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1]).astype(bool)

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
# Create an estimator
estimator = op.eig.EnumeratedMonteCarloEstimator([False,True], 50000)

# Use the estimator to sample the model
candidates = np.linspace(-0.3, 1.2, 10).reshape(-1, 1)

t0 = time.time()
eig = estimator.estimate(model, candidates)
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
num_draws = 2000
num_tune = 500
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
