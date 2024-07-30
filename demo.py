from optimal_psychophysics import optimal_psychophysics as op
import numpy as np
import time
import matplotlib.pyplot as plt

# Create a model
model = op.models.TwoParameterPsychometricModel(5.0, 1.0, 0.5, 1.0)
print(model)

# Create same fake data
X = np.array([[-100.0, -50.0, 50.0, 100.0]]).astype(float)
Y = np.array([1, 1, 0,0]).astype(bool)

# Create an estimator
estimator = op.posterior.NUTSEstimator(num_draws=1000, num_tune=500, num_chains=1)

# Use the estimator to sample the model
t0 = time.time()
res = estimator.estimate(model, X, Y)

t1 = time.time()

# stack the chains
res = np.stack(res)

# print mean and std of the posterior (skip the first 400 samples)
print(res[:, 500:].mean(axis=(0,1)))
print(res[:, 500:].std(axis=(0,1)))

print(f"Posterior estimation took {t1 - t0} seconds")

# OED
# Create an estimator
estimator = op.eig.EnumeratedMonteCarloEstimator([0,1], 5000)

# Use the estimator to sample the model
candidates = np.linspace(-1, 1, 10).reshape(-1, 1)

t0 = time.time()
eig = estimator.estimate(model, candidates)
t1 = time.time()

ax, fig = plt.subplots()

print(f"OED took {t1 - t0} seconds")

plt.plot(candidates, eig)

plt.show()
