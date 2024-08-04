# Optimal Adaptive Psychometrics

This package provides a Rust implementation of different ways to obtain  estimates for real-time adaptive psychometric testing. This includes parameter estimation and Expected Information Gain (EIG) calculations to select the next stimulus or item to present to the test taker.

# Goals
- Speed: High perfomance for small sample sizes and real-time applications ideally using SIMD instructions where possible
- Correctness: Accurate and reliable estimates
- Simplicity: Prioritise simplicity and readability over flexibility and scalability
- Portability: Should work on most platforms (even those that don't allow JIT compilation)

# Non-goals
- Scalability: Not designed for big datasets (i.e., no support for GPUs or distributed computing)
- Flexibility: Not designed for complex models or (exploratory) data analysis - you should probbaly use PyMC3 or Stan instead

# (Planned) Features
- [ ] Posterior estimation
  - [x] using Hamiltonian Monte Carlo (HMC) using no-U-turn sampling (using the [`nuts-rs`](https://docs.rs/nuts-rs/latest/nuts_rs/) crate)
  - [ ] using grid approximation
  - [ ] using Variational Inference methods (maybe)
- [ ] Expected Information Gain (EIG) calculations
  - [x] using a Rao-Blackwellized Monte Carlo estimator (for cases where outcomes can be enumerated)
  - [ ] using a Nested Monte Carlo estimator
  - [ ] using grid approximation
  - [ ] using a Laplace approximation
  - [ ] using Variational Inference methods (maybe)

# Short-term roadmap
- [ ] Make the base psychometric model more flexible (e.g., allow for different link functions, prior distributions, snd multivariate outcomes/designs)
- [ ] Allow better vectorisation using `ndarray` (will mainly be helpful when calling from Python with `pyo3`)
- [ ] Implement support for [Arviz](https://python.arviz.org/en/stable/) for posterior diagnostics on the Python side
- [ ] Use [Enyme](https://github.com/EnzymeAD/rust) instead of manually deriving gradients (this currently requires building the Rust compiler from source)


# Glossary
- **Outcome**: A possible result or state that can arise from an experiment or random process in a probabilistic model. For example, in a psychophysics experiment studying reaction times to visual stimuli, an outcome could be the specific time (e.g., 350 milliseconds) it takes for a participant to respond to a stimulus.
- **Design**: The set of deterministic inputs or predictors used in a model to explain or predict the outcome variable. These inputs are the variables that provide the framework for analyzing the relationship between them and the dependent variable in a statistical model, such as in linear regression. For example, in a psychophysics experiment, the design could include the luminance of a visual stimulus, the contrast of the stimulus, and the orientation of the stimulus.
- **Parameter**: A quantity that defines a statistical model. Parameters are estimated from data and used to make inferences about the population from which the data were sampled. For example, in a psychometric model, parameters could include the threshold at which a participant can detect a stimulus, the slope of the psychometric function, and the guessing rate of the participant.
