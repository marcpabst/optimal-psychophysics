use crate::{
    dists::{ContinuousUnivariateDistribution, DiscreteUnivariateDistribution, Samplable},
    model::PsychometricModel,
};
use pyo3::prelude::*;
use rand_distr::{Bernoulli, Normal};

/** A two-parameter psychometric model. The model is defined as:

     k ~ Normal(k_mu, k_sigma)
     m ~ Normal(m_mu, m_sigma)
     y ~ BernoulliLogit(k * (x - m))

 where:
 - `k` is the slope of the psychometric function,
 - `m` is the location of the psychometric function,
 - `y` is the binary response,
**/
#[pyclass]
#[derive(Debug, Clone)]
pub struct TwoParameterPsychometricModel {
    // priors
    k_prior: Normal<f64>, // k ~ Normal(mu_k, sigma_k)
    m_prior: Normal<f64>, // m ~ Normal(mu_m, sigma_m)
}

impl TwoParameterPsychometricModel {
    pub fn new(mu_k: f64, sigma_k: f64, mu_m: f64, sigma_m: f64) -> Self {
        Self {
            k_prior: Normal::new(mu_k, sigma_k).unwrap(),
            m_prior: Normal::new(mu_m, sigma_m).unwrap(),
        }
    }
}

#[pymethods]
impl TwoParameterPsychometricModel {
    #[new]
    pub fn new_py(mu_k: f64, sigma_k: f64, mu_m: f64, sigma_m: f64) -> Self {
        Self::new(mu_k, sigma_k, mu_m, sigma_m)
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

impl PsychometricModel for TwoParameterPsychometricModel {
    // The Priors

    fn log_prior(&self, params: &[f64]) -> f64 {
        self.k_prior.logp(params[0]) + self.m_prior.logp(params[1])
    }

    fn log_prior_with_grad(&self, params: &[f64], grad: &mut [f64]) -> f64 {
        grad[0] += (-params[0] + self.k_prior.mean()) / self.k_prior.std_dev().powi(2);
        grad[1] += (-params[1] + self.m_prior.mean()) / self.m_prior.std_dev().powi(2);

        self.log_prior(params)
    }

    // The Likelihood

    #[allow(non_snake_case)]
    fn log_likelihood(&self, params: &[f64], design: &[f64], observations: f64) -> f64 {
        let k = params[0];
        let m = params[1];

        let x = design[0];
        let y = observations;

        let p = 1.0 / (1.0 + (-k * (m - x)).exp());

        Bernoulli::logpmf(&[p], y == 1.0)
    }

    #[allow(non_snake_case)]
    fn log_likelihood_with_grad(
        &self,
        params: &[f64],
        grad: &mut [f64],
        design: &[f64],
        observations: f64,
    ) -> f64 {
        let k = params[0];
        let m = params[1];

        let x = design[0];
        let y = observations;

        let grad_k = (-m * (k * m).exp() * (y)
            - m * (k * x * (y)
                + m * (k * x).exp()
                + (k * m).exp() * x * (y)
                + (k * x).exp() * x * (y)
                - (k * x).exp() * x)
                .exp())
            / ((k * m).exp() + (k * x).exp());
        let y = y;
        let grad_m = k * (-(-k * m).exp() * y - (-k * x).exp() * y + (-k * x).exp())
            / ((-k * m).exp() + (-k * x).exp());

        grad[0] += grad_k;
        grad[1] += grad_m;

        self.log_likelihood(params, design, observations)
    }

    // The Posterior
    fn log_posterior(&self, params: &[f64], design: &[f64], observations: f64) -> f64 {
        self.log_prior(params) + self.log_likelihood(params, design, observations)
    }

    fn log_posterior_with_grad(
        &self,
        params: &[f64],
        grad: &mut [f64],
        design: &[f64],
        observations: f64,
    ) -> f64 {
        let log_prior_grad = self.log_prior_with_grad(params, grad);
        let log_likelihood_grad = self.log_likelihood_with_grad(params, grad, design, observations);

        log_prior_grad + log_likelihood_grad
    }

    fn n_params(&self) -> usize {
        2
    }

    fn sample_prior<R: rand::Rng>(&self, rng: &mut R) -> Vec<f64> {
        vec![self.k_prior.sample(rng), self.m_prior.sample(rng)]
    }
}

impl Samplable<[f64; 2]> for TwoParameterPsychometricModel {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> [f64; 2] {
        [self.k_prior.sample(rng), self.m_prior.sample(rng)]
    }
}
