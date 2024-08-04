use crate::dists::{BernoulliLogit, Normal};
use crate::{
    dists::{ContinuousUnivariateDistribution, DiscreteUnivariateDistribution, Samplable},
    model::PsychometricModel,
};
use ndarray::Data;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand_distr::Bernoulli;

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
    pub fn py_new(mu_k: f64, sigma_k: f64, mu_m: f64, sigma_m: f64) -> Self {
        Self::new(mu_k, sigma_k, mu_m, sigma_m)
    }

    #[pyo3(name = "sample_prior")]
    pub fn py_sample_prior(&self) -> PyResult<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let samples = self.sample_prior(&mut rng);
        Ok(samples)
    }

    #[pyo3(name = "sample_prior_predictive")]
    pub fn py_sample_prior_predictive<'py>(
        &self,
        py: Python<'py>,
        design: PyReadonlyArray1<'py, f64>,
    ) -> Vec<bool> {
        let mut rng = rand::thread_rng();
        let design = design.as_array();

        self.sample_prior_predictive(&mut rng, &design.view())
    }

    #[pyo3(name = "log_likelihood")]
    pub fn py_log_likelihood(
        &self,
        params: Vec<f64>,
        design: PyReadonlyArray2<f64>,
        observations: PyReadonlyArray1<bool>,
    ) -> f64 {
        let design = design.as_array();
        let observations = observations.as_array();

        self.log_likelihood_vec(&params, &design.view(), &observations.view())
    }

    #[pyo3(name = "log_posterior")]
    pub fn py_log_posterior(
        &self,
        params: Vec<f64>,
        design: PyReadonlyArray2<f64>,
        observations: PyReadonlyArray1<bool>,
    ) -> f64 {
        let design = design.as_array();
        let observations = observations.as_array();

        self.log_posterior_vec(&params, &design.view(), &observations.view())
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

impl PsychometricModel for TwoParameterPsychometricModel {
    // The parameters
    fn param_names(&self) -> Vec<&str> {
        vec!["k", "m"]
    }

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
    fn log_likelihood(&self, params: &[f64], design: &[f64], observation: bool) -> f64 {
        let p_logit = params[0] * (params[1] - design[0]); // a * (b - x)
        BernoulliLogit::logpmf(&[p_logit], observation)
    }

    #[allow(non_snake_case)]
    fn log_likelihood_with_grad(
        &self,
        params: &[f64],
        grad: &mut [f64],
        design: &[f64],
        observation: bool,
    ) -> f64 {
        let k = params[0];
        let m = params[1];
        let x = design[0];
        let y = if observation { 0.0 } else { 1.0 };

        // (m - x)*(y + (y - 1)*exp(k*(m - x)))/(exp(k*(m - x)) + 1)
        grad[0] += (m - x) * (y + (y - 1.0) * (k * (m - x)).exp()) / ((k * (m - x)).exp() + 1.0);

        // k*(y*exp(k*m) + y*exp(k*x) - exp(k*m))/(exp(k*m) + exp(k*x))
        grad[1] += k * (y * (k * m).exp() + y * (k * x).exp() - (k * m).exp())
            / ((k * m).exp() + (k * x).exp());

        // return the log likelihood
        self.log_likelihood(params, design, observation)
    }

    // The Posterior
    fn log_posterior(&self, params: &[f64], design: &[f64], observations: bool) -> f64 {
        self.log_prior(params) + self.log_likelihood(params, design, observations)
    }

    fn log_posterior_with_grad(
        &self,
        params: &[f64],
        grad: &mut [f64],
        design: &[f64],
        observations: bool,
    ) -> f64 {
        let log_prior = self.log_prior_with_grad(params, grad);
        let log_likelihood = self.log_likelihood_with_grad(params, grad, design, observations);

        log_likelihood + log_prior
    }

    fn n_params(&self) -> usize {
        2
    }

    fn sample_prior<R: rand::Rng>(&self, rng: &mut R) -> Vec<f64> {
        vec![self.k_prior.sample(rng), self.m_prior.sample(rng)]
    }

    fn sample_likelihood<R: rand::Rng, S: ndarray::Data<Elem = f64>>(
        &self,
        rng: &mut R,
        params: &[f64],
        design: &crate::model::ArrayBase1<S>,
    ) -> Vec<bool> {
        let k = params[0];
        let m = params[1];

        let x = design[0];

        let p_logit = k * (x - m);

        let dist = BernoulliLogit::new(p_logit);

        vec![dist.sample(rng)]
    }
}
