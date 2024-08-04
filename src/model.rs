use ndarray as nd;
#[allow(dead_code)]
use ndarray::ArrayBase;
use ndarray::Data;
use ndarray::Dim;
use numpy::ndarray::Array1;
use numpy::ndarray::Array2;
use nuts_rs::CpuLogpFunc;
use nuts_rs::CpuMath;
use nuts_rs::LogpError;
use rand::Rng;
use thiserror::Error;

pub type ArrayBase1<S> = ArrayBase<S, ndarray::Ix1>;
pub type ArrayBase2<S> = ArrayBase<S, ndarray::Ix2>;

// The density might fail in a recoverable or non-recoverable manner...
#[derive(Debug, Error)]
pub enum PosteriorLogpError {}
impl LogpError for PosteriorLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

pub trait PsychometricModel: Send + Sync + Clone + 'static {
    fn n_params(&self) -> usize;

    /// Return the names of the parameters.
    fn param_names(&self) -> Vec<&str>;

    fn param_index(&self, name: &str) -> Option<usize> {
        self.param_names().iter().position(|&x| x == name)
    }

    fn log_prior(&self, params: &[f64]) -> f64;
    fn log_prior_with_grad(&self, params: &[f64], grad: &mut [f64]) -> f64;

    /// Compute the log likelihood of the model given the parameters, the design and an observation.
    fn log_likelihood(&self, params: &[f64], design: &[f64], observation: bool) -> f64;

    /// Vectorized version of the log likelihood.
    fn log_likelihood_vec<S: Data<Elem = f64>, T: Data<Elem = bool>>(
        &self,
        params: &[f64],
        design: &ArrayBase2<S>,
        observations: &ArrayBase1<T>,
    ) -> f64 {
        let mut logl = 0.0;

        for (x, y) in design.outer_iter().zip(observations.into_iter()) {
            let x = x.as_slice().unwrap();
            logl += self.log_likelihood(params, x, *y);
        }

        logl
    }

    /// Compute the log likelihood of the model given the parameters, the design and an observation.
    /// Also compute the gradient of the log likelihood with respect to the parameters.
    fn log_likelihood_with_grad(
        &self,
        params: &[f64],
        grad: &mut [f64],
        design: &[f64],
        observations: bool,
    ) -> f64;

    /// Vectorized version of the log likelihood with gradient. An implementation is provided
    /// by default, but can be overriden for performance reasons.
    #[allow(non_snake_case)]
    fn log_likelihood_with_grad_vec<S: Data<Elem = f64>, T: Data<Elem = bool>>(
        &self,
        params: &[f64],
        grad: &mut [f64],
        design: &ArrayBase2<S>,
        observations: &ArrayBase1<T>,
    ) -> f64 {
        let mut logl = 0.0;
        // convert arrays into 1d arrays
        let X = design;
        let Y = observations;

        for (i, (x, y)) in X.outer_iter().zip(Y.into_iter()).enumerate() {
            // println!("i={}, x={:?}, y={}", i, x, y);
            let x = x.as_slice().unwrap();
            logl += self.log_likelihood_with_grad(params, grad, x, *y);
        }

        logl
    }

    /// Compute the log posterior of the model given the parameters, the design and an observation.
    fn log_posterior(&self, params: &[f64], design: &[f64], observations: bool) -> f64 {
        self.log_prior(params) + self.log_likelihood(params, design, observations)
    }

    /// Vectorized version of the log posterior.
    fn log_posterior_vec<S: Data<Elem = f64>, T: Data<Elem = bool>>(
        &self,
        params: &[f64],
        design: &ArrayBase2<S>,
        observations: &ArrayBase1<T>,
    ) -> f64 {
        self.log_likelihood_vec(params, design, observations) + self.log_prior(params)
    }

    /// Compute the log posterior of the model given the parameters, the design and an observation.
    /// Also compute the gradient of the log posterior with respect to the parameters.
    fn log_posterior_with_grad(
        &self,
        params: &[f64],
        grad: &mut [f64],
        design: &[f64],
        observations: bool,
    ) -> f64 {
        let log_likelihood_grad = self.log_likelihood_with_grad(params, grad, design, observations);
        let log_prior_grad = self.log_prior_with_grad(params, grad);

        log_prior_grad + log_likelihood_grad
    }

    /// Vectorized version of the log posterior with gradient. An implementation is provided
    /// by default, but can be overriden for performance reasons.
    fn log_posterior_with_grad_vec<S: Data<Elem = f64>, T: Data<Elem = bool>>(
        &self,
        params: &[f64],
        grad: &mut [f64],
        design: &ArrayBase2<S>,
        observations: &ArrayBase1<T>,
    ) -> f64 {
        let log_prior_grad = self.log_prior_with_grad(params, grad);
        let log_likelihood_grad =
            self.log_likelihood_with_grad_vec(params, grad, design, observations);

        log_likelihood_grad + log_prior_grad
    }

    /// Sample from the prior distribution of the model.
    fn sample_prior<R: Rng>(&self, rng: &mut R) -> Vec<f64>;

    /// Sample from the likelihood of the model given the parameters and the design.
    fn sample_likelihood<R: Rng, S: Data<Elem = f64>>(
        &self,
        rng: &mut R,
        params: &[f64],
        design: &ArrayBase1<S>,
    ) -> Vec<bool>;

    /// Sample from the prior predictive distribution of the model.
    fn sample_prior_predictive<R: Rng, S: Data<Elem = f64>>(
        &self,
        rng: &mut R,
        design: &ArrayBase1<S>,
    ) -> Vec<bool> {
        let params = self.sample_prior(rng);
        self.sample_likelihood(rng, &params, design)
    }

    /// Combine the model with data to create a new `PsychometricModelWithData`
    /// that can be used for inference.
    fn with_data<S: Data<Elem = f64>, T: Data<Elem = bool>>(
        &self,
        design: &ArrayBase2<S>,
        observations: &ArrayBase1<T>,
    ) -> PsychometricModelWithData<Self> {
        PsychometricModelWithData::new(self.clone(), design.to_owned(), observations.to_owned())
    }
}

#[derive(Clone)]
pub struct PsychometricModelWithData<M: PsychometricModel> {
    pub model: M,
    pub design: Array2<f64>,
    pub observations: Array1<bool>,
}

impl<M: PsychometricModel> PsychometricModelWithData<M> {
    pub fn new(model: M, design: Array2<f64>, observations: Array1<bool>) -> Self {
        Self {
            model,
            design,
            observations,
        }
    }
}

impl<M: PsychometricModel> std::ops::Deref for PsychometricModelWithData<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

impl<M: PsychometricModel> CpuLogpFunc for PsychometricModelWithData<M> {
    type LogpError = PosteriorLogpError;

    fn dim(&self) -> usize {
        self.model.n_params()
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        // zero out the gradient
        grad.iter_mut().for_each(|x| *x = 0.0);

        Ok(self
            .model
            .log_posterior_with_grad_vec(position, grad, &self.design, &self.observations))
    }
}

impl<M: PsychometricModel> nuts_rs::Model for PsychometricModelWithData<M> {
    type Math<'model> = CpuMath<Self> where Self: 'model;
    type DrawStorage<'model, S: nuts_rs::Settings> = crate::storage::ArrowStorage where Self: 'model;

    fn new_trace<'model, S: nuts_rs::Settings, R: Rng + ?Sized>(
        &'model self,
        _rng: &mut R,
        _chain_id: u64,
        _settings: &'model S,
    ) -> anyhow::Result<Self::DrawStorage<'model, S>> {
        Ok(crate::storage::ArrowStorage::new(2))
    }

    fn math(&self) -> anyhow::Result<Self::Math<'_>> {
        Ok(CpuMath::new(self.clone()))
    }

    fn init_position<R: Rng + ?Sized>(
        &self,
        _rng: &mut R,
        position: &mut [f64],
    ) -> anyhow::Result<()> {
        position.iter_mut().for_each(|x| *x = 0.0001);
        Ok(())
    }
}
