// Tools for Bayesian Optimal Experimental Design
//

use ndarray::{s, Array1, Array3, ArrayView2};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::{pyclass, pymethods, Bound, PyAny, Python};

use crate::model::PsychometricModel;
use crate::utils::try_extract_model;

pub trait ExpectedInformationGainEstimator {
    /// Estimate the expected information gain of a design given a model and a set of candidate designs.
    fn estimate(
        &self,
        model: &impl PsychometricModel,
        params: Vec<&str>,
        candidate_designs: ArrayView2<f64>,
    ) -> Array1<f64>;
}

#[pyclass]
pub struct NestedMonteCarloEstimator {}

#[pyclass]
/// Rao- Blackwellized Monte Carlo estimator.
/// Used to estimate the expected information gain of a design using simple Monte Carlo
/// estimation while enumerating the outcomes of the experiment. This works well for
/// when the number of outcomes is discrete and small.
///
/// Assuming that we have a design D and a set of outcomes 𝒴 = {y_1, y_2, ..., y_N},
///
/// μ̂_N := Σ_{y ∈ 𝒴} (1/N) Σ_{n=1}^N p(y | θ_n, ξ) log p(y | θ_n, ξ) - p̂(y | ξ) log p̂(y | ξ)
/// with p̂(y | ξ) = (1/N) Σ_{n=1}^N p(y | θ_n, ξ)
pub struct EnumeratedMonteCarloEstimator {
    outcomes: Vec<bool>,
    n_samples: usize,
}

impl EnumeratedMonteCarloEstimator {
    /// Create a new EnumrMonte Carlo estimator.
    pub fn new(outcomes: Vec<bool>, n_samples: usize) -> Self {
        Self {
            outcomes,
            n_samples,
        }
    }
}

#[allow(non_snake_case)]
impl ExpectedInformationGainEstimator for EnumeratedMonteCarloEstimator {
    fn estimate(
        &self,
        model: &impl PsychometricModel,
        _params: Vec<&str>,
        candidate_designs: ArrayView2<f64>,
    ) -> Array1<f64> {
        let mut A = Array3::<f64>::zeros((
            self.outcomes.len(),
            candidate_designs.nrows(),
            self.n_samples,
        ));

        let mut B = Array2::<f64>::zeros((self.outcomes.len(), candidate_designs.nrows()));

        let mut eig = Array1::<f64>::zeros(candidate_designs.nrows());

        let mut rng = rand::thread_rng();

        // the full estimator is Σ_{y ∈ 𝒴} (1/N) Σ_{n=1}^N p(y | θ_n, ξ) log p(y | θ_n, ξ) - p̂(y | ξ) log p̂(y | ξ)
        // where p̂(y | ξ) = (1/N) Σ_{n=1}^N p(y | θ_n, ξ)
        let mut p_y_given_θ_n_and_ξ = Array3::<f64>::zeros((
            self.outcomes.len(),
            candidate_designs.nrows(),
            self.n_samples,
        ));

        let mut log_p_y_given_θ_n_and_ξ = Array3::<f64>::zeros((
            self.outcomes.len(),
            candidate_designs.nrows(),
            self.n_samples,
        ));

        for (i, design) in candidate_designs.outer_iter().enumerate() {
            for (j, y) in self.outcomes.iter().enumerate() {
                for n in 0..self.n_samples {
                    let design = design.as_slice().unwrap();

                    let params = model.sample_prior(&mut rng);
                    // compute the likelihood p(y | params_n, design)
                    let log_likelihood = model.log_likelihood(&params, design, *y);

                    let likelihood = log_likelihood.exp();
                    // write the likelihood to the array
                    p_y_given_θ_n_and_ξ[[j, i, n]] = likelihood;
                    log_p_y_given_θ_n_and_ξ[[j, i, n]] = log_likelihood;
                    // write the first term into eig
                    A[[j, i, n]] = likelihood * log_likelihood;
                }
                // compute p̂(y | ξ) = (1/N) Σ_{n=1}^N p(y | θ_n, ξ)
                let p̂_y_given_ξ = p_y_given_θ_n_and_ξ.slice(s![j, i, ..]).mean().unwrap();
                // mean of the first term
                B[[j, i]] = A.slice(s![j, i, ..]).mean().unwrap();
                // compute the second term
                let second_term = p̂_y_given_ξ.ln() * p̂_y_given_ξ;
                // compute the difference
                B[[j, i]] -= second_term;
            }
            // the eig is the sum across the outcomes
            eig[i] = B.slice(s![.., i]).sum();
        }
        eig
    }
}

#[pymethods]
impl EnumeratedMonteCarloEstimator {
    #[new]
    pub fn py_new(outcomes: Vec<bool>, n_samples: usize) -> Self {
        EnumeratedMonteCarloEstimator::new(outcomes, n_samples)
    }

    #[pyo3(name = "estimate")]
    fn py_estimate<'py>(
        &self,
        py: Python<'py>,
        model: &Bound<'py, PyAny>,
        params: Vec<String>,
        candidate_designs: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let model = try_extract_model(model).expect("Invalid model");
        let candidate_designs = candidate_designs.as_array();
        let params = params.iter().map(|x| x.as_str()).collect();
        let eig = self.estimate(&model, params, candidate_designs);
        eig.into_pyarray_bound(py).to_owned()
    }
}
