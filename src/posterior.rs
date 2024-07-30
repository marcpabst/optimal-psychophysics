use std::time::Duration;

use arrow::array::{FixedSizeListArray, Float64Array};
use ndarray::{ArrayView1, ArrayView2};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use nuts_rs::{CpuLogpFunc, DiagGradNutsSettings, Sampler, SamplerWaitResult};
use pyo3::{pyclass, pymethods, Bound, PyAny, Python};

use crate::model::PsychometricModel;
use crate::utils::try_extract_model;

#[allow(dead_code)]
pub trait PosteriorEstimator {}

#[pyclass]
pub struct NUTSEstimator {
    settings: DiagGradNutsSettings,
    timeout: Duration,
}

impl PosteriorEstimator for NUTSEstimator {}

#[pymethods]
impl NUTSEstimator {
    #[new]
    pub fn new(
        num_tune: Option<u64>,
        num_draws: Option<u64>,
        num_chains: Option<usize>,
        maxdepth: Option<u64>,
        timeout: Option<f64>,
        max_energy_error: Option<f64>,
        seed: Option<u64>,
    ) -> Self {
        let mut settings = DiagGradNutsSettings::default();

        if let Some(num_tune) = num_tune {
            settings.num_tune = num_tune;
        }
        if let Some(num_draws) = num_draws {
            settings.num_draws = num_draws;
        }
        if let Some(num_chains) = num_chains {
            settings.num_chains = num_chains;
        }
        if let Some(maxdepth) = maxdepth {
            settings.maxdepth = maxdepth;
        }
        if let Some(max_energy_error) = max_energy_error {
            settings.max_energy_error = max_energy_error;
        }
        if let Some(seed) = seed {
            settings.seed = seed;
        } else {
            // random seed
            settings.seed = rand::random();
        }

        let timeout = timeout
            .map(Duration::from_secs_f64)
            .unwrap_or(Duration::from_secs(100));

        Self { settings, timeout }
    }

    #[pyo3(name = "estimate")]
    fn py_estimate<'py>(
        &self,
        py: Python<'py>,
        model: &Bound<'py, PyAny>,
        design: PyReadonlyArray2<'py, f64>,
        observations: PyReadonlyArray1<'py, bool>,
    ) -> Vec<Bound<'py, PyArray2<f64>>> {
        // try to extract the model

        let model = try_extract_model(model).expect("Invalid model");
        let design = design.as_array().to_owned();
        let observations = observations.as_array().to_owned();

        let samples = self.estimate(&model, design.view(), observations.view());

        Vec::from_iter(samples.into_iter().map(|s| s.into_pyarray_bound(py)))
    }
}

impl NUTSEstimator {
    pub fn estimate(
        &self,
        model: &impl PsychometricModel,
        design: ArrayView2<f64>,
        observations: ArrayView1<bool>,
    ) -> Vec<Array2<f64>> {
        let model_with_data = model.with_data(&design, &observations);
        let n_params = model_with_data.dim();

        let sampler = Sampler::new(model_with_data, self.settings, 4, None)
            .expect("Failed to create sampler");

        let trace = loop {
            match sampler.wait_timeout(self.timeout) {
                SamplerWaitResult::Trace(trace) => break trace,
                SamplerWaitResult::Timeout(_) => panic!("Timeout"),
                SamplerWaitResult::Err(err, _trace) => panic!("Error: {:?}", err),
            };
        };

        // convert into Array2<f64> without copying
        let result = trace
            .chains
            .iter()
            .map(|chain| {
                chain
                    .draws
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .unwrap()
                    .values()
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .map(|chain_draws| {
                Array2::from_shape_vec((chain_draws.len() / n_params, n_params), chain_draws)
                    .expect("Failed to create Array2")
            })
            .collect::<Vec<_>>();

        result
    }
}

#[pyclass]
pub struct NutsResult {
    pub trace: Array3<f64>, // (n_chains, n_draws, n_params)
}
