mod dists;
mod eig;
mod model;
mod posterior;
mod storage;
mod two_param_psychometric;
mod utils;

use pyo3::{pymodule, types::PyModule, Bound, PyResult};
use two_param_psychometric::TwoParameterPsychometricModel;

#[pymodule]
fn optimal_psychophysics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let models = PyModule::new_bound(m.py(), "models")?;
    models.add_class::<TwoParameterPsychometricModel>()?;

    let posterior = PyModule::new_bound(m.py(), "posterior")?;
    posterior.add_class::<posterior::NUTSEstimator>()?;

    let eig = PyModule::new_bound(m.py(), "eig")?;
    eig.add_class::<eig::EnumeratedMonteCarloEstimator>()?;

    m.add_submodule(&models)?;
    m.add_submodule(&posterior)?;
    m.add_submodule(&eig)?;

    Ok(())
}
