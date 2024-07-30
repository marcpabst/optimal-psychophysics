use pyo3::{types::PyAnyMethods, Bound, PyAny};

use crate::{model::PsychometricModel, two_param_psychometric::TwoParameterPsychometricModel};

pub fn try_extract_model<'py>(
    model: &Bound<'py, PyAny>,
) -> Option<impl PsychometricModel + 'static> {
    let model = model.extract::<TwoParameterPsychometricModel>();
    match model {
        Ok(model) => Some(model),
        Err(_) => None,
    }
}
