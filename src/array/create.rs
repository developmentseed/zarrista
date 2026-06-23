use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use zarrs::array::ArrayBuilder;

use crate::array::{PyArray, PyAsyncArray};
use crate::error::ZarristaResult;

#[pyclass(module = "zarrista.array", frozen, name = "Config")]
pub struct PyArrayBuilder(ArrayBuilder);

#[pymethods]
impl PyArrayBuilder {
    #[staticmethod]
    fn like<'py>(array: Bound<'py, PyAny>) -> ZarristaResult<Self> {
        if let Ok(array) = array.cast::<PyArray>() {
            Ok(Self(ArrayBuilder::from_array(array.get().inner())))
        } else if let Ok(array) = array.cast::<PyAsyncArray>() {
            Ok(Self(ArrayBuilder::from_array(array.get().inner())))
        } else {
            Err(PyTypeError::new_err(format!(
                "expected an Array or AsyncArray, got {}",
                array.get_type().name()?
            ))
            .into())
        }
    }
}
