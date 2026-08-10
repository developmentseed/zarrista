//! Helpers for building `__repr__` strings.
//!
//! A `__repr__` must never format a zarrs type with `Debug`. That prints the
//! Rust type name and its struct syntax, such as
//! ```text
//! ZstdCodec { compression: 3, checksum: false }
//! ```
//! which are implementation details of this crate.
//!
//! Build the string from the Zarr v3 name and configuration instead.
//!
//! Every value comes from Python's own `repr`, never from a Rust formatter.
//! Python then owns the syntax, so a configuration reads `{'level': 3}` rather
//! than the JSON `{"level":3}`, and a boolean reads `False` rather than `false`.
//! Rendering one value in Rust and another in Python would also mix quoting
//! styles inside a single repr.

use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

use crate::dtype::PyDataType;
use crate::metadata::PyConfiguration;
use crate::node::PyNodePath;

/// Build the repr of a type that Zarr v3 describes with a name and a configuration.
///
/// The result reads `Class('name', config={...})`. The name is absent for a
/// type that Zarr v3 does not name, and the configuration is absent when it
/// holds no entries.
pub(crate) fn named_config_repr(
    py: Python,
    class: &str,
    name: Option<Cow<'_, str>>,
    config: Option<PyConfiguration>,
) -> PyResult<String> {
    let mut parts = Vec::with_capacity(2);
    if let Some(name) = name {
        parts.push(PyString::new(py, &name).repr()?.to_string());
    }
    if let Some(config) = config {
        let config = config.into_pyobject(py)?;
        // An empty configuration adds nothing that the name does not already say.
        if config.is_truthy()? {
            parts.push(format!("config={}", config.repr()?));
        }
    }
    Ok(format!("{class}({})", parts.join(", ")))
}

/// Render `shape=(4, 4), dtype='int32'`, which every array-like repr ends with.
///
/// The data type shows its Zarr v3 name. A data type that Zarr v3 does not name
/// shows `None`, because no shorter description of it exists.
pub(crate) fn shape_and_dtype(py: Python, shape: &[u64], dtype: &PyDataType) -> PyResult<String> {
    let shape = PyTuple::new(py, shape)?.repr()?;
    let dtype = dtype.name().into_pyobject(py)?.repr()?;
    Ok(format!("shape={shape}, dtype={dtype}"))
}

/// Build the repr of a decoded array, which reads `Class(shape=(4, 4), dtype='int32')`.
pub(crate) fn tensor_repr(
    py: Python,
    class: &str,
    shape: &[u64],
    dtype: &PyDataType,
) -> PyResult<String> {
    Ok(format!("{class}({})", shape_and_dtype(py, shape, dtype)?))
}

/// Build the repr of an array, which reads `Class(path='/a', shape=(4, 4), dtype='int32')`.
///
/// The path comes first, because it is what tells two arrays of one store apart.
pub(crate) fn array_repr(
    py: Python,
    class: &str,
    path: &PyNodePath,
    shape: &[u64],
    dtype: &PyDataType,
) -> PyResult<String> {
    let path = path.repr(py)?;
    Ok(format!(
        "{class}(path={path}, {})",
        shape_and_dtype(py, shape, dtype)?
    ))
}
