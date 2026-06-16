use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::{Borrowed, FromPyObject};
use zarrs::node::NodePath;

/// A [`NodePath`] extractable from a Python string (e.g. `"/group/array"`).
pub struct PyNodePath(NodePath);

impl PyNodePath {
    pub fn root() -> Self {
        Self(NodePath::root())
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyNodePath {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let path = obj.extract::<PyBackedStr>()?;
        NodePath::new(&path)
            .map(PyNodePath)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl From<PyNodePath> for NodePath {
    fn from(py_path: PyNodePath) -> Self {
        py_path.0
    }
}

impl From<NodePath> for PyNodePath {
    fn from(path: NodePath) -> Self {
        Self(path)
    }
}

impl AsRef<NodePath> for PyNodePath {
    fn as_ref(&self) -> &NodePath {
        &self.0
    }
}
