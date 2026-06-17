//! Parsing of Python selection objects — the argument to `__getitem__` and
//! `retrieve_array_subset` — into a shape-independent Rust representation.
//!
//! This module only models *what was written* inside the brackets. Normalizing
//! it against a concrete array shape (resolving negatives, expanding `Ellipsis`,
//! building an `ArraySubset`) is a separate, shape-aware step.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyEllipsis, PySlice, PyTuple};
use pyo3::Borrowed;

/// A selector for a single axis, as written inside `[]`.
///
/// Values are captured verbatim; negative indices and slice bounds are *not* yet
/// normalized against the axis length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AxisSelector {
    /// An integer index, e.g. `5` or `-1`.
    Index(i64),
    /// A slice, e.g. `0:10`, `::`, or `:5`.
    Slice {
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    },
    /// `...` (Ellipsis).
    Ellipsis,
}

impl<'a, 'py> FromPyObject<'a, 'py> for AxisSelector {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if obj.is_none() {
            return Err(PyNotImplementedError::new_err(
                "None / np.newaxis indexing is not supported",
            ));
        }

        // `bool` is a subclass of `int`; reject it before integer extraction so
        // boolean indexing is not silently read as 0/1.
        if obj.cast::<PyBool>().is_ok() {
            return Err(PyNotImplementedError::new_err(
                "boolean indexing is not supported",
            ));
        }

        if obj.cast::<PyEllipsis>().is_ok() {
            return Ok(AxisSelector::Ellipsis);
        }

        if let Ok(slice) = obj.cast::<PySlice>() {
            return Ok(AxisSelector::Slice {
                start: slice.getattr("start")?.extract()?,
                stop: slice.getattr("stop")?.extract()?,
                step: slice.getattr("step")?.extract()?,
            });
        }

        Ok(AxisSelector::Index(obj.extract::<i64>()?))
    }
}

/// A full selection: either a single axis selector (`arr[5]`) or a tuple of them
/// (`arr[5, 0:10, ...]`). Nested tuples are rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PySelectionInput {
    Single(AxisSelector),
    Tuple(Vec<AxisSelector>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for PySelectionInput {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(tuple) = obj.cast::<PyTuple>() {
            let mut axes = Vec::with_capacity(tuple.len());
            for item in tuple.iter() {
                axes.push(item.extract::<AxisSelector>()?);
            }
            return Ok(PySelectionInput::Tuple(axes));
        }
        Ok(PySelectionInput::Single(obj.extract::<AxisSelector>()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyNotImplementedError;

    #[test]
    fn extracts_integer_index() {
        Python::attach(|py| {
            let obj = py.eval(c"5", None, None).unwrap();
            let sel: PySelectionInput = obj.extract().unwrap();
            assert_eq!(sel, PySelectionInput::Single(AxisSelector::Index(5)));
        });
    }

    #[test]
    fn extracts_negative_integer_index() {
        Python::attach(|py| {
            let obj = py.eval(c"-1", None, None).unwrap();
            let sel: PySelectionInput = obj.extract().unwrap();
            assert_eq!(sel, PySelectionInput::Single(AxisSelector::Index(-1)));
        });
    }

    #[test]
    fn extracts_slice_with_start_stop() {
        Python::attach(|py| {
            let obj = py.eval(c"slice(0, 10)", None, None).unwrap();
            let sel: PySelectionInput = obj.extract().unwrap();
            assert_eq!(
                sel,
                PySelectionInput::Single(AxisSelector::Slice {
                    start: Some(0),
                    stop: Some(10),
                    step: None,
                })
            );
        });
    }

    #[test]
    fn extracts_full_slice() {
        Python::attach(|py| {
            let obj = py.eval(c"slice(None)", None, None).unwrap();
            let sel: PySelectionInput = obj.extract().unwrap();
            assert_eq!(
                sel,
                PySelectionInput::Single(AxisSelector::Slice {
                    start: None,
                    stop: None,
                    step: None,
                })
            );
        });
    }

    #[test]
    fn extracts_strided_slice() {
        Python::attach(|py| {
            let obj = py.eval(c"slice(1, 20, 2)", None, None).unwrap();
            let sel: PySelectionInput = obj.extract().unwrap();
            assert_eq!(
                sel,
                PySelectionInput::Single(AxisSelector::Slice {
                    start: Some(1),
                    stop: Some(20),
                    step: Some(2),
                })
            );
        });
    }

    #[test]
    fn extracts_ellipsis() {
        Python::attach(|py| {
            let obj = py.eval(c"...", None, None).unwrap();
            let sel: PySelectionInput = obj.extract().unwrap();
            assert_eq!(sel, PySelectionInput::Single(AxisSelector::Ellipsis));
        });
    }

    #[test]
    fn extracts_tuple_of_mixed() {
        Python::attach(|py| {
            let obj = py.eval(c"(5, slice(0, 4), ...)", None, None).unwrap();
            let sel: PySelectionInput = obj.extract().unwrap();
            assert_eq!(
                sel,
                PySelectionInput::Tuple(vec![
                    AxisSelector::Index(5),
                    AxisSelector::Slice {
                        start: Some(0),
                        stop: Some(4),
                        step: None,
                    },
                    AxisSelector::Ellipsis,
                ])
            );
        });
    }

    #[test]
    fn empty_tuple_is_empty_selection() {
        Python::attach(|py| {
            let obj = py.eval(c"()", None, None).unwrap();
            let sel: PySelectionInput = obj.extract().unwrap();
            assert_eq!(sel, PySelectionInput::Tuple(vec![]));
        });
    }

    #[test]
    fn rejects_nested_tuple() {
        Python::attach(|py| {
            let obj = py.eval(c"(5, (0, 1))", None, None).unwrap();
            let result: PyResult<PySelectionInput> = obj.extract();
            assert!(result.is_err());
        });
    }

    #[test]
    fn rejects_none_as_newaxis() {
        Python::attach(|py| {
            let obj = py.eval(c"None", None, None).unwrap();
            let err = obj.extract::<PySelectionInput>().unwrap_err();
            assert!(err.is_instance_of::<PyNotImplementedError>(py));
        });
    }

    #[test]
    fn rejects_bool_index() {
        Python::attach(|py| {
            // `bool` is an `int` subclass; it must not be read as an integer index.
            let obj = py.eval(c"True", None, None).unwrap();
            let err = obj.extract::<PySelectionInput>().unwrap_err();
            assert!(err.is_instance_of::<PyNotImplementedError>(py));
        });
    }

    #[test]
    fn rejects_list_index() {
        Python::attach(|py| {
            let obj = py.eval(c"[1, 2, 3]", None, None).unwrap();
            let result: PyResult<PySelectionInput> = obj.extract();
            assert!(result.is_err());
        });
    }
}
