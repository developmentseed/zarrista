//! Parsing of Python selection objects — the argument to `__getitem__` and
//! `retrieve_array_subset` — into a shape-independent Rust representation.
//!
//! This module only models *what was written* inside the brackets. Normalizing
//! it against a concrete array shape (resolving negatives, expanding `Ellipsis`,
//! building an `ArraySubset`) is a separate, shape-aware step.

use std::ops::Range;

use pyo3::exceptions::{PyIndexError, PyNotImplementedError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyEllipsis, PySlice, PyTuple};
use pyo3::Borrowed;
use zarrs::array::ArraySubset;

use crate::error::ZarristaResult;

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
pub(crate) enum PySelection {
    Single(AxisSelector),
    Tuple(Vec<AxisSelector>),
}

impl PySelection {
    // TODO: this function is vibe coded, come back to this and clean it up.

    /// Resolve this selection against a concrete array `shape`, producing the
    /// `ArraySubset` to read. Negative indices and slice bounds are normalized,
    /// a single `Ellipsis` (or fewer-than-`ndim` selectors) expands to full
    /// axes, and an integer selects a length-1 range (the axis is retained).
    pub(crate) fn to_array_subset(&self, shape: &[u64]) -> ZarristaResult<ArraySubset> {
        let ndim = shape.len();
        let selectors: &[AxisSelector] = match self {
            PySelection::Single(sel) => std::slice::from_ref(sel),
            PySelection::Tuple(sels) => sels.as_slice(),
        };

        let n_ellipsis = selectors
            .iter()
            .filter(|s| matches!(s, AxisSelector::Ellipsis))
            .count();
        if n_ellipsis > 1 {
            return Err(
                PyIndexError::new_err("an index can only have a single ellipsis ('...')").into(),
            );
        }
        let n_explicit = selectors.len() - n_ellipsis;
        if n_explicit > ndim {
            return Err(PyIndexError::new_err(format!(
                "too many indices for array: array is {ndim}-dimensional, but {n_explicit} were indexed"
            ))
            .into());
        }
        let n_fill = ndim - n_explicit;

        let mut ranges: Vec<Range<u64>> = Vec::with_capacity(ndim);
        for sel in selectors {
            let axis = ranges.len();
            match sel {
                AxisSelector::Ellipsis => {
                    for d in 0..n_fill {
                        ranges.push(0..shape[axis + d]);
                    }
                }
                AxisSelector::Index(i) => {
                    let len = shape[axis] as i64;
                    let resolved = if *i < 0 { *i + len } else { *i };
                    if resolved < 0 || resolved >= len {
                        return Err(PyIndexError::new_err(format!(
                            "index {i} is out of bounds for axis {axis} with size {}",
                            shape[axis]
                        ))
                        .into());
                    }
                    ranges.push(resolved as u64..resolved as u64 + 1);
                }
                AxisSelector::Slice { start, stop, step } => {
                    if step.is_some_and(|s| s != 1) {
                        return Err(PyNotImplementedError::new_err(
                            "slices with a step other than 1 are not supported",
                        )
                        .into());
                    }
                    let len = shape[axis] as i64;
                    let norm = |v: i64| if v < 0 { (v + len).max(0) } else { v.min(len) };
                    let lo = start.map_or(0, norm);
                    let hi = stop.map_or(len, norm).max(lo);
                    ranges.push(lo as u64..hi as u64);
                }
            }
        }

        // Pad any remaining trailing axes (when there was no ellipsis).
        for &len in &shape[ranges.len()..] {
            ranges.push(0..len);
        }

        Ok(ArraySubset::new_with_ranges(&ranges))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PySelection {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(tuple) = obj.cast::<PyTuple>() {
            let mut axes = Vec::with_capacity(tuple.len());
            for item in tuple.iter() {
                axes.push(item.extract::<AxisSelector>()?);
            }
            return Ok(PySelection::Tuple(axes));
        }
        Ok(PySelection::Single(obj.extract::<AxisSelector>()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::{PyIndexError, PyNotImplementedError};

    #[test]
    fn extracts_integer_index() {
        Python::attach(|py| {
            let obj = py.eval(c"5", None, None).unwrap();
            let sel: PySelection = obj.extract().unwrap();
            assert_eq!(sel, PySelection::Single(AxisSelector::Index(5)));
        });
    }

    #[test]
    fn extracts_negative_integer_index() {
        Python::attach(|py| {
            let obj = py.eval(c"-1", None, None).unwrap();
            let sel: PySelection = obj.extract().unwrap();
            assert_eq!(sel, PySelection::Single(AxisSelector::Index(-1)));
        });
    }

    #[test]
    fn extracts_slice_with_start_stop() {
        Python::attach(|py| {
            let obj = py.eval(c"slice(0, 10)", None, None).unwrap();
            let sel: PySelection = obj.extract().unwrap();
            assert_eq!(
                sel,
                PySelection::Single(AxisSelector::Slice {
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
            let sel: PySelection = obj.extract().unwrap();
            assert_eq!(
                sel,
                PySelection::Single(AxisSelector::Slice {
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
            let sel: PySelection = obj.extract().unwrap();
            assert_eq!(
                sel,
                PySelection::Single(AxisSelector::Slice {
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
            let sel: PySelection = obj.extract().unwrap();
            assert_eq!(sel, PySelection::Single(AxisSelector::Ellipsis));
        });
    }

    #[test]
    fn extracts_tuple_of_mixed() {
        Python::attach(|py| {
            let obj = py.eval(c"(5, slice(0, 4), ...)", None, None).unwrap();
            let sel: PySelection = obj.extract().unwrap();
            assert_eq!(
                sel,
                PySelection::Tuple(vec![
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
            let sel: PySelection = obj.extract().unwrap();
            assert_eq!(sel, PySelection::Tuple(vec![]));
        });
    }

    #[test]
    fn rejects_nested_tuple() {
        Python::attach(|py| {
            let obj = py.eval(c"(5, (0, 1))", None, None).unwrap();
            let result: PyResult<PySelection> = obj.extract();
            assert!(result.is_err());
        });
    }

    #[test]
    fn rejects_none_as_newaxis() {
        Python::attach(|py| {
            let obj = py.eval(c"None", None, None).unwrap();
            let err = obj.extract::<PySelection>().unwrap_err();
            assert!(err.is_instance_of::<PyNotImplementedError>(py));
        });
    }

    #[test]
    fn rejects_bool_index() {
        Python::attach(|py| {
            // `bool` is an `int` subclass; it must not be read as an integer index.
            let obj = py.eval(c"True", None, None).unwrap();
            let err = obj.extract::<PySelection>().unwrap_err();
            assert!(err.is_instance_of::<PyNotImplementedError>(py));
        });
    }

    #[test]
    fn rejects_list_index() {
        Python::attach(|py| {
            let obj = py.eval(c"[1, 2, 3]", None, None).unwrap();
            let result: PyResult<PySelection> = obj.extract();
            assert!(result.is_err());
        });
    }

    // --- to_array_subset: shape resolution ---

    const SHAPE: &[u64] = &[9, 64, 100];

    fn subset(ranges: &[Range<u64>]) -> ArraySubset {
        ArraySubset::new_with_ranges(ranges)
    }

    fn slice(start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> AxisSelector {
        AxisSelector::Slice { start, stop, step }
    }

    #[test]
    fn resolves_int_with_trailing_axes_full() {
        let input = PySelection::Single(AxisSelector::Index(5));
        assert_eq!(
            input.to_array_subset(SHAPE).unwrap(),
            subset(&[5..6, 0..64, 0..100])
        );
    }

    #[test]
    fn resolves_negative_int() {
        let input = PySelection::Single(AxisSelector::Index(-1));
        assert_eq!(
            input.to_array_subset(SHAPE).unwrap(),
            subset(&[8..9, 0..64, 0..100])
        );
    }

    #[test]
    fn resolves_tuple_int_and_slice() {
        let input = PySelection::Tuple(vec![AxisSelector::Index(5), slice(Some(0), Some(4), None)]);
        assert_eq!(
            input.to_array_subset(SHAPE).unwrap(),
            subset(&[5..6, 0..4, 0..100])
        );
    }

    #[test]
    fn resolves_full_slice() {
        let input = PySelection::Single(slice(None, None, None));
        assert_eq!(
            input.to_array_subset(SHAPE).unwrap(),
            subset(&[0..9, 0..64, 0..100])
        );
    }

    #[test]
    fn resolves_negative_slice_start() {
        let input = PySelection::Single(slice(Some(-2), None, None));
        assert_eq!(
            input.to_array_subset(SHAPE).unwrap(),
            subset(&[7..9, 0..64, 0..100])
        );
    }

    #[test]
    fn resolves_empty_range_slice() {
        let input = PySelection::Single(slice(Some(5), Some(5), None));
        assert_eq!(
            input.to_array_subset(SHAPE).unwrap(),
            subset(&[5..5, 0..64, 0..100])
        );
    }

    #[test]
    fn resolves_ellipsis_in_middle() {
        let input = PySelection::Tuple(vec![
            AxisSelector::Index(5),
            AxisSelector::Ellipsis,
            AxisSelector::Index(3),
        ]);
        assert_eq!(
            input.to_array_subset(SHAPE).unwrap(),
            subset(&[5..6, 0..64, 3..4])
        );
    }

    #[test]
    fn resolves_single_ellipsis_all_full() {
        let input = PySelection::Single(AxisSelector::Ellipsis);
        assert_eq!(
            input.to_array_subset(SHAPE).unwrap(),
            subset(&[0..9, 0..64, 0..100])
        );
    }

    #[test]
    fn resolves_empty_tuple_all_full() {
        let input = PySelection::Tuple(vec![]);
        assert_eq!(
            input.to_array_subset(SHAPE).unwrap(),
            subset(&[0..9, 0..64, 0..100])
        );
    }

    /// The Python exception produced by a failed resolution, for type checks.
    fn resolve_err(input: &PySelection) -> PyErr {
        input.to_array_subset(SHAPE).unwrap_err().into()
    }

    #[test]
    fn out_of_bounds_positive_int_errors() {
        Python::attach(|py| {
            let input = PySelection::Single(AxisSelector::Index(9)); // len 9 -> max valid 8
            assert!(resolve_err(&input).is_instance_of::<PyIndexError>(py));
        });
    }

    #[test]
    fn out_of_bounds_negative_int_errors() {
        Python::attach(|py| {
            let input = PySelection::Single(AxisSelector::Index(-10)); // -10 + 9 = -1
            assert!(resolve_err(&input).is_instance_of::<PyIndexError>(py));
        });
    }

    #[test]
    fn slice_with_step_errors() {
        Python::attach(|py| {
            let input = PySelection::Single(slice(None, None, Some(2)));
            assert!(resolve_err(&input).is_instance_of::<PyNotImplementedError>(py));
        });
    }

    #[test]
    fn too_many_indices_errors() {
        Python::attach(|py| {
            let input = PySelection::Tuple(vec![
                AxisSelector::Index(0),
                AxisSelector::Index(0),
                AxisSelector::Index(0),
                AxisSelector::Index(0), // 4 indices for a 3-D array
            ]);
            assert!(resolve_err(&input).is_instance_of::<PyIndexError>(py));
        });
    }

    #[test]
    fn double_ellipsis_errors() {
        Python::attach(|py| {
            let input = PySelection::Tuple(vec![AxisSelector::Ellipsis, AxisSelector::Ellipsis]);
            assert!(resolve_err(&input).is_instance_of::<PyIndexError>(py));
        });
    }
}
