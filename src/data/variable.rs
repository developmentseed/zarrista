use std::sync::Arc;

use arrow_array::{ArrayRef, LargeBinaryArray, LargeStringArray};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::Field;
use bytes::Bytes;
use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList, PyString, PyTuple};
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::ffi::{to_array_pycapsules, to_schema_pycapsule};
use zarrs::array::DataType;
use zarrs::array::data_type::{BytesDataType, StringDataType};

use crate::dtype::PyDataType;

/// Variable-length data (string/bytes).
#[pyclass(module = "zarrista", frozen, name = "VariableArray")]
pub struct PyVariableArray {
    bytes: Bytes,
    offsets: Vec<usize>,
    data_type: DataType,
    shape: Arc<[u64]>,
}

crate::wasm_send_sync!(PyVariableArray);

impl PyVariableArray {
    pub fn new(bytes: Bytes, offsets: Vec<usize>, data_type: DataType, shape: Arc<[u64]>) -> Self {
        Self {
            bytes,
            offsets,
            data_type,
            shape,
        }
    }

    /// Build an Arrow array over this data. The values buffer is shared
    /// zero-copy from the `bytes::Bytes`; only the small offsets array is copied
    /// (zarrs `usize` → Arrow `i64`).
    fn to_arrow_array(&self) -> PyArrowResult<ArrayRef> {
        let values = Buffer::from(self.bytes.clone());
        let offsets = self
            .offsets
            .iter()
            .map(|&offset| i64::try_from(offset).expect("offset overflows i64"))
            .collect::<Vec<_>>();
        let scalar_buffer = ScalarBuffer::from(offsets);

        // Safety: Zarrs guarantees that the offsets are valid and monotonically increasing, and
        // that the final offset is within bounds of the values buffer.
        let offsets = unsafe { OffsetBuffer::new_unchecked(scalar_buffer) };

        if self.data_type.is::<StringDataType>() {
            Ok(Arc::new(LargeStringArray::try_new(offsets, values, None)?))
        } else if self.data_type.is::<BytesDataType>() {
            Ok(Arc::new(LargeBinaryArray::try_new(offsets, values, None)?))
        } else {
            Err(PyTypeError::new_err(format!(
                "Arrow export of variable-length data type {} is not supported",
                self.data_type
            ))
            .into())
        }
    }

    fn arrow_data_type(&self) -> PyResult<arrow_schema::DataType> {
        if self.data_type.is::<StringDataType>() {
            Ok(arrow_schema::DataType::LargeUtf8)
        } else if self.data_type.is::<BytesDataType>() {
            Ok(arrow_schema::DataType::LargeBinary)
        } else {
            Err(PyTypeError::new_err(format!(
                "Arrow export of variable-length data type {} is not supported",
                self.data_type
            )))
        }
    }

    /// The Arrow field describing [`Self::to_arrow_array`].
    fn arrow_field(&self) -> PyResult<Field> {
        Ok(Field::new("", self.arrow_data_type()?, false))
    }
}

#[pymethods]
impl PyVariableArray {
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyTuple>> {
        let array = self.to_arrow_array()?;
        let field = Arc::new(self.arrow_field()?);
        to_array_pycapsules(py, field, array.as_ref(), requested_schema)
    }

    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyArrowResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, &self.arrow_field()?)
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.data_type.clone().into()
    }

    #[getter]
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.data_type.is::<StringDataType>() {
            string_to_numpy(py, &self.bytes, &self.offsets, &self.shape)
        } else {
            Err(PyNotImplementedError::new_err(format!(
                "NumPy export of variable-length data type {} is not supported",
                self.data_type
            )))
        }
    }
}

/// Variable-length data with a validity mask. Skeleton.
#[pyclass(module = "zarrista", frozen, name = "MaskedVariableArray")]
pub struct PyMaskedVariableArray {
    #[expect(dead_code)]
    bytes: Bytes,
    #[expect(dead_code)]
    offsets: Vec<usize>,
    /// The mask is 1 byte per element where 0 = invalid/missing, non-zero = valid/present.
    #[expect(dead_code)]
    mask: Bytes,
    data_type: DataType,
    shape: Arc<[u64]>,
}

crate::wasm_send_sync!(PyMaskedVariableArray);

impl PyMaskedVariableArray {
    /// Construct a new PyMaskedVariableArray from the given bytes, offsets, mask, data type, and shape.
    pub fn new(
        bytes: Bytes,
        offsets: Vec<usize>,
        mask: Bytes,
        data_type: DataType,
        shape: Arc<[u64]>,
    ) -> Self {
        Self {
            bytes,
            offsets,
            mask,
            data_type,
            shape,
        }
    }
}

#[pymethods]
impl PyMaskedVariableArray {
    #[getter]
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.data_type.clone().into()
    }
}

/// Decode zarr bytes to a NumPy array with dtype `StringDType`
fn string_to_numpy<'py>(
    py: Python<'py>,
    bytes: &Bytes,
    offsets: &[usize],
    shape: &[u64],
) -> PyResult<Bound<'py, PyAny>> {
    let mut elements = Vec::with_capacity(offsets.len().saturating_sub(1));
    for window in offsets.windows(2) {
        let element = &bytes[window[0]..window[1]];
        let s = std::str::from_utf8(element)
            .map_err(|err| PyUnicodeDecodeError::new_err_from_utf8(py, element, err))?;
        elements.push(PyString::new(py, s));
    }

    let numpy = py.import(intern!(py, "numpy"))?;
    let string_dtype = numpy
        .getattr(intern!(py, "dtypes"))?
        .getattr(intern!(py, "StringDType"))?
        .call0()?;

    let flat = numpy.call_method1(
        intern!(py, "array"),
        (PyList::new(py, elements)?, string_dtype),
    )?;
    flat.call_method1(intern!(py, "reshape"), (shape,))
}
