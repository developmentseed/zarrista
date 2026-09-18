use num_complex::Complex;
use pyo3::exceptions::{PyTypeError, PyUnicodeDecodeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3_bytes::PyBytes;
use pythonize::depythonize;
use zarrs::array::{DataType, FillValue, FillValueMetadata};

use crate::data::numpy_dtype_name;
use crate::dtype::{PyDataType, data_type_display};
use crate::error::{ZarristaError, ZarristaResult};
use crate::exceptions as exc;
use crate::metadata::PyFillValueMetadata;

#[derive(Debug, Clone)]
#[pyclass(module = "zarrista", frozen, name = "FillValue", from_py_object)]
pub struct PyFillValue {
    fill_value: FillValue,
    /// We hold a DataType to enable easy conversion to typed scalar representations
    dtype: DataType,
}

impl PyFillValue {
    pub fn new(fill_value: FillValue, dtype: DataType) -> Self {
        Self { fill_value, dtype }
    }

    pub(crate) fn inner(&self) -> &FillValue {
        &self.fill_value
    }

    pub fn into_inner(self) -> FillValue {
        self.fill_value
    }

    pub fn data_type(&self) -> &DataType {
        &self.dtype
    }
}

#[pymethods]
impl PyFillValue {
    #[new]
    #[pyo3(signature = (value, /, dtype))]
    fn py_new(value: PyFillValueInput, dtype: PyDataType) -> ZarristaResult<Self> {
        let dtype = dtype.into_inner();
        let fill_value = value.resolve(&dtype)?;
        Ok(Self { fill_value, dtype })
    }

    /// The data type that applies to these bytes
    #[getter]
    fn dtype(&self) -> PyDataType {
        self.dtype.clone().into()
    }

    #[getter]
    fn size(&self) -> usize {
        self.fill_value.size()
    }

    /// The Zarr v3 fill value metadata of this fill value.
    #[getter]
    fn metadata(&self) -> ZarristaResult<PyFillValueMetadata> {
        Ok(self.dtype.metadata_fill_value(&self.fill_value)?.into())
    }

    fn as_bytes(&self) -> &[u8] {
        self.fill_value.as_ne_bytes()
    }

    /// Return the fill value as a NumPy scalar.
    fn to_numpy<'py>(&self, py: Python<'py>) -> ZarristaResult<Bound<'py, PyAny>> {
        use zarrs::array::data_type::{BytesDataType, StringDataType};

        let numpy = py.import(intern!(py, "numpy"))?;
        let bytes = self.fill_value.as_ne_bytes();

        // A variable-length data type has no NumPy data type of a fixed size.
        // Its fill value bytes are the whole element.
        if self.dtype.is::<StringDataType>() {
            let string = std::str::from_utf8(bytes)
                .map_err(|err| PyUnicodeDecodeError::new_err_from_utf8(py, bytes, err))?;
            return Ok(numpy.call_method1(intern!(py, "str_"), (string,))?);
        }
        if self.dtype.is::<BytesDataType>() {
            return Ok(numpy.call_method1(
                intern!(py, "bytes_"),
                (pyo3::types::PyBytes::new(py, bytes),),
            )?);
        }

        // One fill value is one element, so read it as a one-element array and
        // take the only item. That gives a NumPy scalar, not an array.
        let name = numpy_dtype_name(&self.dtype)?;
        let flat = numpy.call_method1(
            intern!(py, "frombuffer"),
            (pyo3::types::PyBytes::new(py, bytes), name.as_ref()),
        )?;
        Ok(flat.get_item(0)?)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        // Show the fill value the way the user gave it, which is also the way
        // the metadata stores it. A data type that cannot describe its own fill
        // value falls back to the bytes.
        let value = match self.metadata() {
            Ok(metadata) => metadata.into_pyobject(py)?.repr()?.to_string(),
            // Use the Python bytes type, not our PyBytes adapter, to create the repr
            Err(_) => pyo3::types::PyBytes::new(py, self.fill_value.as_ne_bytes())
                .repr()?
                .to_string(),
        };
        // Show the Zarr v3 name, as every other repr does. The constructor
        // accepts that name, so this repr round-trips.
        let dtype = self.dtype.name_v3().into_pyobject(py)?.repr()?;
        Ok(format!("FillValue({value}, dtype={dtype})"))
    }

    fn __eq__(&self, other: &Bound<PyAny>) -> bool {
        // Equal bytes under different data types are different fill values.
        if let Ok(other) = other.cast::<Self>() {
            let other = other.get();
            self.fill_value == other.fill_value && self.dtype == other.dtype
        } else {
            false
        }
    }

    #[pyo3(signature = (other, /))]
    fn equals_all(&self, other: PyBytes) -> bool {
        self.fill_value.equals_all(other.as_ref())
    }
}

impl From<PyFillValue> for FillValue {
    fn from(py_fill_value: PyFillValue) -> Self {
        py_fill_value.fill_value
    }
}

#[derive(Debug, Clone, FromPyObject)]
pub struct PyFillValueInput<'py>(Bound<'py, PyAny>);

impl PyFillValueInput<'_> {
    pub fn resolve(&self, dtype: &DataType) -> ZarristaResult<FillValue> {
        // A fill value that is already resolved needs no conversion, and the
        // error below already names both data types.
        if let Ok(fill_value) = self.0.cast::<PyFillValue>() {
            let fill_value = fill_value.get();
            if fill_value.data_type() != dtype {
                return Err(exc::FillValueError::new_err(format!(
                    "fill value has data type '{}', but the array has data type '{}'",
                    data_type_display(fill_value.data_type()),
                    data_type_display(dtype)
                ))
                .into());
            }
            return Ok(fill_value.inner().clone());
        }

        self.convert(dtype)
            .map_err(|error| self.add_error_context(error, dtype))
    }

    /// Name the value and the data type, which the errors from below do not.
    fn add_error_context(&self, error: ZarristaError, dtype: &DataType) -> ZarristaError {
        let value = self
            .0
            .repr()
            .map_or_else(|_| "<unknown>".to_string(), |repr| repr.to_string());
        exc::FillValueError::new_err(format!(
            "cannot use {value} as a fill value of data type '{}': {error}",
            data_type_display(dtype)
        ))
        .into()
    }

    /// Convert the Python value into the bytes of one `dtype` element.
    fn convert(&self, dtype: &DataType) -> ZarristaResult<FillValue> {
        use zarrs::array::data_type::*;

        let fill_value = if dtype.is::<BoolDataType>() {
            FillValue::from(self.0.extract::<bool>()?)
        } else if dtype.is::<UInt8DataType>() {
            FillValue::from(self.0.extract::<u8>()?)
        } else if dtype.is::<UInt16DataType>() {
            FillValue::from(self.0.extract::<u16>()?)
        } else if dtype.is::<UInt32DataType>() {
            FillValue::from(self.0.extract::<u32>()?)
        } else if dtype.is::<UInt64DataType>() {
            FillValue::from(self.0.extract::<u64>()?)
        } else if dtype.is::<Int8DataType>() {
            FillValue::from(self.0.extract::<i8>()?)
        } else if dtype.is::<Int16DataType>() {
            FillValue::from(self.0.extract::<i16>()?)
        } else if dtype.is::<Int32DataType>() {
            FillValue::from(self.0.extract::<i32>()?)
        } else if dtype.is::<Int64DataType>() {
            FillValue::from(self.0.extract::<i64>()?)
        } else if dtype.is::<BFloat16DataType>() {
            FillValue::from(half::bf16::from_f64(self.0.extract()?))
        } else if dtype.is::<Float16DataType>() {
            FillValue::from(half::f16::from_f64(self.0.extract()?))
        } else if dtype.is::<Float32DataType>() {
            FillValue::from(self.0.extract::<f32>()?)
        } else if dtype.is::<Float64DataType>() {
            FillValue::from(self.0.extract::<f64>()?)
        } else if dtype.is::<BytesDataType>() {
            FillValue::from(self.0.extract::<PyBackedBytes>()?.as_ref())
        } else if dtype.is::<StringDataType>() {
            FillValue::from(self.0.extract::<PyBackedStr>()?.as_str())
        } else {
            // Otherwise: convert to Zarr v3 fill value metadata; let zarrs parse it
            dtype.fill_value_v3(&self.to_fill_value_metadata()?)?
        };

        Ok(fill_value)
    }

    /// Convert a Python value into Zarr v3 fill value metadata.
    ///
    /// Three inputs need more than a plain JSON conversion.
    ///
    /// **NumPy scalars.** A scalar such as `np.float32(1.5)` for a `float8_e4m3`
    /// array is not a JSON value, and it is not a `float` subclass either. Its
    /// `item()` method gives the equivalent Python scalar. Without `item()`, the
    /// complex branch below would accept it and give `[1.5, 0.0]`, which is not
    /// a scalar fill value. `np.float64` and `np.int64` do subclass `float` and
    /// `int`, so they also work without `item()`.
    ///
    /// **Non-finite floats.** JSON has no NaN or infinity. The spec writes these as
    /// the strings `"NaN"`, `"Infinity"` and `"-Infinity"`, which is what
    /// `FillValueMetadata::from` gives for an `f64`.
    ///
    /// **Complex numbers.** The spec writes a complex value as the two-element
    /// array `[real, imaginary]`.
    ///
    /// An object that is not JSON but that acts like a number, such as
    /// `decimal.Decimal`, goes through `__index__` or `__float__`. The data
    /// types that this method serves then accept the same inputs as the data
    /// types above, which extract those methods directly.
    fn to_fill_value_metadata(&self) -> ZarristaResult<FillValueMetadata> {
        let ob = &self.0;

        // Call .item on a NumPy scalar to get the Python value. Otherwise this stays as-is.
        let ob = ob
            .call_method0(intern!(ob.py(), "item"))
            .unwrap_or_else(|_| ob.clone());

        if let Ok(value) = ob.extract::<f64>()
            && !value.is_finite()
        {
            return Ok(FillValueMetadata::from(value));
        }

        // Generic JSON conversion
        if let Ok(metadata) = depythonize(&ob) {
            return Ok(metadata);
        }

        // An object that is not JSON, but that acts like a number. `__index__`
        // comes first, so that an integer stays an integer: an integer data
        // type rejects the metadata `3.0`.
        if let Ok(value) = ob.extract::<i64>() {
            return Ok(FillValueMetadata::from(value));
        }
        if let Ok(value) = ob.extract::<u64>() {
            return Ok(FillValueMetadata::from(value));
        }
        if let Ok(value) = ob.extract::<f64>() {
            return Ok(FillValueMetadata::from(value));
        }

        let Ok(complex) = ob.extract::<Complex<f64>>() else {
            let type_name = ob.get_type().name()?;
            return Err(PyTypeError::new_err(format!(
                "a value of type '{type_name}' is not a number or a JSON value"
            ))
            .into());
        };
        Ok(FillValueMetadata::from([
            FillValueMetadata::from(complex.re),
            FillValueMetadata::from(complex.im),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use zarrs::array::data_type;

    use super::*;

    /// Resolve `value` for `dtype`, where `value` is a Python expression.
    fn resolve(value: &std::ffi::CStr, dtype: &DataType) -> ZarristaResult<FillValue> {
        Python::attach(|py| {
            let obj = py.eval(value, None, None).unwrap();
            PyFillValueInput(obj).resolve(dtype)
        })
    }

    #[test]
    fn resolves_int_for_each_integer_width() {
        let fill_value = resolve(c"-9999", &data_type::int32()).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-9999i32).to_ne_bytes());
    }

    #[test]
    fn resolves_the_same_int_differently_per_data_type() {
        let as_int = resolve(c"-9999", &data_type::int32()).unwrap();
        let as_float = resolve(c"-9999", &data_type::float32()).unwrap();
        assert_ne!(as_int.as_ne_bytes(), as_float.as_ne_bytes());
        assert_eq!(as_float.as_ne_bytes(), (-9999f32).to_ne_bytes());
    }

    #[test]
    fn rejects_an_out_of_range_int() {
        assert!(resolve(c"300", &data_type::int8()).is_err());
    }

    #[test]
    fn resolves_float16_by_rounding_once() {
        // The value sits just above the midpoint of two float16 values. A
        // conversion through float32 would round it down to 1.0.
        let fill_value = resolve(c"1.0 + 2**-11 + 2**-30", &data_type::float16()).unwrap();
        assert_eq!(
            fill_value.as_ne_bytes(),
            half::f16::from_bits(0x3c01).to_ne_bytes()
        );
    }

    #[test]
    fn resolves_nan_through_the_metadata_path() {
        let fill_value = resolve(c"float('nan')", &data_type::float8_e4m3()).unwrap();
        assert_eq!(fill_value.size(), 1);
    }

    #[test]
    fn resolves_a_complex_value() {
        let fill_value = resolve(c"complex(1.5, -2.5)", &data_type::complex64()).unwrap();
        let mut expected = 1.5f32.to_ne_bytes().to_vec();
        expected.extend_from_slice(&(-2.5f32).to_ne_bytes());
        assert_eq!(fill_value.as_ne_bytes(), expected);
    }

    /// Build a Python `FillValue` object for `dtype` from the Python `value`.
    fn fill_value_object<'py>(
        py: Python<'py>,
        value: &std::ffi::CStr,
        dtype: &DataType,
    ) -> Bound<'py, PyAny> {
        let value = py.eval(value, None, None).unwrap();
        let fill_value =
            PyFillValue::py_new(PyFillValueInput(value), dtype.clone().into()).unwrap();
        Bound::new(py, fill_value).unwrap().into_any()
    }

    #[test]
    fn passes_through_a_fill_value_of_the_same_data_type() {
        Python::attach(|py| {
            let dtype = data_type::int32();
            let object = fill_value_object(py, c"-9999", &dtype);
            let fill_value = PyFillValueInput(object).resolve(&dtype).unwrap();
            assert_eq!(fill_value.as_ne_bytes(), (-9999i32).to_ne_bytes());
        });
    }

    #[test]
    fn rejects_a_fill_value_of_another_data_type() {
        Python::attach(|py| {
            // int32 and float32 are both 4 bytes, so only the data type tells
            // these apart.
            let object = fill_value_object(py, c"-9999", &data_type::int32());
            let error = PyFillValueInput(object)
                .resolve(&data_type::float32())
                .unwrap_err();
            assert!(format!("{error}").contains(
                "fill value has data type 'int32', but the array has data type 'float32'"
            ));
        });
    }

    #[test]
    fn rejects_a_value_that_is_not_a_fill_value() {
        let error = resolve(c"object()", &data_type::complex64()).unwrap_err();
        // The message names the value and the data type, and the cause names
        // what was wrong with the value.
        let message = format!("{error}");
        assert!(message.contains("as a fill value of data type 'complex64'"));
        assert!(message.contains("not a number or a JSON value"));
    }
}
