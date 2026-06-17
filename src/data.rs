//! A Python-exposed array type that implements the buffer protocol.
//!
//! This module provides `PyData`, an ND array type that can be used with numpy
//! via Python's buffer protocol. The buffer protocol allows Python objects to
//! expose raw memory buffers, enabling zero-copy interoperability with numpy.
//!
//! ## Buffer Protocol Overview
//!
//! The buffer protocol is defined in [PEP 3118] and allows objects to expose their
//! internal data as a contiguous or strided memory region. Key concepts:
//!
//! [PEP 3118](https://peps.python.org/pep-3118/)
//!
//! - **view**: A `Py_buffer` struct that describes how to interpret the memory
//! - **format**: A string describing the element type (e.g., "<H" for little-endian uint16)
//! - **shape**: Array dimensions (e.g., [height, width, bands] for a 3D array)
//! - **strides**: Byte offsets between consecutive elements in each dimension
//!
//! ## Reference Counting
//!
//! When `__getbuffer__` is called, we increment the reference count on the PyData
//! object (`Py_INCREF`) to ensure it stays alive while the buffer view exists.
//! Python's `PyBuffer_Release` automatically calls `Py_DECREF` after `__releasebuffer__`
//! returns, so we don't need to manually decrement the count.
//!
//! ## Memory Safety
//!
//! The shape and strides arrays are stored as `Box<[isize]>` on the PyData struct
//! itself. This means their lifetime is tied to the PyData object, which is kept
//! alive by the `Py_INCREF` call. This avoids the need to leak/free allocations
//! in `__getbuffer__`/`__releasebuffer__`.

use std::ffi::{c_char, c_void, CStr};
use std::os::raw::c_int;

use ndarray::ArrayD;
use numpy::PyArray;
use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;

/// The decoded data of a chunk, with variants for each supported dtype.
///
/// This is kept separate from `PyData` 1) because Python classes can't be defined as Rust enums and 2) so that we have a clean place to manage data types not supported by numpy.
pub enum DataInner {
    Bool(ArrayD<bool>),
    Float16(ArrayD<half::f16>),
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Int16(ArrayD<i16>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    Int8(ArrayD<i8>),
    Uint16(ArrayD<u16>),
    Uint32(ArrayD<u32>),
    Uint64(ArrayD<u64>),
    Uint8(ArrayD<u8>),
}

/// Run `$body` against the inner `ArrayD`, with `$a` bound to it regardless of
/// element type. Exhaustive, so a new `DataInner` variant is a compile error.
macro_rules! with_array {
    ($data:expr, $a:ident => $body:expr) => {{
        use DataInner::*;
        match $data {
            Bool($a) => $body,
            Float16($a) => $body,
            Float32($a) => $body,
            Float64($a) => $body,
            Int16($a) => $body,
            Int32($a) => $body,
            Int64($a) => $body,
            Int8($a) => $body,
            Uint16($a) => $body,
            Uint32($a) => $body,
            Uint64($a) => $body,
            Uint8($a) => $body,
        }
    }};
}

impl DataInner {
    /// The PEP 3118 buffer-protocol format string for this dtype, or `None` if
    /// the dtype has no buffer-protocol representation (e.g. dtypes with no
    /// matching native layout). `None` is the signal that consumers must copy.
    fn buffer_format(&self) -> Option<&'static CStr> {
        use DataInner::*;

        let format = match self {
            Bool(_) => c"?",
            Int8(_) => c"b",
            Int16(_) => c"h",
            Int32(_) => c"i",
            Int64(_) => c"q",
            Uint8(_) => c"B",
            Uint16(_) => c"H",
            Uint32(_) => c"I",
            Uint64(_) => c"Q",
            Float16(_) => c"e",
            Float32(_) => c"f",
            Float64(_) => c"d",
        };
        Some(format)
    }

    /// The size of a single element in bytes.
    fn itemsize(&self) -> usize {
        use DataInner::*;

        match self {
            Bool(_) | Int8(_) | Uint8(_) => 1,
            Float16(_) | Int16(_) | Uint16(_) => 2,
            Float32(_) | Int32(_) | Uint32(_) => 4,
            Float64(_) | Int64(_) | Uint64(_) => 8,
        }
    }

    /// A raw pointer to the first element of the backing buffer.
    fn data_ptr(&self) -> *mut c_void {
        with_array!(self, a => a.as_ptr() as *mut c_void)
    }

    /// Copy the decoded chunk into a fresh NumPy array. Used as the fallback for
    /// dtypes that cannot be exposed via the buffer protocol.
    fn to_numpy_with_copy<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        with_array!(&self, a => PyArray::from_array(py, a).into_any())
    }
}

#[pyclass(module = "zarrista", frozen, name = "Data")]
pub struct PyData {
    inner: DataInner,
    /// Shape in elements. Cached here so the buffer protocol can hand out a
    /// pointer with a lifetime tied to this (frozen) object.
    shape: Box<[isize]>,
    /// Strides in bytes, cached for the same reason.
    strides: Box<[isize]>,
}

impl From<DataInner> for PyData {
    fn from(inner: DataInner) -> Self {
        let itemsize = inner.itemsize() as isize;
        let shape: Box<[isize]> =
            with_array!(&inner, a => a.shape().iter().map(|&d| d as isize).collect());
        let strides: Box<[isize]> = with_array!(&inner,
            a => a.strides().iter().map(|&s| s * itemsize).collect());
        Self {
            inner,
            shape,
            strides,
        }
    }
}

#[pymethods]
impl PyData {
    /// Convert the decoded chunk into a NumPy array.
    ///
    /// When the dtype has a buffer-protocol representation this is a zero-copy
    /// view (numpy reads our buffer directly). Otherwise the data is copied and
    /// converted into a fresh array.
    fn to_numpy<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if slf.borrow().inner.buffer_format().is_some() {
            // Zero-copy: `np.asarray` views our buffer via `__getbuffer__`.
            py.import("numpy")?.call_method1("asarray", (&slf,))
        } else {
            // Fallback: copy/convert into a new numpy array.
            Ok(slf.borrow().inner.to_numpy_with_copy(py))
        }
    }

    /// Buffer-protocol export (PEP 3118): lets any consumer (`memoryview`,
    /// numpy, pyarrow, …) read the data without a copy.
    ///
    /// # Safety
    /// `view` is a valid `Py_buffer` provided by the interpreter. We pin `self`
    /// for the view's lifetime with `Py_INCREF`, and `shape`/`strides` are owned
    /// by `self`, so all pointers stay valid until `__releasebuffer__`.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("null buffer view"));
        }
        let Some(format) = slf.inner.buffer_format() else {
            return Err(PyBufferError::new_err(
                "this data type has no buffer-protocol representation",
            ));
        };
        // We only ever expose a read-only buffer.
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("buffer is read-only"));
        }

        let itemsize = slf.inner.itemsize() as isize;
        let len = slf.shape.iter().product::<isize>() * itemsize;

        (*view).buf = slf.inner.data_ptr();
        (*view).len = len;
        (*view).itemsize = itemsize;
        (*view).readonly = 1;
        (*view).ndim = slf.shape.len() as c_int;
        (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
            format.as_ptr() as *mut c_char
        } else {
            std::ptr::null_mut()
        };
        (*view).shape = slf.shape.as_ptr() as *mut isize;
        (*view).strides = slf.strides.as_ptr() as *mut isize;
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();

        // Keep `self` (and therefore the buffer) alive while the view exists;
        // Python calls `Py_DECREF` after `__releasebuffer__` returns.
        (*view).obj = slf.as_ptr();
        ffi::Py_INCREF((*view).obj);

        Ok(())
    }

    /// Required counterpart to `__getbuffer__`. Nothing to free: all memory is
    /// owned by `self`, and Python handles the `Py_DECREF` on `view.obj`.
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

impl PyData {}
