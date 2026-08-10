use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use bytes::Bytes;
pub use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::{
    ArrayError, ArrayToBytesCodecTraits, CodecChain, CodecOptions, DataType, FillValue,
    FromArrayBytes,
};

use crate::array::PyFillValue;
use crate::codec::{PyCodecChain, PyCodecOptions};
use crate::data::Tensor;
use crate::dtype::PyDataType;
use crate::error::ZarristaResult;
use crate::repr::shape_and_dtype;
#[cfg(feature = "async")]
use crate::thread_pool::PyThreadPool;

#[pyclass(
    module = "zarrista",
    frozen,
    name = "EncodedChunk",
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyEncodedChunk {
    bytes: Bytes,
    codecs: Arc<CodecChain>,
    data_type: DataType,
    fill_value: FillValue,
    shape: Vec<NonZeroU64>,
}

crate::wasm_send_sync!(PyEncodedChunk);

impl PyEncodedChunk {
    pub fn new(
        bytes: Bytes,
        codecs: Arc<CodecChain>,
        data_type: DataType,
        fill_value: FillValue,
        shape: Vec<NonZeroU64>,
    ) -> Self {
        Self {
            bytes,
            codecs,
            data_type,
            fill_value,
            shape,
        }
    }

    fn _decode(&self, codec_options: &CodecOptions) -> Result<Tensor, ArrayError> {
        let bytes = self.codecs.decode(
            Cow::Borrowed(&self.bytes),
            &self.shape,
            &self.data_type,
            &self.fill_value,
            codec_options,
        )?;
        let shape = self.shape.iter().map(|v| v.get()).collect::<Vec<_>>();
        Tensor::from_array_bytes(bytes.into_owned(), &shape, &self.data_type)
    }
}

#[pymethods]
impl PyEncodedChunk {
    /// The raw, still-encoded chunk bytes.
    #[getter]
    fn buffer(&self) -> PyBytes {
        PyBytes::new(self.bytes.clone())
    }

    /// The codec chain that decodes the bytes.
    #[getter]
    fn codecs(&self) -> PyCodecChain {
        self.codecs.clone().into()
    }

    /// The Zarr data type of the decoded chunk.
    #[getter]
    fn data_type(&self) -> PyDataType {
        self.data_type.clone().into()
    }

    /// Decode the chunk bytes on the calling thread.
    #[pyo3(signature = (**codec_options))]
    fn decode(&self, py: Python, codec_options: Option<PyCodecOptions>) -> ZarristaResult<Tensor> {
        crate::py::detach(py, || {
            let codec_options = codec_options
                .map(|opts| opts.into_inner())
                .unwrap_or_default();
            Ok(self._decode(&codec_options)?)
        })
    }

    /// Decode the chunk bytes on a Rust thread pool.
    #[cfg(feature = "async")]
    #[pyo3(signature = (*, pool=None, **codec_options))]
    fn decode_async<'py>(
        &self,
        py: Python<'py>,
        pool: Option<&PyThreadPool>,
        codec_options: Option<PyCodecOptions>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use pyo3_async_runtimes::tokio::future_into_py;
        use tokio_rayon::AsyncThreadPool;

        use crate::error::ZarristaError;

        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();
        let pool = pool.map(|p| p.inner().clone());

        // Everything is under an Arc except for FillValue
        let encoded_chunk = self.clone();

        future_into_py(py, async move {
            // Use the free function spawn_fifo unless a pool was provided
            let decoded = if let Some(pool) = pool {
                pool.spawn_fifo_async(move || encoded_chunk._decode(&codec_options))
            } else {
                tokio_rayon::spawn_fifo(move || encoded_chunk._decode(&codec_options))
            };
            Ok(decoded.await.map_err(ZarristaError::from)?)
        })
    }

    /// The fill value of the decoded chunk.
    #[getter]
    fn fill_value(&self) -> PyFillValue {
        self.fill_value.clone().into()
    }

    /// The shape of the decoded chunk, in elements along each dimension.
    #[getter]
    fn shape(&self) -> Vec<NonZeroU64> {
        self.shape.clone()
    }

    /// Shows the encoded size, because that is what distinguishes this from the
    /// decoded chunk that has the same shape and data type.
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let shape: Vec<u64> = self.shape.iter().map(|edge| edge.get()).collect();
        let shape_and_dtype = shape_and_dtype(py, &shape, &self.data_type())?;
        Ok(format!(
            "EncodedChunk({shape_and_dtype}, nbytes={})",
            self.bytes.len()
        ))
    }
}
