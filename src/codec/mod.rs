mod array_to_array;
mod array_to_bytes;
mod bytes_to_bytes;
mod options;

use pyo3::prelude::*;

pub use array_to_array::{bitround, transpose, PyArrayToArrayCodec};
pub use array_to_bytes::PyArrayToBytesCodec;
pub use bytes_to_bytes::blosc::blosc;
pub use bytes_to_bytes::crc32c::crc32c;
pub use bytes_to_bytes::gzip::gzip;
pub use bytes_to_bytes::zstd::zstd;
pub use bytes_to_bytes::PyBytesToBytesCodec;
pub use options::PyCodecOptions;

/// Build the `zarrista.codec` submodule and attach it to `parent`.
///
/// The module is also registered in `sys.modules` as
/// `zarrista._zarrista.codec` so that `from zarrista._zarrista.codec import ...`
/// (and therefore the `zarrista.codec` re-export shim) resolves at runtime.
pub fn register_codec_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let codec = PyModule::new(py, "codec")?;

    codec.add_class::<PyArrayToArrayCodec>()?;
    codec.add_class::<PyArrayToBytesCodec>()?;
    codec.add_class::<PyBytesToBytesCodec>()?;
    codec.add_function(wrap_pyfunction!(transpose, &codec)?)?;
    codec.add_function(wrap_pyfunction!(bitround, &codec)?)?;
    codec.add_function(wrap_pyfunction!(blosc, &codec)?)?;
    codec.add_function(wrap_pyfunction!(crc32c, &codec)?)?;
    codec.add_function(wrap_pyfunction!(gzip, &codec)?)?;
    codec.add_function(wrap_pyfunction!(zstd, &codec)?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("zarrista._zarrista.codec", &codec)?;

    parent.add_submodule(&codec)?;
    Ok(())
}
