//! Helpers for building `__repr__` strings.
//!
//! A `__repr__` must never format a zarrs type with `Debug`. That prints the
//! Rust type name and its struct syntax, such as
//! ```text
//! ZstdCodec { compression: 3, checksum: false }
//! ```
//! which are implementation details of this crate.
//!
//! Build the string from the Zarr v3 name and configuration instead, and render the configuration
//! through Python so that it uses Python syntax rather than JSON.

use std::borrow::Cow;

use pyo3::prelude::*;

use crate::metadata::PyConfiguration;

/// Build the repr of a type that Zarr v3 describes with a name and a configuration.
///
/// The result reads `Class("name", config={...})`. The name is absent for a
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
        parts.push(format!("{name:?}"));
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
