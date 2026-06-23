//! Python exception types for zarrista.
//!
//! [`ZarristaError`] is the base class; every other exception subclasses it.
//! Each subclass corresponds to one underlying `zarrs` error category (plus a
//! couple of zarrista-specific cases). The `From<ZarristaError> for PyErr` impl
//! in [`crate::error`] maps Rust errors onto these classes.
//!
//! Note: the `ZarristaError` exception type here is distinct from the
//! `ZarristaError` *enum* in [`crate::error`]; they live in separate modules.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    zarrista.exceptions,
    ZarristaError,
    PyException,
    "Base class for all zarrista exceptions."
);
create_exception!(
    zarrista.exceptions,
    ArrayCreateError,
    ZarristaError,
    "Raised when an array cannot be opened or created."
);
create_exception!(
    zarrista.exceptions,
    ArrayError,
    ZarristaError,
    "Raised on an error reading from or operating on an array."
);
create_exception!(
    zarrista.exceptions,
    GroupCreateError,
    ZarristaError,
    "Raised when a group cannot be opened or created."
);
create_exception!(
    zarrista.exceptions,
    NodeCreateError,
    ZarristaError,
    "Raised when a child node cannot be enumerated or created."
);
create_exception!(
    zarrista.exceptions,
    NodePathError,
    ZarristaError,
    "Raised when a node path is invalid."
);
create_exception!(
    zarrista.exceptions,
    StorageError,
    ZarristaError,
    "Raised on an error from the underlying storage backend."
);
create_exception!(
    zarrista.exceptions,
    CodecError,
    ZarristaError,
    "Raised on a codec encode/decode error."
);
create_exception!(
    zarrista.exceptions,
    TransposeOrderError,
    ZarristaError,
    "Raised when a transpose codec order is invalid."
);
create_exception!(
    zarrista.exceptions,
    PluginCreateError,
    ZarristaError,
    "Raised when a codec or other plugin cannot be created from its configuration."
);
create_exception!(
    zarrista.exceptions,
    SerializationError,
    ZarristaError,
    "Raised when (de)serializing JSON or converting to/from Python objects fails."
);
create_exception!(
    zarrista.exceptions,
    ChunkGridCreateError,
    ZarristaError,
    "Raised when a chunk grid cannot be created from the given shapes."
);
create_exception!(
    zarrista.exceptions,
    IncompatibleDimensionalityError,
    ZarristaError,
    "Raised when a shape's dimensionality is incompatible with another."
);

/// Build the `zarrista.exceptions` submodule and attach it to `parent`.
///
/// The module is also registered in `sys.modules` as
/// `zarrista._zarrista.exceptions` so that
/// `from zarrista._zarrista.exceptions import ...` (and the `zarrista.exceptions`
/// re-export shim) resolves at runtime.
pub fn register_exceptions_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let exceptions = PyModule::new(py, "exceptions")?;

    exceptions.add("ZarristaError", py.get_type::<ZarristaError>())?;
    exceptions.add("ArrayCreateError", py.get_type::<ArrayCreateError>())?;
    exceptions.add("ArrayError", py.get_type::<ArrayError>())?;
    exceptions.add("GroupCreateError", py.get_type::<GroupCreateError>())?;
    exceptions.add("NodeCreateError", py.get_type::<NodeCreateError>())?;
    exceptions.add("NodePathError", py.get_type::<NodePathError>())?;
    exceptions.add("StorageError", py.get_type::<StorageError>())?;
    exceptions.add("CodecError", py.get_type::<CodecError>())?;
    exceptions.add("TransposeOrderError", py.get_type::<TransposeOrderError>())?;
    exceptions.add("PluginCreateError", py.get_type::<PluginCreateError>())?;
    exceptions.add("SerializationError", py.get_type::<SerializationError>())?;
    exceptions.add(
        "ChunkGridCreateError",
        py.get_type::<ChunkGridCreateError>(),
    )?;
    exceptions.add(
        "IncompatibleDimensionalityError",
        py.get_type::<IncompatibleDimensionalityError>(),
    )?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("zarrista._zarrista.exceptions", &exceptions)?;

    parent.add_submodule(&exceptions)?;
    Ok(())
}
