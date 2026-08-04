//! Helpers for the pyo3 boundary.

/// Run `f` with the GIL released, and return its result.
///
/// Use this instead of [`pyo3::Python::detach`] for any closure that returns a
/// zarrs-derived type. `Python::detach` bounds both the closure and its return
/// type by `Ungil`, which implies `Send`. zarrs relaxes its `Send`/`Sync`
/// bounds on `wasm32`, so a return type as ordinary as
/// `ZarristaResult<ArrayData>` does not satisfy that bound there:
/// `ZarristaError` holds a `CodecError`, which holds a `DataType`, which is an
/// `Arc<dyn DataTypeTraits>`.
///
/// On `wasm32` this runs `f` directly. That target is single-threaded, so there
/// is no GIL contention to relieve and nothing is lost.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn detach<T, F>(py: pyo3::Python<'_>, f: F) -> T
where
    F: pyo3::marker::Ungil + FnOnce() -> T,
    T: pyo3::marker::Ungil,
{
    py.detach(f)
}

/// See the native implementation for why `wasm32` runs `f` directly.
#[cfg(target_arch = "wasm32")]
pub(crate) fn detach<T, F>(_py: pyo3::Python<'_>, f: F) -> T
where
    F: FnOnce() -> T,
{
    f()
}
