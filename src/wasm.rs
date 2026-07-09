//! wasm32-specific glue for the Python bindings.

/// Assert `Send + Sync` for a `#[pyclass]` on `wasm32`.
///
/// zarrs relaxes its `Send`/`Sync` bounds on `wasm32` (via its
/// `MaybeSend`/`MaybeSync` shim), so any type wrapping an `Arc<dyn …Traits>`
/// is neither `Send` nor `Sync` there. pyo3 nonetheless requires every
/// `#[pyclass]` to be `Send + Sync`, and `Bound::get` on a `frozen` class
/// requires `Sync`. `wasm32` targets are single-threaded, so these bounds can
/// never be exercised, making the assertion sound. On native the bounds hold
/// for real and this macro expands to nothing.
///
/// The `target_arch = "wasm32"` gate intentionally mirrors zarrs' own gate so
/// the assertion is active on exactly the targets where the bounds are relaxed.
#[macro_export]
macro_rules! wasm_send_sync {
    ($ty:ty) => {
        // SAFETY: `wasm32` is single-threaded, so `Send`/`Sync` are never
        // exercised across threads.
        #[cfg(target_arch = "wasm32")]
        unsafe impl Send for $ty {}
        // SAFETY: see above.
        #[cfg(target_arch = "wasm32")]
        unsafe impl Sync for $ty {}
    };
}
