use pyo3::prelude::*;

/// Return a friendly greeting, proving the Rust <-> Python round-trip works.
#[pyfunction]
fn hello() -> &'static str {
    "Hello from zarrsita!"
}

/// The compiled core of zarrsita, imported as `zarrsita._zarrsita`.
#[pymodule]
fn _zarrsita(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
