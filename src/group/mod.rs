#[cfg(feature = "async")]
mod r#async;
mod shared;
mod sync;

#[cfg(feature = "async")]
pub use r#async::PyAsyncGroup;
pub use sync::PyGroup;

/// The final path segment of an absolute node path (`/a/b` -> `b`).
// TODO: switch to using richer Path type
pub(crate) fn last_segment(path: &str) -> String {
    path.trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("")
        .to_string()
}
