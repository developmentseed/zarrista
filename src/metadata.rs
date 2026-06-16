use pyo3::prelude::*;
use pythonize::{depythonize, pythonize, PythonizeError};
use zarrs::metadata::v2::{ArrayMetadataV2, GroupMetadataV2, MetadataV2};
use zarrs::metadata::v3::{ArrayMetadataV3, GroupMetadataV3, MetadataV3};
use zarrs::metadata::{ArrayMetadata, GroupMetadata};

/// Generate a pythonize-compatible newtype wrapper around a zarrs metadata type.
macro_rules! pythonized_metadata {
    ($name:ident, $inner:ty) => {
        #[doc = concat!("A Python-convertible wrapper around [`", stringify!($inner), "`].")]
        pub struct $name($inner);

        #[allow(dead_code)]
        impl $name {
            pub fn into_inner(self) -> $inner {
                self.0
            }
        }

        impl<'a, 'py> FromPyObject<'a, 'py> for $name {
            type Error = PythonizeError;

            fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
                Ok($name(depythonize(&obj)?))
            }
        }

        impl<'py> IntoPyObject<'py> for $name {
            type Target = PyAny;
            type Error = PythonizeError;
            type Output = Bound<'py, Self::Target>;

            fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                pythonize(py, &self.0)
            }
        }

        impl AsRef<$inner> for $name {
            fn as_ref(&self) -> &$inner {
                &self.0
            }
        }

        impl From<$inner> for $name {
            fn from(metadata: $inner) -> Self {
                $name(metadata)
            }
        }

        impl From<$name> for $inner {
            fn from(py_metadata: $name) -> Self {
                py_metadata.0
            }
        }
    };
}

pythonized_metadata!(PyMetadataV2, MetadataV2);
pythonized_metadata!(PyMetadataV3, MetadataV3);
pythonized_metadata!(PyArrayMetadata, ArrayMetadata);
pythonized_metadata!(PyArrayMetadataV2, ArrayMetadataV2);
pythonized_metadata!(PyArrayMetadataV3, ArrayMetadataV3);
pythonized_metadata!(PyGroupMetadata, GroupMetadata);
pythonized_metadata!(PyGroupMetadataV2, GroupMetadataV2);
pythonized_metadata!(PyGroupMetadataV3, GroupMetadataV3);
