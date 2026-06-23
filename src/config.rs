use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use zarrs::config::{
    global_config, global_config_mut, MetadataConvertVersion, MetadataEraseVersion,
    UseConsolidatedMetadata,
};

/// A proxy to the `zarrs` global configuration.
///
/// This type is not constructable from Python; use the module-level singleton
/// [`zarrista.config`] instead. Its getters and setters read from and write to
/// the process-wide global configuration.
#[pyclass(module = "zarrista", name = "Config")]
pub struct PyConfig;

#[pymethods]
impl PyConfig {
    #[getter]
    fn validate_checksums(&self) -> bool {
        global_config().validate_checksums()
    }

    #[setter]
    fn set_validate_checksums(&mut self, value: bool) {
        global_config_mut().set_validate_checksums(value);
    }

    #[getter]
    fn store_empty_chunks(&self) -> bool {
        global_config().store_empty_chunks()
    }

    #[setter]
    fn set_store_empty_chunks(&mut self, value: bool) {
        global_config_mut().set_store_empty_chunks(value);
    }

    #[getter]
    fn codec_concurrent_target(&self) -> usize {
        global_config().codec_concurrent_target()
    }

    #[setter]
    fn set_codec_concurrent_target(&mut self, value: usize) {
        global_config_mut().set_codec_concurrent_target(value);
    }

    #[getter]
    fn chunk_concurrent_minimum(&self) -> usize {
        global_config().chunk_concurrent_minimum()
    }

    #[setter]
    fn set_chunk_concurrent_minimum(&mut self, value: usize) {
        global_config_mut().set_chunk_concurrent_minimum(value);
    }

    #[getter]
    fn codec_store_metadata_if_encode_only(&self) -> bool {
        global_config().codec_store_metadata_if_encode_only()
    }

    #[setter]
    fn set_codec_store_metadata_if_encode_only(&mut self, value: bool) {
        global_config_mut().set_codec_store_metadata_if_encode_only(value);
    }

    #[getter]
    fn include_zarrs_metadata(&self) -> bool {
        global_config().include_zarrs_metadata()
    }

    #[setter]
    fn set_include_zarrs_metadata(&mut self, value: bool) {
        global_config_mut().set_include_zarrs_metadata(value);
    }

    #[getter]
    fn experimental_partial_encoding(&self) -> bool {
        global_config().experimental_partial_encoding()
    }

    #[setter]
    fn set_experimental_partial_encoding(&mut self, value: bool) {
        global_config_mut().set_experimental_partial_encoding(value);
    }

    #[getter]
    fn convert_aliased_extension_names(&self) -> bool {
        global_config().convert_aliased_extension_names()
    }

    #[setter]
    fn set_convert_aliased_extension_names(&mut self, value: bool) {
        global_config_mut().set_convert_aliased_extension_names(value);
    }

    #[getter]
    fn metadata_convert_version(&self) -> &'static str {
        metadata_convert_version_to_str(global_config().metadata_convert_version())
    }

    #[setter]
    fn set_metadata_convert_version(&mut self, value: PyBackedStr) -> PyResult<()> {
        let version = parse_metadata_convert_version(&value)?;
        global_config_mut().set_metadata_convert_version(version);
        Ok(())
    }

    #[getter]
    fn metadata_erase_version(&self) -> &'static str {
        metadata_erase_version_to_str(global_config().metadata_erase_version())
    }

    #[setter]
    fn set_metadata_erase_version(&mut self, value: PyBackedStr) -> PyResult<()> {
        let version = parse_metadata_erase_version(&value)?;
        global_config_mut().set_metadata_erase_version(version);
        Ok(())
    }

    #[getter]
    fn use_consolidated_metadata(&self) -> &'static str {
        use_consolidated_metadata_to_str(global_config().use_consolidated_metadata())
    }

    #[setter]
    fn set_use_consolidated_metadata(&mut self, value: PyBackedStr) -> PyResult<()> {
        let mode = parse_use_consolidated_metadata(&value)?;
        global_config_mut().set_use_consolidated_metadata(mode);
        Ok(())
    }
}

fn metadata_convert_version_to_str(version: MetadataConvertVersion) -> &'static str {
    match version {
        MetadataConvertVersion::Default => "default",
        MetadataConvertVersion::V3 => "v3",
    }
}

fn parse_metadata_convert_version(value: &str) -> PyResult<MetadataConvertVersion> {
    match value.to_ascii_lowercase().as_str() {
        "default" => Ok(MetadataConvertVersion::Default),
        "v3" => Ok(MetadataConvertVersion::V3),
        other => Err(PyValueError::new_err(format!(
            "unknown metadata convert version {other:?}; expected one of 'default', 'v3'"
        ))),
    }
}

fn metadata_erase_version_to_str(version: MetadataEraseVersion) -> &'static str {
    match version {
        MetadataEraseVersion::Default => "default",
        MetadataEraseVersion::All => "all",
        MetadataEraseVersion::V3 => "v3",
        MetadataEraseVersion::V2 => "v2",
    }
}

fn parse_metadata_erase_version(value: &str) -> PyResult<MetadataEraseVersion> {
    match value.to_ascii_lowercase().as_str() {
        "default" => Ok(MetadataEraseVersion::Default),
        "all" => Ok(MetadataEraseVersion::All),
        "v3" => Ok(MetadataEraseVersion::V3),
        "v2" => Ok(MetadataEraseVersion::V2),
        other => Err(PyValueError::new_err(format!(
            "unknown metadata erase version {other:?}; expected one of \
             'default', 'all', 'v3', 'v2'"
        ))),
    }
}

fn use_consolidated_metadata_to_str(mode: UseConsolidatedMetadata) -> &'static str {
    match mode {
        UseConsolidatedMetadata::Auto => "auto",
        UseConsolidatedMetadata::Must => "must",
        UseConsolidatedMetadata::Never => "never",
    }
}

fn parse_use_consolidated_metadata(value: &str) -> PyResult<UseConsolidatedMetadata> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(UseConsolidatedMetadata::Auto),
        "must" => Ok(UseConsolidatedMetadata::Must),
        "never" => Ok(UseConsolidatedMetadata::Never),
        other => Err(PyValueError::new_err(format!(
            "unknown use consolidated metadata mode {other:?}; expected one of \
             'auto', 'must', 'never'"
        ))),
    }
}
