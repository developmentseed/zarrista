from typing import Literal

MetadataConvertVersion = Literal["default", "v3"]
"""The Zarr version to write when converting metadata.

- `"default"`: write the same version as the input metadata.
- `"v3"`: write Zarr V3 metadata (existing V2 metadata is not removed).
"""

MetadataEraseVersion = Literal["default", "all", "v3", "v2"]
"""The Zarr version of metadata to erase.

- `"default"`: erase the same version as the input metadata.
- `"all"`: erase all metadata.
- `"v3"`: erase Zarr V3 metadata.
- `"v2"`: erase Zarr V2 metadata.
"""

UseConsolidatedMetadata = Literal["auto", "must", "never"]
"""Whether to use a root group's consolidated metadata when opening a hierarchy.

- `"auto"`: use consolidated metadata if present, otherwise list storage.
- `"must"`: require consolidated metadata to be present, else fail.
- `"never"`: never use consolidated metadata, always re-discover from storage.
"""

class Config:
    """A proxy to the `zarrs` global configuration.

    This type is not constructable; use the module-level singleton
    [`zarrista.config`][zarrista.config] instead. Its getters and setters read
    from and write to the process-wide global configuration.
    """

    @property
    def validate_checksums(self) -> bool:
        """Whether checksum codecs (e.g. `crc32c`, `fletcher32`) validate that
        encoded data matches stored checksums. Default `True`."""

    @validate_checksums.setter
    def validate_checksums(self, value: bool) -> None: ...
    @property
    def store_empty_chunks(self) -> bool:
        """If `False`, chunks where every element equals the fill value are not
        stored. If `True`, the fill-value check is skipped and empty chunks are
        stored. Default `False`."""

    @store_empty_chunks.setter
    def store_empty_chunks(self, value: bool) -> None: ...
    @property
    def codec_concurrent_target(self) -> int:
        """The default number of concurrent operations to target for codec
        encoding and decoding. Zero means unconstrained. Defaults to the number
        of available threads."""

    @codec_concurrent_target.setter
    def codec_concurrent_target(self, value: int) -> None: ...
    @property
    def chunk_concurrent_minimum(self) -> int:
        """The preferred minimum chunk concurrency for array operations spanning
        multiple chunks. Default `4`."""

    @chunk_concurrent_minimum.setter
    def chunk_concurrent_minimum(self, value: int) -> None: ...
    @property
    def codec_store_metadata_if_encode_only(self) -> bool:
        """Whether codecs performing irreversible encode-only transformations
        (currently only `bitround`) write their metadata. Default `True`."""

    @codec_store_metadata_if_encode_only.setter
    def codec_store_metadata_if_encode_only(self, value: bool) -> None: ...
    @property
    def include_zarrs_metadata(self) -> bool:
        """Whether generated array metadata includes the `_zarrs` attribute
        recording the `zarrs` version and source repository. Default `True`."""

    @include_zarrs_metadata.setter
    def include_zarrs_metadata(self, value: bool) -> None: ...
    @property
    def experimental_partial_encoding(self) -> bool:
        """Whether `store_chunk_subset` / `store_array_subset` may use partial
        encoding (relevant to the sharding codec). Experimental. Default
        `False`."""

    @experimental_partial_encoding.setter
    def experimental_partial_encoding(self, value: bool) -> None: ...
    @property
    def convert_aliased_extension_names(self) -> bool:
        """Whether aliased extension names are replaced by their standard name
        when metadata is resaved. Default `False`."""

    @convert_aliased_extension_names.setter
    def convert_aliased_extension_names(self, value: bool) -> None: ...
    @property
    def metadata_convert_version(self) -> MetadataConvertVersion:
        """The Zarr version to write when converting metadata. Default
        `"default"`."""

    @metadata_convert_version.setter
    def metadata_convert_version(self, value: MetadataConvertVersion) -> None: ...
    @property
    def metadata_erase_version(self) -> MetadataEraseVersion:
        """The Zarr version of metadata to erase. Default `"default"`."""

    @metadata_erase_version.setter
    def metadata_erase_version(self, value: MetadataEraseVersion) -> None: ...
    @property
    def use_consolidated_metadata(self) -> UseConsolidatedMetadata:
        """Whether to use a root group's consolidated metadata when opening a
        hierarchy. Default `"auto"`."""

    @use_consolidated_metadata.setter
    def use_consolidated_metadata(self, value: UseConsolidatedMetadata) -> None: ...
