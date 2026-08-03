use std::sync::Arc;

use pyo3::exceptions::PyValueError;
pub use pyo3::prelude::*;
use zarrs::array::codec::ShardingCodecConfiguration;
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodec;
use zarrs::array::{ArrayToBytesCodecTraits, CodecChain};
use zarrs::metadata::ConfigurationSerialize;

use crate::error::ZarristaResult;

/// Subchunk behaviour for [`CodecChain`].
pub(crate) trait CodecChainSubchunkExt {
    /// Derive the codec chain that decodes one subchunk of this chain.
    ///
    /// Returns `None` if the serializer is not a `sharding_indexed` codec. Such
    /// a chain has no subchunk level.
    ///
    /// The bytes of a subchunk are the output of the sharding codec's inner
    /// chain. To decode subchunk bytes back to array values:
    ///
    /// - reverse that inner subchunk chain
    /// - then reverse the outer array-to-array codecs, which ran before
    ///   sharding.
    ///
    /// The outer bytes-to-bytes codecs are absent by design. They encode the
    /// whole shard, not one subchunk.
    ///
    /// The rule is self-similar. Applied to its own output, it gives the next
    /// level down for a nested shard.
    fn subchunk_chain(&self) -> ZarristaResult<Option<Arc<CodecChain>>>;
}

impl CodecChainSubchunkExt for CodecChain {
    fn subchunk_chain(&self) -> ZarristaResult<Option<Arc<CodecChain>>> {
        let Some(inner) = sharding_inner_codecs(&**self.array_to_bytes_codec())? else {
            return Ok(None);
        };

        let filters = self
            .array_to_array_codecs()
            .iter()
            .chain(inner.array_to_array_codecs())
            .cloned()
            .collect();

        Ok(Some(Arc::new(CodecChain::new(
            filters,
            inner.array_to_bytes_codec().clone(),
            inner.bytes_to_bytes_codecs().to_vec(),
        ))))
    }
}

/// Access the inner codec chain of a `sharding_indexed` codec.
///
/// Returns `None` if `serializer` is not a `sharding_indexed` codec.
///
/// Ideally upstream will expose ShardingCodec::inner_codecs directly:
/// <https://github.com/zarrs/zarrs/issues/438>
pub(crate) fn sharding_inner_codecs(
    serializer: &dyn ArrayToBytesCodecTraits,
) -> ZarristaResult<Option<Arc<CodecChain>>> {
    if !serializer.as_any().is::<ShardingCodec>() {
        return Ok(None);
    }

    let configuration = serializer
        .configuration_v3(&Default::default())
        .expect("a sharding codec always has a configuration");
    let ShardingCodecConfiguration::V1(sharding) =
        ShardingCodecConfiguration::try_from_configuration(configuration).map_err(|err| {
            PyValueError::new_err(format!("could not read the sharding configuration: {err}"))
        })?
    else {
        // Only V1 exists today, and the array would not have opened if the
        // codec had failed to build. Degrade to "not sharded".
        return Ok(None);
    };

    Ok(Some(Arc::new(CodecChain::from_metadata(&sharding.codecs)?)))
}
