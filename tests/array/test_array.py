from typing import TYPE_CHECKING

from zarrista import Array
from zarrista.store import MemoryStore

if TYPE_CHECKING:
    from zarr_metadata import ArrayMetadataV3


def test_array_from_metadata():
    """
    Test that Array.from_metadata works
    """
    store = MemoryStore()

    meta: ArrayMetadataV3 = {
        "zarr_format": 3,
        "attributes": {},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
        "data_type": "int16",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "node_type": "array",
        "shape": (4, 4),
        "codecs": ({"name": "bytes"},),
    }

    Array.from_metadata(meta, store)
