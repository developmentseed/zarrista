"""Arrow PyCapsule export from `VariableArray`, verified with arro3.

A variable-length (string/bytes) array written with zarr-python is read back with
zarrista, exported through the Arrow C Data Interface, and the reconstructed Arrow
array is compared to the original values.
"""

from pathlib import Path

import arro3.core as arro3
import numpy as np
import zarr
from zarrista import Array, FilesystemStore, VariableArray


def test_variable_length_string_to_arrow(tmp_path: Path):
    path = tmp_path / "s.zarr"
    values = ["a", "bb", "ccc", "dddd"]
    z = zarr.create_array(store=str(path), shape=(4,), chunks=(2,), dtype="string")
    z[:] = np.array(values, dtype=object)

    arr = Array.open(FilesystemStore(path))

    # A string dtype decodes to a VariableArray (not a Tensor).
    full = arr[:]
    assert isinstance(full, VariableArray)
    assert full.shape == [4]

    # Consume the Arrow C interface and check the values round-trip.
    arrow_array = arro3.Array.from_arrow(full)
    assert arrow_array.to_pylist() == values
    assert "string" in str(arrow_array.type).lower()

    # Per-chunk reads expose the same interface.
    chunk0 = arr.retrieve_chunk([0])
    assert isinstance(chunk0, VariableArray)
    assert arro3.Array.from_arrow(chunk0).to_pylist() == values[0:2]
