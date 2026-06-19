"""Array indexing: read regions with zarrista and compare against zarr-python.

Fixtures are written with zarr-python, read back with zarrista, and the decoded
region is compared to the equivalent numpy slice. This doubles as a round-trip
compatibility check between the two implementations.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray
from obstore.store import LocalStore
from zarrista import Array, AsyncArray, FilesystemStore, Tensor


@pytest.fixture
def int32_array(tmp_path: Path) -> tuple[Path, NDArray[np.int32]]:
    """A (9, 64, 100) int32 array written with zarr-python; returns (path, data)."""
    path = tmp_path / "a.zarr"
    data = np.arange(9 * 64 * 100, dtype="int32").reshape(9, 64, 100)
    z = zarr.create_array(
        store=str(path), shape=data.shape, chunks=(3, 16, 50), dtype=data.dtype,
    )
    z[:] = data
    return path, data


def test_slice_read_matches_numpy(int32_array: tuple[Path, NDArray[np.int32]]):
    path, data = int32_array
    arr = Array.open(FilesystemStore(path))

    result = arr.retrieve_array_subset(
        (slice(0, 2), slice(None), slice(5, 7)),
    ).to_numpy()

    np.testing.assert_array_equal(result, data[0:2, :, 5:7])


def test_fixed_dtype_returns_tensor(int32_array: tuple[Path, NDArray[np.int32]]):
    """A fixed-width dtype decodes to a `Tensor` carrying `shape`/`dtype`; its raw
    `buffer()` reinterprets to the same array as `to_numpy()`."""
    path, data = int32_array
    arr = Array.open(FilesystemStore(path))

    tensor = arr.retrieve_array_subset((slice(0, 2), slice(None), slice(5, 7)))
    assert isinstance(tensor, Tensor)
    assert tensor.shape == [2, 64, 2]
    assert tensor.dtype == arr.dtype

    expected = data[0:2, :, 5:7]
    np.testing.assert_array_equal(tensor.to_numpy(), expected)

    # buffer() exposes the raw decoded bytes; reinterpreting matches to_numpy().
    from_buffer = np.frombuffer(tensor.buffer(), dtype="int32").reshape(tensor.shape)
    np.testing.assert_array_equal(from_buffer, expected)


def test_getitem_matches_retrieve_array_subset(
    int32_array: tuple[Path, NDArray[np.int32]],
):
    path, _data = int32_array
    arr = Array.open(FilesystemStore(path))

    key = (slice(0, 2), slice(None), slice(5, 7))
    np.testing.assert_array_equal(
        arr[key].to_numpy(), arr.retrieve_array_subset(key).to_numpy(),
    )


def test_int_index_retains_axis(int32_array: tuple[Path, NDArray[np.int32]]):
    """ndim-preserving (zarrs-consistent): an int keeps a length-1 axis."""
    path, data = int32_array
    arr = Array.open(FilesystemStore(path))

    result = arr[5].to_numpy()
    assert result.shape == (1, 64, 100)
    np.testing.assert_array_equal(result, data[5:6])


def test_negative_index(int32_array: tuple[Path, NDArray[np.int32]]):
    path, data = int32_array
    arr = Array.open(FilesystemStore(path))
    np.testing.assert_array_equal(arr[-1].to_numpy(), data[8:9])


def test_ellipsis(int32_array: tuple[Path, NDArray[np.int32]]):
    path, data = int32_array
    arr = Array.open(FilesystemStore(path))
    result = arr[5, ..., 3].to_numpy()
    np.testing.assert_array_equal(result, data[5:6, :, 3:4])


def test_full_array(int32_array: tuple[Path, NDArray[np.int32]]):
    path, data = int32_array
    arr = Array.open(FilesystemStore(path))
    np.testing.assert_array_equal(arr[...].to_numpy(), data)


def test_out_of_bounds_int_raises(int32_array: tuple[Path, NDArray[np.int32]]):
    path, _ = int32_array
    arr = Array.open(FilesystemStore(path))
    with pytest.raises(IndexError):
        arr[9]  # axis 0 has size 9 -> max valid index is 8


def test_step_not_one_raises(int32_array: tuple[Path, NDArray[np.int32]]):
    path, _ = int32_array
    arr = Array.open(FilesystemStore(path))
    with pytest.raises(NotImplementedError):
        arr[0:9:2]


def test_float64_dtype(tmp_path):
    """Exercise the dtype dispatch on a non-integer type."""
    path = tmp_path / "f.zarr"
    data = (np.arange(4 * 5, dtype="float64") * 0.5).reshape(4, 5)
    z = zarr.create_array(
        store=str(path), shape=data.shape, chunks=(2, 5), dtype=data.dtype,
    )
    z[:] = data

    arr = Array.open(FilesystemStore(path))
    np.testing.assert_array_equal(arr[1:3, 0:4].to_numpy(), data[1:3, 0:4])


async def test_async_getitem_matches_numpy(int32_array: tuple[Path, NDArray[np.int32]]):
    """The async path (obstore + `await arr[...]`) returns the same region."""
    path, data = int32_array

    arr = await AsyncArray.open_async(LocalStore(str(path)))
    result = (await arr[0:2, :, 5:7]).to_numpy()

    np.testing.assert_array_equal(result, data[0:2, :, 5:7])
