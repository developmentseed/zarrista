"""Tests for the `zarrista.xarray` lazy BackendArray adapter.

Fixtures are written with zarr-python and read back through zarrista, then
wrapped for xarray. Reads are compared against the equivalent numpy slice.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray
from xarray.core import indexing

from zarrista import Array, FilesystemStore
from zarrista.xarray import ZarristaBackendArray, to_dataarray


@pytest.fixture
def int32_array(tmp_path: Path) -> tuple[Path, NDArray[np.int32]]:
    """A (9, 64, 100) int32 array with dim names and attrs; returns (path, data)."""
    path = tmp_path / "a.zarr"
    data = np.arange(9 * 64 * 100, dtype="int32").reshape(9, 64, 100)
    z = zarr.create_array(
        store=str(path),
        shape=data.shape,
        chunks=(3, 16, 50),
        dtype=data.dtype,
        dimension_names=("t", "y", "x"),
    )
    z[:] = data
    z.attrs["units"] = "m"
    return path, data


def test_backend_array_shape_and_dtype(int32_array):
    path, _data = int32_array
    arr = Array.open(FilesystemStore(path))
    backend = ZarristaBackendArray(arr)
    assert backend.shape == (9, 64, 100)
    assert backend.dtype == np.dtype("int32")


def test_raw_indexing_slice_matches_numpy(int32_array):
    path, data = int32_array
    backend = ZarristaBackendArray(Array.open(FilesystemStore(path)))
    result = backend._raw_indexing((slice(0, 2), slice(None), slice(5, 7)))
    np.testing.assert_array_equal(result, data[0:2, :, 5:7])


def test_raw_indexing_int_squeezes_axis(int32_array):
    path, data = int32_array
    backend = ZarristaBackendArray(Array.open(FilesystemStore(path)))
    result = backend._raw_indexing((5, slice(None), slice(None)))
    assert result.shape == (64, 100)
    np.testing.assert_array_equal(result, data[5])


def test_raw_indexing_negative_int(int32_array):
    path, data = int32_array
    backend = ZarristaBackendArray(Array.open(FilesystemStore(path)))
    result = backend._raw_indexing((-1, slice(None), slice(None)))
    assert result.shape == (64, 100)
    np.testing.assert_array_equal(result, data[-1])


def test_raw_indexing_step_not_one_raises(int32_array):
    path, _ = int32_array
    backend = ZarristaBackendArray(Array.open(FilesystemStore(path)))
    with pytest.raises(NotImplementedError):
        backend._raw_indexing((slice(0, 9, 2), slice(None), slice(None)))


def test_variable_length_dtype_raises(tmp_path: Path):
    path = tmp_path / "s.zarr"
    z = zarr.create_array(store=str(path), shape=(4,), chunks=(4,), dtype=str)
    z[:] = np.array(["a", "bb", "ccc", "dddd"], dtype=object)
    arr = Array.open(FilesystemStore(path))
    assert arr.dtype.size is None  # precondition: variable-length
    with pytest.raises(NotImplementedError):
        ZarristaBackendArray(arr)


def test_to_dataarray_dims_attrs_and_lazy(int32_array):
    path, _data = int32_array
    arr = Array.open(FilesystemStore(path))
    da = to_dataarray(arr, name="temp")

    assert da.name == "temp"
    assert da.dims == ("t", "y", "x")
    assert da.shape == (9, 64, 100)
    assert da.dtype == np.dtype("int32")
    assert da.attrs["units"] == "m"
    # The data is wrapped lazily and not yet loaded into memory.
    assert isinstance(da.variable._data, indexing.LazilyIndexedArray)


def test_to_dataarray_indexing_matches_numpy(int32_array):
    path, data = int32_array
    da = to_dataarray(Array.open(FilesystemStore(path)))
    np.testing.assert_array_equal(da[0:2, :, 5:7].to_numpy(), data[0:2, :, 5:7])
    np.testing.assert_array_equal(da[5].to_numpy(), data[5])
    np.testing.assert_array_equal(da.to_numpy(), data)


def test_to_dataarray_synthesizes_dim_names(tmp_path: Path):
    path = tmp_path / "nodims.zarr"
    data = np.arange(2 * 3, dtype="int16").reshape(2, 3)
    z = zarr.create_array(
        store=str(path),
        shape=data.shape,
        chunks=(2, 3),
        dtype=data.dtype,
    )
    z[:] = data
    da = to_dataarray(Array.open(FilesystemStore(path)))
    assert da.dims == ("dim_0", "dim_1")
