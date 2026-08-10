"""Buffer-protocol export on `FixedLengthTensor`.

A fixed-width `FixedLengthTensor` is itself a PEP 3118 N-dimensional, typed, read-only
buffer. Fixtures are written with zarr-python and read back with zarrista.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from zarrista import Array, FixedLengthTensor
from zarrista.store import FilesystemStore


def _tensor(path: Path, data: NDArray) -> FixedLengthTensor:
    z = zarr.create_array(
        store=str(path),
        shape=data.shape,
        chunks=data.shape,
        dtype=data.dtype,
    )
    z[:] = data
    arr = Array.open(FilesystemStore(path))
    ndim = data.ndim
    tensor = arr.retrieve_array_subset((slice(None),) * ndim)
    assert isinstance(tensor, FixedLengthTensor)
    return tensor


def test_memoryview_reports_shape_format_itemsize(tmp_path: Path):
    data = np.arange(2 * 3 * 4, dtype="int32").reshape(2, 3, 4)
    mv = memoryview(_tensor(tmp_path / "a.zarr", data))
    assert mv.shape == (2, 3, 4)
    assert mv.ndim == 3
    assert mv.itemsize == 4
    assert mv.format == "i"
    assert mv.readonly is True
    assert mv.strides == (48, 16, 4)
    np.testing.assert_array_equal(np.asarray(mv), data)


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ],
)
def test_np_asarray_roundtrips_dtypes(tmp_path: Path, dtype: str):
    if dtype == "bool":
        data = (np.arange(12) % 2 == 0).reshape(3, 4)
    else:
        data = np.arange(12, dtype=dtype).reshape(3, 4)
    result = np.asarray(_tensor(tmp_path / "d.zarr", data))
    assert result.dtype == data.dtype
    np.testing.assert_array_equal(result, data)


def test_asarray_is_readonly(tmp_path: Path):
    data = np.arange(6, dtype="float64").reshape(2, 3)
    result = np.asarray(_tensor(tmp_path / "r.zarr", data))
    assert result.flags.writeable is False


def test_complex_raises_buffererror_but_to_numpy_works(tmp_path: Path):
    data = (
        (np.arange(12, dtype="float32") + 1j * np.arange(12, dtype="float32"))
        .astype("complex64")
        .reshape(3, 4)
    )
    tensor = _tensor(tmp_path / "c.zarr", data)

    with pytest.raises(BufferError):
        memoryview(tensor)

    # to_numpy() is a separate path (numpy's dtype set is wider than the buffer
    # protocol's) and still works for complex.
    np.testing.assert_array_equal(tensor.to_numpy(), data)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="__buffer__(flags) / inspect.BufferFlags requires Python 3.12+",
)
def test_writable_request_raises(tmp_path: Path):
    import inspect

    data = np.arange(6, dtype="int16").reshape(2, 3)
    tensor = _tensor(tmp_path / "w.zarr", data)
    with pytest.raises(BufferError):
        tensor.__buffer__(inspect.BufferFlags.WRITABLE)


def test_raw_buffer_still_flat_u8(tmp_path: Path):
    data = np.arange(6, dtype="int16").reshape(2, 3)
    tensor = _tensor(tmp_path / "b.zarr", data)
    raw = memoryview(tensor.buffer())
    assert raw.format == "B"
    assert raw.nbytes == data.nbytes


def test_array_protocol_roundtrips(tmp_path: Path):
    data = np.arange(2 * 3 * 4, dtype="int32").reshape(2, 3, 4)
    result = np.array(_tensor(tmp_path / "a.zarr", data))
    assert result.dtype == data.dtype
    np.testing.assert_array_equal(result, data)


def test_asarray_complex_is_real_array_not_object(tmp_path: Path):
    """Regression for #66: complex has no buffer format code, so without the
    `__array__` protocol `np.asarray` silently yields a 0-d object array."""
    data = (
        (np.arange(12, dtype="float32") + 1j * np.arange(12, dtype="float32"))
        .astype("complex64")
        .reshape(3, 4)
    )
    result = np.asarray(_tensor(tmp_path / "c.zarr", data))
    assert result.dtype == np.dtype("complex64")
    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result, data)


def test_array_honors_dtype_cast(tmp_path: Path):
    data = np.arange(6, dtype="int32").reshape(2, 3)
    result = np.asarray(_tensor(tmp_path / "d.zarr", data), dtype="float64")
    assert result.dtype == np.dtype("float64")
    np.testing.assert_array_equal(result, data.astype("float64"))


def test_array_copy_true_returns_independent_copy(tmp_path: Path):
    data = np.arange(6, dtype="int32").reshape(2, 3)
    result = np.array(_tensor(tmp_path / "e.zarr", data), copy=True)
    assert result.flags.writeable is True
    np.testing.assert_array_equal(result, data)


def test_array_copy_false_differing_dtype_raises(tmp_path: Path):
    data = np.arange(6, dtype="int32").reshape(2, 3)
    tensor = _tensor(tmp_path / "f.zarr", data)
    with pytest.raises(ValueError, match="zero-copy"):
        tensor.__array__(np.dtype("float64"), copy=False)


def _hand_written_tensor(
    path: Path,
    data_type: str,
    data: NDArray,
) -> FixedLengthTensor:
    """Write a fixture directly. zarr-python cannot write these data types."""
    path.mkdir(parents=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [data.size],
                "data_type": data_type,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [data.size]},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": [0, 0],
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            },
        ),
    )
    (path / "c").mkdir()
    (path / "c" / "0").write_bytes(data.tobytes())
    tensor = Array.open(FilesystemStore(path))[:]
    assert isinstance(tensor, FixedLengthTensor)
    return tensor


@pytest.mark.parametrize(
    ("data_type", "numpy_name"),
    [("complex_float32", "complex64"), ("complex_float64", "complex128")],
)
def test_to_numpy_renames_complex_aliases(tmp_path: Path, data_type, numpy_name):
    # The number doubles: Zarr's `complex_float32` names the component type,
    # while NumPy's `complex64` counts the bits of the whole value. Both are a
    # pair of 32-bit floats.
    data = np.array([1 + 2j, -3 - 4j], dtype=numpy_name)
    tensor = _hand_written_tensor(tmp_path / f"{data_type}.zarr", data_type, data)

    result = tensor.to_numpy()

    assert result.dtype == np.dtype(numpy_name)
    np.testing.assert_array_equal(result, data)
