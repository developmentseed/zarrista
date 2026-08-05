"""NumPy export from `VariableArray`.

A variable-length array written with zarr-python is read back with zarrista and
converted to NumPy. Only the `string` data type has a NumPy form.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr

from zarrista import Array, VariableArray
from zarrista.store import FilesystemStore


def _variable_array(path: Path, values: np.ndarray, dtype: str) -> VariableArray:
    z = zarr.create_array(
        store=str(path),
        shape=values.shape,
        chunks=values.shape,
        dtype=dtype,
    )
    z[:] = values
    array = Array.open(FilesystemStore(path))[:]
    assert isinstance(array, VariableArray)
    return array


def test_to_numpy_gives_string_dtype(tmp_path: Path):
    values = ["a", "bb", "ccc", "dddd"]
    array = _variable_array(
        tmp_path / "s.zarr",
        np.array(values, dtype=object),
        "string",
    )

    result = array.to_numpy()

    assert result.dtype == np.dtypes.StringDType()
    assert result.tolist() == values


def test_to_numpy_keeps_shape(tmp_path: Path):
    values = np.array([["a", "bb"], ["ccc", "dddd"]], dtype=object)
    array = _variable_array(tmp_path / "s.zarr", values, "string")

    assert array.to_numpy().tolist() == values.tolist()


def test_array_protocol_casts_to_requested_dtype(tmp_path: Path):
    values = np.array(["a", "bb"], dtype=object)
    array = _variable_array(tmp_path / "s.zarr", values, "string")

    assert np.asarray(array, dtype=object).dtype == np.dtype(object)


def test_array_protocol_rejects_zero_copy(tmp_path: Path):
    # Building a StringDType array always allocates its own arena, so `copy=False`
    # can never be honoured.
    array = _variable_array(
        tmp_path / "s.zarr",
        np.array(["a"], dtype=object),
        "string",
    )

    with pytest.raises(ValueError, match="zero-copy"):
        np.array(array, copy=False)


def test_to_numpy_rejects_bytes_dtype(tmp_path: Path):
    # NumPy has no variable-width binary data type.
    values = np.array([b"a", b"bb"], dtype=object)
    array = _variable_array(tmp_path / "b.zarr", values, "bytes")

    with pytest.raises(NotImplementedError, match="not supported"):
        array.to_numpy()


def test_to_numpy_rejects_invalid_utf8(tmp_path: Path):
    # Zarr defines `string` as UTF-8, but nothing on the read path enforces it,
    # so a corrupt store must fail loudly rather than produce broken text.
    path = tmp_path / "s.zarr"
    z = zarr.create_array(
        store=str(path),
        shape=(1,),
        chunks=(1,),
        dtype="string",
        compressors=None,
    )
    z[:] = np.array(["ab"], dtype=object)

    chunk = next(p for p in path.rglob("*") if p.is_file() and p.name != "zarr.json")
    chunk.write_bytes(chunk.read_bytes().replace(b"ab", b"\xffb"))

    with pytest.raises(UnicodeDecodeError):
        Array.open(FilesystemStore(path))[:].to_numpy()
