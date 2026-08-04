"""DLPack exchange in both directions.

zarrista exports decoded data as a DLPack tensor (`Tensor.__dlpack__`), and
imports array-like data through DLPack when writing. The import path is what
lets `arr[...] = ndarray` check the data type and shape, because DLPack carries
both; raw bytes carry neither.
"""

import gc

import numpy as np
import pytest
from numpy.typing import NDArray
from obstore.store import LocalStore

from zarrista import (
    Array,
    ArrayBuilder,
    ChunkGrid,
    DataType,
    FillValue,
    Tensor,
)
from zarrista.store import MemoryStore

# Every data type the DLPack mapping covers and numpy can express. `bfloat16`
# is in the mapping but numpy has no such type, so it is not listed here.
DTYPES = [
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
]


def _fill_value(dtype: str) -> FillValue:
    return FillValue(np.zeros((), dtype=dtype).tobytes())


def _array(dtype: str = "int32", *, shape: list[int] | None = None) -> Array:
    shape = shape or [4, 4]
    return ArrayBuilder(
        ChunkGrid.regular(shape, [2, 2]),
        DataType.from_string(dtype),
        _fill_value(dtype),
    ).create(MemoryStore(), "/a")


def _data(dtype: str = "int32") -> NDArray:
    values = np.arange(16).reshape(4, 4)
    # `arange` refuses to build a bool array of this length.
    return (values % 3 == 0) if dtype == "bool" else values.astype(dtype)


def _all() -> tuple[slice, slice]:
    return (slice(0, 4), slice(0, 4))


# --- import: writing an array-like -----------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_write_ndarray_round_trips(dtype: str):
    """Every data type in the DLPack mapping survives a write and a read."""
    arr = _array(dtype)
    data = _data(dtype)

    arr.store_array_subset(_all(), data)

    np.testing.assert_array_equal(np.asarray(arr[:, :]), data)


def test_write_needs_no_tobytes():
    """The point of the import path: a typed array goes in directly."""
    arr = _array()
    data = _data()

    arr.store_array_subset(_all(), data)

    np.testing.assert_array_equal(np.asarray(arr[:, :]), data)


def test_write_a_sub_region():
    arr = _array()
    data = _data()
    arr.store_array_subset(_all(), data)

    replacement = np.full((2, 2), 99, dtype="int32")
    arr.store_array_subset((slice(0, 2), slice(0, 2)), replacement)

    expected = data.copy()
    expected[:2, :2] = 99
    np.testing.assert_array_equal(np.asarray(arr[:, :]), expected)


def test_raw_bytes_skip_the_checks():
    """Bytes carry no type information, so the caller opts out of both checks."""
    arr = _array()
    data = _data()

    arr.store_array_subset(_all(), data.tobytes())

    np.testing.assert_array_equal(np.asarray(arr[:, :]), data)


# --- import: what gets rejected ---------------------------------------------


def test_data_type_mismatch_raises():
    """A cast can lose data, so the caller has to ask for it."""
    arr = _array("int32")

    with pytest.raises(TypeError, match=r"float64.*int32"):
        arr.store_array_subset(_all(), _data("float64"))


def test_shape_mismatch_raises():
    """The element count matches here, so only the shape check catches it."""
    arr = _array()

    with pytest.raises(ValueError, match=r"shape \[4, 4\].*shape \[2, 2\]"):
        arr.store_array_subset((slice(0, 2), slice(0, 2)), _data())


@pytest.mark.parametrize(
    ("label", "make"),
    [
        ("strided view", lambda d: d[:, ::2]),
        ("fortran order", np.asfortranarray),
        ("transposed", lambda d: d.T),
    ],
)
def test_non_contiguous_raises(label: str, make):
    """A non-compact tensor's byte length does not describe its elements."""
    arr = _array()
    data = make(_data())
    selection = (slice(0, 4), slice(0, data.shape[1]))

    with pytest.raises(ValueError, match="C-contiguous"):
        arr.store_array_subset(selection, data)


def test_ascontiguousarray_is_the_documented_fix():
    arr = _array()
    strided = _data()[:, ::2]

    arr.store_array_subset(
        (slice(0, 4), slice(0, 2)),
        np.ascontiguousarray(strided),
    )

    np.testing.assert_array_equal(np.asarray(arr[0:4, 0:2]), strided)


# --- import: the DLPack negotiation -----------------------------------------


class _OffDevice:
    """Claims to be on CUDA, and honours a host request like a real producer.

    zarrs encodes on the host, so zarrista asks the producer to move the data
    with `dl_device` rather than moving it itself.
    """

    def __init__(self, data: NDArray) -> None:
        self.data = data
        self.requested: dict | None = None

    def __dlpack_device__(self) -> tuple[int, int]:
        return (2, 0)  # kDLCUDA

    def __dlpack__(self, **kwargs: object) -> object:
        self.requested = kwargs
        return self.data.__dlpack__(max_version=kwargs["max_version"])


def test_off_device_data_is_requested_on_the_host():
    arr = _array()
    data = _data()
    producer = _OffDevice(data)

    arr.store_array_subset(_all(), producer)

    assert producer.requested is not None
    assert producer.requested["dl_device"] == (1, 0)  # kDLCPU
    np.testing.assert_array_equal(np.asarray(arr[:, :]), data)


class _OldProducer:
    """Predates `dl_device`, as a producer from before the 2023.12 array API."""

    def __dlpack_device__(self) -> tuple[int, int]:
        return (2, 0)

    def __dlpack__(self) -> object:
        raise TypeError("__dlpack__() got an unexpected keyword argument 'dl_device'")


def test_producer_without_dl_device_reports_the_device():
    arr = _array()

    with pytest.raises(ValueError, match="could not move it to the host"):
        arr.store_array_subset(_all(), _OldProducer())


# --- import: ownership ------------------------------------------------------


def test_the_source_array_can_be_dropped_after_the_call():
    """The tensor keeps the producer's buffer alive, so the write owns nothing
    borrowed from Python's stack."""
    arr = _array()
    expected = _data()

    data = _data()
    arr.store_array_subset(_all(), data)
    del data
    gc.collect()

    np.testing.assert_array_equal(np.asarray(arr[:, :]), expected)


async def test_async_write_outlives_the_source_array(tmp_path):
    """The async write moves the tensor into a `'static` future, and its
    deleter runs on whichever thread drops that future."""
    arr = await ArrayBuilder(
        ChunkGrid.regular([4, 4], [2, 2]),
        DataType.from_string("int32"),
        _fill_value("int32"),
    ).create_async(LocalStore(str(tmp_path)), "/a")
    expected = _data()

    data = _data()
    pending = arr.store_array_subset(_all(), data)
    del data
    gc.collect()
    await pending

    np.testing.assert_array_equal(np.asarray(await arr[:, :]), expected)


async def test_many_concurrent_async_writes(tmp_path):
    """Exercises dropping several tensors on the runtime's worker threads."""
    import asyncio

    arr = await ArrayBuilder(
        ChunkGrid.regular([4, 4], [2, 2]),
        DataType.from_string("int32"),
        _fill_value("int32"),
    ).create_async(LocalStore(str(tmp_path)), "/a")

    await asyncio.gather(
        *(
            arr.store_array_subset(
                (slice(r, r + 2), slice(c, c + 2)),
                np.full((2, 2), r * 2 + c, dtype="int32"),
            )
            for r in (0, 2)
            for c in (0, 2)
        ),
    )

    result = np.asarray(await arr[:, :])
    assert result[0, 0] == 0
    assert result[2, 2] == 6


# --- export -----------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_export_round_trips_through_numpy(dtype: str):
    arr = _array(dtype)
    data = _data(dtype)
    arr.store_array_subset(_all(), data)

    tensor = arr[:, :]
    assert isinstance(tensor, Tensor)

    np.testing.assert_array_equal(np.from_dlpack(tensor), data)


def test_export_reports_a_cpu_device():
    arr = _array()
    arr.store_array_subset(_all(), _data())

    assert arr[:, :].__dlpack_device__() == (1, 0)  # kDLCPU


@pytest.mark.xfail(
    reason="`Tensor.__dlpack__` ignores `max_version` and always exports a "
    "legacy capsule, which the versioned importer cannot read. See "
    "developmentseed/zarrista#108 and the DLPack ownership design.",
    raises=AttributeError,
    strict=True,
)
def test_export_then_import_round_trips():
    """A `Tensor` read from one array can be written straight into another."""
    source = _array()
    data = _data()
    source.store_array_subset(_all(), data)

    destination = _array()
    destination.store_array_subset(_all(), source[:, :])

    np.testing.assert_array_equal(np.asarray(destination[:, :]), data)
