"""NumPy export of the `numpy.datetime64` and `numpy.timedelta64` data types.

Zarr keeps the temporal unit and the scale factor in the data type
configuration, while NumPy puts them in the data type name. Fixtures are written
with zarr-python and read back with zarrista.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from zarrista import Array, Tensor
from zarrista.store import FilesystemStore


def _tensor(path: Path, values: NDArray) -> Tensor:
    z = zarr.create_array(
        store=str(path),
        shape=values.shape,
        chunks=values.shape,
        dtype=values.dtype,
    )
    z[:] = values
    tensor = Array.open(FilesystemStore(path))[:]
    assert isinstance(tensor, Tensor)
    return tensor


# One per NumPy temporal unit, plus the generic unit, which has no unit code.
@pytest.mark.parametrize(
    "unit",
    ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as", ""],
)
@pytest.mark.parametrize("kind", ["datetime64", "timedelta64"])
def test_round_trip_every_unit(tmp_path: Path, kind: str, unit: str):
    dtype = f"{kind}[{unit}]" if unit else kind
    values = np.array([0, 1, 2], dtype="int64").view(dtype)

    tensor = _tensor(tmp_path / "a.zarr", values)

    assert tensor.to_numpy().dtype == np.dtype(dtype)
    np.testing.assert_array_equal(tensor.to_numpy(), values)


@pytest.mark.parametrize("scale_factor", [1, 10, 25])
def test_scale_factor_moves_into_the_numpy_name(tmp_path: Path, scale_factor: int):
    # Zarr holds the scale factor in the configuration; NumPy spells it
    # `datetime64[10s]`. NumPy reads a scale factor of 1 as no scale factor.
    values = np.array([0, 1, 2], dtype="int64").view(f"datetime64[{scale_factor}s]")

    tensor = _tensor(tmp_path / "a.zarr", values)

    assert tensor.to_numpy().dtype == np.dtype(f"datetime64[{scale_factor}s]")
    np.testing.assert_array_equal(tensor.to_numpy(), values)


def test_not_a_time_round_trips(tmp_path: Path):
    values = np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[s]")

    tensor = _tensor(tmp_path / "a.zarr", values)

    assert np.array_equal(tensor.to_numpy(), values, equal_nan=True)


def test_array_protocol_gives_the_temporal_dtype(tmp_path: Path):
    values = np.array([0, 1, 2], dtype="int64").view("timedelta64[D]")

    tensor = _tensor(tmp_path / "a.zarr", values)

    assert np.asarray(tensor).dtype == np.dtype("timedelta64[D]")
