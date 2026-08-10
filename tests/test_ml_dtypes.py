"""Reading the machine-learning float and sub-byte integer data types.

zarrs decodes these through its `microfloat` feature. NumPy has no built-in
equivalents, so `ml_dtypes` supplies them. Importing `ml_dtypes` registers the
data type names with NumPy, which is what lets `FixedLengthTensor.to_numpy`
resolve a name such as `bfloat16`.

zarr-python does not register these data types, so this module registers minimal
`ZDType` shims to write the fixtures. The shims use `zarr.core.dtype`, which is
an internal zarr-python API and may change between releases.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import ml_dtypes
import numpy as np
import pytest
import zarr
from zarr.core.dtype import data_type_registry
from zarr.core.dtype.common import DataTypeValidationError, HasEndianness, HasItemSize
from zarr.core.dtype.wrapper import ZDType

from zarrista import Array, FixedLengthTensor
from zarrista.store import FilesystemStore

# Every data type that zarrs and `ml_dtypes` both name identically, mapped to
# whether its fill value is an integer. zarrs also has `complex_bfloat16`, which
# `ml_dtypes` does not provide.
ML_DTYPES = {
    "bfloat16": False,
    "float4_e2m1fn": False,
    "float6_e2m3fn": False,
    "float6_e3m2fn": False,
    "float8_e3m4": False,
    "float8_e4m3": False,
    "float8_e4m3b11fnuz": False,
    "float8_e4m3fnuz": False,
    "float8_e5m2": False,
    "float8_e5m2fnuz": False,
    "float8_e8m0fnu": False,
    "int2": True,
    "int4": True,
    "uint2": True,
    "uint4": True,
}


@dataclass(frozen=True, kw_only=True)
class _MLDType(ZDType[Any, Any], HasEndianness, HasItemSize):
    """Minimal shim that lets zarr-python write an `ml_dtypes` fixture."""

    _native: ClassVar[np.dtype]
    _integral: ClassVar[bool]

    @property
    def item_size(self) -> int:
        return self._native.itemsize

    @classmethod
    def from_native_dtype(cls, dtype) -> Self:
        if dtype == cls._native:
            return cls()
        raise DataTypeValidationError(f"{dtype} is not {cls._zarr_v3_name}")

    def to_native_dtype(self) -> np.dtype:
        return self._native

    @classmethod
    def _from_json_v3(cls, data) -> Self:
        if data == cls._zarr_v3_name:
            return cls()
        raise DataTypeValidationError(f"{data} is not {cls._zarr_v3_name}")

    @classmethod
    def _from_json_v2(cls, data) -> Self:  # noqa: ARG003
        raise DataTypeValidationError(f"{cls._zarr_v3_name} has no zarr v2 form")

    def to_json(self, zarr_format) -> str:
        if zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"{self._zarr_v3_name} is a zarr v3 data type")

    def _check_scalar(self, data) -> bool:  # noqa: ARG002
        return True

    def cast_scalar(self, data) -> np.generic:
        return self._native.type(data)

    def default_scalar(self) -> np.generic:
        return self._native.type(0)

    def from_json_scalar(self, data, *, zarr_format) -> np.generic:  # noqa: ARG002
        return self._native.type(data)

    def to_json_scalar(self, data, *, zarr_format) -> int | float | str:  # noqa: ARG002
        # An integer data type needs an integer fill value in the metadata.
        # `ml_dtypes` reports kind "V" for every type, so the caller states this.
        if self._integral:
            return int(data)
        # Zarr v3 encodes a non-finite float as a string. JSON has no literal
        # for these. `float8_e8m0fnu` needs this: it has no zero, so its default
        # fill value is NaN.
        value = float(data)
        if np.isnan(value):
            return "NaN"
        if np.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value


def _register(name: str, *, integral: bool) -> type[_MLDType]:
    native = np.dtype(getattr(ml_dtypes, name))
    cls = dataclass(frozen=True, kw_only=True)(
        type(
            name,
            (_MLDType,),
            {
                "__annotations__": {},
                "dtype_cls": type(native),
                "_zarr_v3_name": name,
                "_native": native,
                "_integral": integral,
            },
        ),
    )
    data_type_registry.register(name, cls)
    return cls


for _name, _integral in ML_DTYPES.items():
    _register(_name, integral=_integral)


def _tensor(path: Path, name: str) -> tuple[FixedLengthTensor, np.ndarray]:
    native = np.dtype(getattr(ml_dtypes, name))
    values = np.arange(1, 5).astype(native)
    z = zarr.create_array(store=str(path), shape=(4,), chunks=(2,), dtype=native)
    z[:] = values
    tensor = Array.open(FilesystemStore(path))[:]
    assert isinstance(tensor, FixedLengthTensor)
    return tensor, values


@pytest.mark.parametrize("name", ML_DTYPES)
def test_round_trip_through_zarr_python(tmp_path: Path, name: str):
    tensor, values = _tensor(tmp_path / f"{name}.zarr", name)

    assert tensor.dtype.name == name
    assert tensor.to_numpy().dtype == np.dtype(getattr(ml_dtypes, name))
    assert (tensor.to_numpy() == values).all()


def test_to_numpy_is_zero_copy(tmp_path: Path):
    # `to_numpy` reads the decoded buffer through `np.frombuffer`, so it is a
    # view even though the buffer protocol has no format code for these types.
    tensor, _ = _tensor(tmp_path / "bf.zarr", "bfloat16")

    array = tensor.to_numpy()

    assert not array.flags["OWNDATA"]
    raw = np.frombuffer(tensor.buffer(), dtype="u1")
    assert array.__array_interface__["data"][0] == raw.__array_interface__["data"][0]


def test_buffer_protocol_rejects_ml_dtype(tmp_path: Path):
    # PEP 3118 has no format code for bfloat16, so only `to_numpy` works.
    tensor, _ = _tensor(tmp_path / "bf.zarr", "bfloat16")

    with pytest.raises(BufferError, match="format code"):
        memoryview(tensor)
