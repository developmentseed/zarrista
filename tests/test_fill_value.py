"""Fill value construction from Python values.

A fill value holds its data type, because the bytes alone are ambiguous:
`-9999` is `f1 d8 ff ff` as `int32` and `00 3c 1c c5` as `float32`.
"""

from decimal import Decimal

import numpy as np
import pytest

from zarrista import DataType, FillValue
from zarrista.exceptions import FillValueError


def test_int_uses_the_bytes_of_its_data_type():
    assert FillValue(-9999, dtype="int32").as_bytes() == (-9999).to_bytes(
        4,
        "little",
        signed=True,
    )


def test_the_same_int_differs_between_data_types():
    assert FillValue(-9999, dtype="int32") != FillValue(-9999, dtype="float32")


def test_equal_values_of_one_data_type_are_equal():
    assert FillValue(0, dtype="int32") == FillValue(0, dtype="int32")


def test_zero_bytes_of_different_data_types_are_not_equal():
    """Both are four zero bytes, so only the data type tells them apart."""
    assert FillValue(0, dtype="int32") != FillValue(0.0, dtype="float32")


def test_eq_non_fill_value_is_false():
    assert FillValue(0, dtype="int32") != 0


def test_out_of_range_int_raises():
    with pytest.raises(FillValueError):
        FillValue(300, dtype="int8")


def test_float_for_an_int_data_type_raises():
    with pytest.raises(FillValueError):
        FillValue(1.5, dtype="int32")


def test_metadata_gives_the_zarr_v3_form():
    assert FillValue(-9999, dtype="int32").metadata == -9999


@pytest.mark.parametrize(
    ("value", "metadata"),
    [(float("nan"), "NaN"), (float("inf"), "Infinity"), (float("-inf"), "-Infinity")],
)
def test_non_finite_floats_use_their_spec_names(value, metadata):
    assert FillValue(value, dtype="float32").metadata == metadata


def test_complex_uses_a_two_element_array():
    assert FillValue(1 + 2j, dtype="complex64").metadata == [1.0, 2.0]


def test_float16_rounds_once():
    """A conversion through float32 would round this down to 1.0."""
    value = 1.0 + 2**-11 + 2**-30

    assert FillValue(value, dtype="float16").as_bytes() == np.float16(value).tobytes()


@pytest.mark.parametrize(
    ("value", "metadata"),
    [(np.float32(1.5), 1.5), (np.float64(1.5), 1.5), (np.int8(1), 1.0)],
)
def test_numpy_scalars_resolve_for_a_data_type_without_a_direct_branch(value, metadata):
    """A NumPy scalar is not JSON, so `item()` gives the Python value first."""
    assert FillValue(value, dtype="float8_e4m3").metadata == metadata


def test_a_numpy_complex_scalar_resolves_for_a_complex_data_type():
    assert FillValue(np.complex64(1.5 + 2j), dtype="complex64").metadata == [1.5, 2.0]


def test_a_complex_value_is_rejected_for_a_scalar_data_type():
    with pytest.raises(FillValueError):
        FillValue(np.complex64(1.5), dtype="float8_e4m3")


@pytest.mark.parametrize("dtype", ["float32", "float8_e4m3"])
def test_an_object_that_acts_like_a_float_resolves(dtype):
    """A float data type reads `__float__`, whichever path it takes."""
    assert FillValue(Decimal("1.5"), dtype=dtype).metadata == 1.5


def test_an_object_that_acts_like_a_float_is_rejected_by_an_int_data_type():
    with pytest.raises(FillValueError):
        FillValue(Decimal("1.5"), dtype="int32")


def test_numpy_bool_resolves_for_bool():
    assert FillValue(np.True_, dtype="bool").metadata is True


def test_string_data_type_takes_a_str():
    assert FillValue("missing", dtype="string").metadata == "missing"


def test_an_object_that_is_not_a_value_raises():
    with pytest.raises(FillValueError, match="not a number or a JSON value"):
        FillValue(object(), dtype="float8_e4m3")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        (-9999, "int32", np.int32(-9999)),
        (1.5, "float16", np.float16(1.5)),
        (1 + 2j, "complex64", np.complex64(1 + 2j)),
        ("missing", "string", np.str_("missing")),
        (b"ab", "bytes", np.bytes_(b"ab")),
    ],
)
def test_to_numpy_gives_a_scalar_of_the_data_type(value, dtype, expected):
    result = FillValue(value, dtype=dtype).to_numpy()

    assert result == expected
    assert result.dtype == expected.dtype


def test_to_numpy_of_a_configured_data_type():
    """The NumPy name carries the unit, which the Zarr configuration holds."""
    fill_value = FillValue(
        0,
        dtype={
            "name": "numpy.datetime64",
            "configuration": {"unit": "s", "scale_factor": 1},
        },
    )

    assert fill_value.to_numpy() == np.datetime64("1970-01-01T00:00:00", "s")


def test_to_numpy_of_nan():
    """NaN is never equal to itself, so compare the bits."""
    assert np.isnan(FillValue(float("nan"), dtype="float32").to_numpy())


def test_an_error_names_the_value_and_the_data_type():
    """The errors from NumPy and zarrs name neither, so they are wrapped."""
    with pytest.raises(
        FillValueError,
        match="cannot use 300 as a fill value of data type 'int8'",
    ) as error:
        FillValue(300, dtype="int8")

    # The text of the original error stays in the message.
    assert "out of range" in str(error.value)


def test_dtype_accepts_a_data_type():
    assert FillValue(0, dtype=DataType.from_string("int32")).dtype.name == "int32"


def test_dtype_accepts_metadata_with_a_configuration():
    """A configured data type cannot be named by a string alone."""
    fill_value = FillValue(
        0,
        dtype={
            "name": "numpy.datetime64",
            "configuration": {"unit": "s", "scale_factor": 1},
        },
    )

    assert fill_value.dtype.name == "numpy.datetime64"


def test_repr_round_trips():
    fill_value = FillValue(-9999, dtype="int32")

    assert eval(repr(fill_value)) == fill_value  # noqa: S307


def test_a_fill_value_of_another_data_type_is_rejected():
    with pytest.raises(FillValueError, match="has data type 'int32'"):
        FillValue(FillValue(1, dtype="int32"), dtype="float32")


def test_a_fill_value_of_the_same_data_type_passes_through():
    assert FillValue(FillValue(1, dtype="int32"), dtype="int32").metadata == 1
