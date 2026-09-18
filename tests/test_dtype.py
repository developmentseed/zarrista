import pytest

from zarrista import DataType, FillValue
from zarrista.exceptions import PluginCreateError


def test_from_metadata():
    dtype = DataType.from_metadata({"name": "float32"})
    assert dtype.name == "float32"
    assert dtype.size == 4


def test_from_string():
    dtype = DataType.from_string("float32")
    assert dtype.name == "float32"
    assert dtype.size == 4


def test_from_string_matches_metadata_construction():
    from_string = DataType.from_string("float32")
    from_metadata = DataType.from_metadata({"name": "float32"})
    assert from_string == from_metadata


def test_from_string_variable_length_has_no_size():
    assert DataType.from_string("string").size is None


def test_from_string_rejects_unknown_name():
    with pytest.raises(Exception):  # noqa: B017, PT011
        DataType.from_string("not_a_real_dtype")


def test_eq_same_dtype():
    assert DataType.from_string("float32") == DataType.from_string("float32")


def test_eq_different_dtype():
    assert DataType.from_string("float32") != DataType.from_string("int8")


def test_eq_non_dtype_is_false():
    # __eq__ is strict: a string is not equal to a DataType. Conversion is
    # explicit via `from_string`, never implicit.
    assert DataType.from_string("float32") != "float32"


def test_repr():
    """The repr shows the Zarr v3 name, not the numpy descr."""
    assert repr(DataType.from_string("float32")) == "DataType('float32')"


def test_argument_accepts_a_name():
    """Every argument that takes a data type also takes its Zarr v3 name."""
    assert FillValue(0, dtype="int32").dtype == DataType.from_string("int32")


def test_argument_accepts_metadata():
    fill_value = FillValue(0, dtype={"name": "int32"})

    assert fill_value.dtype == DataType.from_string("int32")


def test_argument_rejects_an_unknown_name():
    with pytest.raises(PluginCreateError):
        FillValue(0, dtype="not_a_real_dtype")


def test_argument_rejects_a_value_that_is_not_a_data_type():
    with pytest.raises(TypeError, match="expected a DataType, a data type name"):
        FillValue(0, dtype=3)  # type: ignore[arg-type]
