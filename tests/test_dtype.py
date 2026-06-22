import pytest

from zarrista import DataType


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
    assert repr(DataType.from_string("float32")) == "DataType(float32 / <f4)"
