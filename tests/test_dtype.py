import pytest

from zarrista import DataType


def test_construct_from_metadata():
    dtype = DataType({"name": "float32"})
    assert dtype.name == "float32"
    assert dtype.size == 4


def test_variable_length_has_no_size():
    dtype = DataType({"name": "string"})
    assert dtype.size is None


def test_eq_same_dtype():
    assert DataType({"name": "float32"}) == DataType({"name": "float32"})


def test_eq_different_dtype():
    assert DataType({"name": "float32"}) != DataType({"name": "int8"})


def test_eq_non_dtype_is_false():
    # __eq__ is strict: a string is *not* equal to a DataType, even though a
    # string can be *coerced* into one at function boundaries (see below).
    assert DataType({"name": "float32"}) != "float32"


def test_repr():
    assert repr(DataType({"name": "float32"})) == "DataType(float32 / <f4)"


# The `FromPyObject` coercion (str / dict / DataType -> DataType) is currently
# only reachable through the scratch `extract_dtype` helper. Repoint these at a
# real consumer (e.g. an array-creation API) once one exists, and delete the
# scratch function.
extract_dtype = pytest.importorskip("zarrista._zarrista").extract_dtype


def test_coerce_from_string():
    assert extract_dtype("float32") == DataType({"name": "float32"})


def test_coerce_from_dtype_is_identity():
    dtype = DataType({"name": "float32"})
    assert extract_dtype(dtype) == dtype


def test_coerce_rejects_unknown_name():
    with pytest.raises(Exception):
        extract_dtype("not_a_real_dtype")
