"""Codec methods that take a data type and a fill value."""

import numpy as np
import pytest

from zarrista import ArrayBytes, FillValue
from zarrista.codec import transpose
from zarrista.exceptions import FillValueError

DATA = ArrayBytes(np.arange(6, dtype="int32").tobytes())


def test_encode_and_decode_take_a_plain_fill_value():
    codec = transpose([1, 0])

    encoded = codec.encode(DATA, shape=[2, 3], data_type="int32", fill_value=0)
    # `shape` is the decoded shape in both directions.
    decoded = codec.decode(encoded, shape=[2, 3], data_type="int32", fill_value=0)

    np.testing.assert_array_equal(
        np.frombuffer(decoded.bytes, dtype="int32"),
        np.arange(6),
    )


def test_encode_takes_a_fill_value_of_that_data_type():
    codec = transpose([1, 0])
    fill_value = FillValue(0, dtype="int32")

    encoded = codec.encode(DATA, shape=[2, 3], data_type="int32", fill_value=fill_value)

    assert isinstance(encoded, ArrayBytes)


def test_encode_rejects_a_fill_value_of_another_data_type():
    codec = transpose([1, 0])

    with pytest.raises(FillValueError, match="has data type 'int32'"):
        codec.encode(
            DATA,
            shape=[2, 3],
            data_type="float32",
            fill_value=FillValue(0, dtype="int32"),
        )


def test_encoded_fill_value_takes_only_the_fill_value():
    """The fill value carries its data type, so the codec needs nothing else."""
    codec = transpose([1, 0])

    encoded = codec.encoded_fill_value(FillValue(-9999, dtype="int32"))

    assert encoded == FillValue(-9999, dtype="int32")
