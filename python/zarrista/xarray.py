"""A read-only, lazily-indexed xarray adapter for zarrista arrays.

This module wraps a zarrista `Array` so it can be used as an xarray backend
array. It is pure Python and builds only on zarrista's public API. Importing
this module requires the optional `xarray` dependency (`zarrista[xarray]`).

Only fixed-width data types are supported. Variable-length and masked decoded
layouts, slices with `step != 1`, and async stores are out of scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from xarray.backends.common import BackendArray
from xarray.core import indexing

from zarrista import Tensor

if TYPE_CHECKING:
    from zarrista import Array


class ZarristaBackendArray(BackendArray):
    """A lazily-indexed xarray backend array wrapping a zarrista `Array`.

    The wrapped `Array` is held without reading any chunk data; reads happen
    only when the array is indexed. Only fixed-width data types are supported.
    """

    def __init__(self, array: Array) -> None:
        """Wrap `array`, deriving the numpy `shape` and `dtype` from its metadata.

        Raises `NotImplementedError` for variable-length data types (those whose
        `DataType.size` is `None`), which have no fixed-width numpy layout.
        """
        if array.dtype.size is None:
            raise NotImplementedError(
                f"variable-length data type {array.dtype.name!r} is not supported",
            )
        self._array = array
        self.shape = tuple(array.shape)
        self.dtype = np.dtype(array.dtype.name)

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        """Read the region selected by `key`, returning a numpy array.

        Declares `BASIC` indexing support; xarray decomposes outer/vectorized
        indexing into a basic backend read plus numpy post-indexing.
        """
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing,
        )

    def _raw_indexing(self, key: tuple[int | slice, ...]) -> np.ndarray:
        """Read `key` (one int or slice per axis) and squeeze integer axes.

        Integer indexers are passed through to `retrieve_array_subset`, which is
        ndim-preserving (an integer keeps a length-1 axis); those axes are then
        squeezed so the result matches xarray's `BASIC` indexing contract.
        Slices with `step != 1` are not supported.
        """
        selection: list[int | slice] = []
        squeeze_axes: list[int] = []
        for axis, indexer in enumerate(key):
            if isinstance(indexer, slice):
                if indexer.step not in (None, 1):
                    raise NotImplementedError(
                        "slicing with step != 1 is not supported",
                    )
                selection.append(indexer)
            else:
                selection.append(int(indexer))
                squeeze_axes.append(axis)

        decoded = self._array.retrieve_array_subset(tuple(selection))
        if not isinstance(decoded, Tensor):
            raise NotImplementedError(
                f"data type {self._array.dtype.name!r} is not supported",
            )
        result = decoded.to_numpy()
        if squeeze_axes:
            result = np.squeeze(result, axis=tuple(squeeze_axes))
        return result


def to_dataarray(array: Array, *, name: str | None = None) -> xr.DataArray:
    """Wrap a zarrista `Array` as a lazily-indexed `xarray.DataArray`.

    Dimension names are taken from the array's `dimension_names`; entries that
    are `None` (or absent entirely) are synthesised as `dim_{i}`. The array's
    user attributes are copied onto the variable. The returned `DataArray` reads
    lazily: chunk data is fetched only when the array is indexed or computed.
    """
    backend_array = ZarristaBackendArray(array)
    names = array.dimension_names or [None] * array.ndim
    dims = [
        name_i if name_i is not None else f"dim_{i}"
        for i, name_i in enumerate(names)
    ]
    data = indexing.LazilyIndexedArray(backend_array)
    variable = xr.Variable(dims, data, attrs=dict(array.attrs))
    return xr.DataArray(variable, name=name)
