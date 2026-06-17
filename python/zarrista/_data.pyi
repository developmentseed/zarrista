import sys
from typing import Any

from numpy.typing import NDArray

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing_extensions import Buffer

class Data(Buffer):
    """A decoded chunk of array data.

    Implements the Python buffer protocol (PEP 3118), so dtypes with a native
    buffer representation can be read zero-copy via `memoryview(data)` or
    `np.asarray(data)`. Dtypes without one raise `BufferError`.
    """

    def __buffer__(self, flags: int) -> memoryview: ...
    def to_numpy(self) -> NDArray[Any]:
        """Convert the chunk into a NumPy array.

        Zero-copy when the dtype supports the buffer protocol; otherwise the
        data is copied and converted.
        """
