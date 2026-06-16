from typing import Any

from numpy.typing import NDArray

class Data:
    """A decoded chunk of array data."""

    def to_numpy(self) -> NDArray[Any]:
        """Copy the decoded chunk into a NumPy array."""
