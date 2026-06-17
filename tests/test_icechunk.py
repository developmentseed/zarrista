"""Open Zarr data stored in an icechunk repository with zarrista's async API.

A fixture writes an array into an icechunk repo with zarr-python, commits it, and
hands back a read-only session. zarrista reopens it from that session and the
decoded region is compared against the equivalent numpy slice.

Local filesystem storage is used (not in-memory): the session is serialized and
reconstructed inside the Rust extension, which is a separate icechunk instance,
so the data must live somewhere both processes can read.
"""

from pathlib import Path

import icechunk
import numpy as np
import pytest
import zarr
from numpy.typing import NDArray
from zarrista import AsyncArray, AsyncGroup


@pytest.fixture
def icechunk_session(tmp_path: Path) -> tuple[icechunk.Session, NDArray[np.int32]]:
    """A read-only session holding a (9, 64, 100) int32 array at `/embeddings`."""
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo"))
    )
    session = repo.writable_session("main")
    data = np.arange(9 * 64 * 100, dtype="int32").reshape(9, 64, 100)
    root = zarr.group(store=session.store)
    z = root.create_array(
        "embeddings", shape=data.shape, chunks=(3, 16, 50), dtype=data.dtype
    )
    z[:] = data
    session.commit("write embeddings")
    return repo.readonly_session("main"), data


async def test_open_array_from_session(
    icechunk_session: tuple[icechunk.Session, NDArray[np.int32]],
):
    session, data = icechunk_session

    arr = await AsyncArray.open_async(session, "/embeddings")
    result = (await arr[0:2, :, 5:7]).to_numpy()

    np.testing.assert_array_equal(result, data[0:2, :, 5:7])


async def test_open_group_from_session(
    icechunk_session: tuple[icechunk.Session, NDArray[np.int32]],
):
    session, data = icechunk_session

    group = await AsyncGroup.open_async(session)
    assert await group.array_keys() == ["embeddings"]

    arr = await group.open_child_async("embeddings")
    result = (await arr[...]).to_numpy()

    np.testing.assert_array_equal(result, data)


async def test_non_session_object_rejected():
    with pytest.raises(TypeError):
        await AsyncArray.open_async(object(), "/embeddings")
