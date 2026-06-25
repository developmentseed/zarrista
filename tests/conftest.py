"""Shared pytest configuration."""

import inspect


def pytest_collection_modifyitems(items):
    """
    Bridges pytest-asyncio and pytest-run-parallel: under `--parallel-threads`,
    pytest-run-parallel calls each test directly inside worker threads, which leaves
    `async def` tests' coroutines un-awaited (pytest-asyncio never gets to drive
    them). Mark every coroutine test as `thread_unsafe` so it runs single-threaded
    on the normal pytest-asyncio path, while synchronous tests still get the
    free-threaded parallel stress.

    The marker is a no-op when pytest-run-parallel isn't installed (Python < 3.13),
    so this is safe across the whole test matrix.
    """
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "obj", None)):
            item.add_marker("thread_unsafe")
