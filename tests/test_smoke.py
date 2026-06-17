import zarrista


def test_version_is_nonempty_string():
    assert isinstance(zarrista.__version__, str)
    assert zarrista.__version__
