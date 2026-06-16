import zarrsita


def test_version_is_nonempty_string():
    assert isinstance(zarrsita.__version__, str)
    assert zarrsita.__version__
