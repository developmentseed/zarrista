import zarrsita


def test_version_is_nonempty_string():
    assert isinstance(zarrsita.__version__, str)
    assert zarrsita.__version__


def test_hello():
    assert zarrsita.hello() == "Hello from zarrsita!"
