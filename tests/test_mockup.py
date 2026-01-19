import pytest

from python_package import hello_world, saved_world


def test_hello_world_3times():
    expected = "hello world hello world hello world"
    result = hello_world(3)
    assert result == expected


@pytest.mark.xfail(raises=TypeError)
def test_hello_world_str():
    hello_world("3")


@pytest.fixture
def temp_file():
    # set up
    filename = "temp_hellooo.txt"
    with open(filename, "w") as f:
        f.write("hello world hello world hello world")
    yield filename
    # clean up
    import os

    os.remove(filename)


def test_saved_world_3times(temp_file):
    result = saved_world(temp_file)
    assert result == 3
