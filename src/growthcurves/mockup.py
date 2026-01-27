from itertools import repeat


def hello_world(n: int) -> str:
    """Print 'hello world' n-times.

    Parameters
    ----------
    n : int
        How many time to return hello world

    Returns
    -------
    str
        str of 'hello world' n-times

    Examples
    --------
    >>> hello_world(3)
    'hello world hello world hello world'
    """
    return " ".join(repeat("hello world", n))


def saved_world(filename: str) -> int:
    """
    Count how many times 'hello world' is in a file.

    Parameters
    ----------
    filename : str
        The file to read

    Returns
    -------
    int
        How many times 'hello world' is in the file

    Examples
    --------
    >>> saved_world("not-real.txt") # doctest: +SKIP
    """
    with open(filename, "r") as f:
        content = f.read()
    return content.count("hello world")
