import pytest

from georges_core.utils import fortran_float, intersperse


def test_fortran():
    val = "0.31674-103"
    assert 3.1674e-104 == pytest.approx(fortran_float(val))


def test_intersperse():
    a = [1, 2, 3]
    b = "a"
    assert [1, "a", 2, "a", 3] == intersperse(a, b)
