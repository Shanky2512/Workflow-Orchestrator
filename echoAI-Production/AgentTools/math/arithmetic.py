import math
from functools import reduce
from numbers import Real
from typing import List


def _require_values(values: List[Real]) -> None:
    if not values:
        raise ValueError("At least one value is required")


def add(values: List[Real]) -> Real:
    _require_values(values)
    return sum(values)


def subtract(values: List[Real]) -> Real:
    _require_values(values)
    return reduce(lambda x, y: x - y, values)


def multiply(values: List[Real]) -> Real:
    _require_values(values)
    return reduce(lambda x, y: x * y, values)


def divide(values: List[Real]) -> Real:
    _require_values(values)

    def _div(x, y):
        if y == 0:
            raise ValueError("Division by zero")
        return x / y

    return reduce(_div, values)


def power(values: List[Real]) -> Real:
    _require_values(values)
    return reduce(lambda x, y: x ** y, values)


def mod(values: List[Real]) -> Real:
    _require_values(values)

    def _mod(x, y):
        if y == 0:
            raise ValueError("Modulo by zero")
        return x % y

    return reduce(_mod, values)


def sqrt(a: Real) -> Real:
    if a < 0:
        raise ValueError("Square root of negative number")
    return math.sqrt(a)
