import math
from numbers import Real
from typing import List


def _require_values(values: List[Real]) -> None:
    if not values:
        raise ValueError("At least one value is required")


def mean(values: List[Real]) -> Real:
    _require_values(values)
    return sum(values) / len(values)


def median(values: List[Real]) -> Real:
    _require_values(values)
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    return (vals[mid - 1] + vals[mid]) / 2 if n % 2 == 0 else vals[mid]


def variance(values: List[Real]) -> Real:
    _require_values(values)
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def stddev(values: List[Real]) -> Real:
    return math.sqrt(variance(values))
