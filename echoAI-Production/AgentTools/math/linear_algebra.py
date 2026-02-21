from numbers import Real
from typing import List

Vector = List[Real]
Matrix = List[List[Real]]


def dot(v1: Vector, v2: Vector) -> Real:
    if len(v1) != len(v2):
        raise ValueError("Vectors must be same length")
    return sum(a * b for a, b in zip(v1, v2))


def vector_add(v1: Vector, v2: Vector) -> Vector:
    if len(v1) != len(v2):
        raise ValueError("Vectors must be same length")
    return [a + b for a, b in zip(v1, v2)]


def vector_subtract(v1: Vector, v2: Vector) -> Vector:
    if len(v1) != len(v2):
        raise ValueError("Vectors must be same length")
    return [a - b for a, b in zip(v1, v2)]


def magnitude(v: Vector) -> Real:
    return sum(x * x for x in v) ** 0.5


def matrix_add(m1: Matrix, m2: Matrix) -> Matrix:
    if len(m1) != len(m2) or any(len(r1) != len(r2) for r1, r2 in zip(m1, m2)):
        raise ValueError("Matrices must have the same dimensions")
    return [
        [a + b for a, b in zip(r1, r2)]
        for r1, r2 in zip(m1, m2)
    ]


def transpose(m: Matrix) -> Matrix:
    return list(map(list, zip(*m)))


def matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix:
    if not m1 or not m2:
        raise ValueError("Matrices must be non-empty")
    if len(m1[0]) != len(m2):
        raise ValueError("Matrix dimensions are not aligned for multiplication")
    t = transpose(m2)
    return [
        [sum(a * b for a, b in zip(row, col)) for col in t]
        for row in m1
    ]
