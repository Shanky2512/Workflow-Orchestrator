import math

from .models import CalculatorInput, CalculatorResult
from AgentTools.math import arithmetic, statistics, linear_algebra


class CalculatorService:
    def calculate(self, data: CalculatorInput) -> CalculatorResult:
        op = data.operation

        if data.values is not None:
            values = data.values
            ops = {
                "add": arithmetic.add,
                "subtract": arithmetic.subtract,
                "multiply": arithmetic.multiply,
                "divide": arithmetic.divide,
                "power": arithmetic.power,
                "mod": arithmetic.mod,
                "sqrt": lambda x: arithmetic.sqrt(x[0]),
                "abs": lambda x: abs(x[0]),
                "exp": lambda x: math.exp(x[0]),
                "sum": sum,
                "mean": statistics.mean,
                "median": statistics.median,
                "variance": statistics.variance,
                "stddev": statistics.stddev,
                "min": min,
                "max": max,
            }
            if op not in ops:
                raise ValueError(f"Operation '{op}' not supported for values")
            result = ops[op](values)

        elif data.vectors is not None:
            vectors = data.vectors
            ops = {
                "dot": lambda x: linear_algebra.dot(x[0], x[1]),
                "vector_add": lambda x: linear_algebra.vector_add(x[0], x[1]),
                "vector_subtract": lambda x: linear_algebra.vector_subtract(x[0], x[1]),
                "magnitude": lambda x: linear_algebra.magnitude(x[0]),
            }
            if op not in ops:
                raise ValueError(f"Operation '{op}' not supported for vectors")
            result = ops[op](vectors)

        elif data.matrices is not None:
            matrices = data.matrices
            ops = {
                "matrix_add": lambda x: linear_algebra.matrix_add(x[0], x[1]),
                "matrix_multiply": lambda x: linear_algebra.matrix_multiply(x[0], x[1]),
                "transpose": lambda x: linear_algebra.transpose(x[0]),
            }
            if op not in ops:
                raise ValueError(f"Operation '{op}' not supported for matrices")
            result = ops[op](matrices)

        else:
            raise ValueError("Invalid input")

        if isinstance(result, (int, float)) and data.precision is not None:
            if data.rounding == "round":
                result = round(result, data.precision)
            elif data.rounding == "floor":
                result = math.floor(result)
            elif data.rounding == "ceil":
                result = math.ceil(result)

        return CalculatorResult(operation=op, result=result)
