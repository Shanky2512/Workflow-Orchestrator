from numbers import Real
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Operation = Literal[
    # arithmetic
    "add", "subtract", "multiply", "divide", "power", "mod",

    # unary
    "sqrt", "abs", "exp",

    # statistics
    "sum", "mean", "median", "variance", "stddev", "min", "max",

    # vectors
    "dot", "vector_add", "vector_subtract", "magnitude",

    # matrices
    "matrix_add", "matrix_multiply", "transpose",
]


class CalculatorInput(BaseModel):
    operation: Operation
    values: Optional[List[Real]] = None
    vectors: Optional[List[List[Real]]] = None
    matrices: Optional[List[List[List[Real]]]] = None

    precision: Optional[int] = Field(
        default=None,
        ge=0,
        le=15,
        description="Decimal precision for rounding",
    )

    rounding: Literal["round", "floor", "ceil", "none"] = "round"

    @model_validator(mode="after")
    def validate_inputs(self):
        if self.values is None and self.vectors is None and self.matrices is None:
            raise ValueError("At least one of values, vectors, or matrices must be provided")
        return self


class CalculatorResult(BaseModel):
    operation: Operation
    result: object
