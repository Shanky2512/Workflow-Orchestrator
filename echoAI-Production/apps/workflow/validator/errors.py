"""
Validation error types.
"""
from typing import List


class ValidationError(Exception):
    """Base validation error."""
    pass


class SchemaValidationError(ValidationError):
    """Schema validation failed."""
    pass


class TopologyError(ValidationError):
    """Workflow topology is invalid."""
    pass


class ContractError(ValidationError):
    """Agent I/O contract violation."""
    pass


class RuntimeError(ValidationError):
    """Runtime feasibility check failed."""
    pass


class ValidationResult:
    """Result of validation with errors and warnings."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "valid": self.is_valid(),
            "errors": self.errors,
            "warnings": self.warnings
        }
