"""
Connector testing and validation execution.
Runs real tests against connectors to verify functionality.
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from .base import BaseMCPConnector, ConnectorStatus
from .http_script import HTTPMCPConnector
from .sse import SSEMCPConnector
from .stdio import STDIOMCPConnector

logger = logging.getLogger(__name__)


class TestResult:
    """Structured test result"""
    
    def __init__(
        self,
        success: bool,
        output: Any,
        error: Optional[str],
        duration_ms: int,
        metadata: Dict[str, Any]
    ):
        self.success = success
        self.output = output
        self.error = error
        self.duration_ms = duration_ms
        self.metadata = metadata
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata
        }
        
    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"<TestResult {status} {self.duration_ms}ms>"


class ConnectorTester:
    """
    Executes tests against connectors.
    
    Responsibilities:
    - Run validation before testing
    - Execute real test calls
    - Aggregate test results
    - Generate test reports
    """
    
    @staticmethod
    async def test_connector(
        connector: BaseMCPConnector,
        test_payload: Dict[str, Any],
        validate_first: bool = True
    ) -> TestResult:
        """
        Test a connector with given payload.
        
        Args:
            connector: Connector instance to test
            test_payload: Test input data
            validate_first: Whether to validate config before testing
            
        Returns:
            TestResult with execution details
        """
        try:
            # Validate if requested
            if validate_first:
                is_valid, errors = connector.validate_config()
                if not is_valid:
                    return TestResult(
                        success=False,
                        output=None,
                        error=f"Validation failed: {', '.join(errors)}",
                        duration_ms=0,
                        metadata={"validation_errors": errors}
                    )
            
            # Execute test
            logger.info(f"Testing connector {connector.name}")
            result_dict = await connector.test(test_payload)
            
            return TestResult(
                success=result_dict["success"],
                output=result_dict["output"],
                error=result_dict.get("error"),
                duration_ms=result_dict["duration_ms"],
                metadata=result_dict.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}", exc_info=True)
            return TestResult(
                success=False,
                output=None,
                error=str(e),
                duration_ms=0,
                metadata={"exception_type": type(e).__name__}
            )
    
    @staticmethod
    async def test_multiple(
        connector: BaseMCPConnector,
        test_cases: List[Dict[str, Any]]
    ) -> List[TestResult]:
        """
        Run multiple test cases against a connector.
        
        Args:
            connector: Connector to test
            test_cases: List of test payloads
            
        Returns:
            List of test results
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test case {i+1}/{len(test_cases)}")
            result = await ConnectorTester.test_connector(
                connector,
                test_case,
                validate_first=(i == 0)  # Only validate once
            )
            results.append(result)
            
        return results
    
    @staticmethod
    def generate_report(
        connector: BaseMCPConnector,
        test_results: List[TestResult]
    ) -> Dict[str, Any]:
        """
        Generate test report from results.
        
        Args:
            connector: Tested connector
            test_results: List of test results
            
        Returns:
            Comprehensive test report
        """
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.success)
        failed_tests = total_tests - passed_tests
        
        avg_duration = (
            sum(r.duration_ms for r in test_results) / total_tests
            if total_tests > 0 else 0
        )
        
        return {
            "connector_id": connector.connector_id,
            "connector_name": connector.name,
            "transport_type": connector.transport_type.value,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "avg_duration_ms": avg_duration
            },
            "results": [r.to_dict() for r in test_results],
            "status": connector.status.value
        }
    
    @staticmethod
    async def validate_and_test(
        connector: BaseMCPConnector,
        test_payload: Dict[str, Any]
    ) -> tuple[bool, List[str], Optional[TestResult]]:
        """
        Validate config and test in one operation.
        
        Returns:
            (is_valid, validation_errors, test_result)
        """
        # Validate
        is_valid, errors = connector.validate_config()
        
        if not is_valid:
            return is_valid, errors, None
            
        # Test
        test_result = await ConnectorTester.test_connector(
            connector,
            test_payload,
            validate_first=False
        )
        
        return True, [], test_result
    
    @staticmethod
    async def dry_run(connector: BaseMCPConnector) -> Dict[str, Any]:
        """
        Perform dry run without actual execution.
        
        Validates config and builds connector spec without testing.
        Useful for verifying configuration before deployment.
        """
        # Validate
        is_valid, errors = connector.validate_config()
        
        # Build spec
        try:
            spec = connector.build_connector()
        except Exception as e:
            spec = None
            errors.append(f"Failed to build spec: {e}")
            is_valid = False
        
        return {
            "valid": is_valid,
            "errors": errors,
            "connector_spec": spec,
            "mcp_spec": connector.get_mcp_spec() if is_valid else None
        }


class TestSuite:
    """
    Manages collections of test cases for connectors.
    
    Allows organizing and running predefined test suites.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.test_cases: List[Dict[str, Any]] = []
        
    def add_test_case(
        self,
        name: str,
        payload: Dict[str, Any],
        expected_success: bool = True
    ):
        """Add test case to suite"""
        self.test_cases.append({
            "name": name,
            "payload": payload,
            "expected_success": expected_success
        })
        
    async def run(
        self,
        connector: BaseMCPConnector
    ) -> Dict[str, Any]:
        """
        Run all test cases in suite.
        
        Returns:
            Suite execution report
        """
        results = []
        
        for test_case in self.test_cases:
            result = await ConnectorTester.test_connector(
                connector,
                test_case["payload"],
                validate_first=False
            )
            
            # Check if result matches expectation
            matches_expectation = result.success == test_case["expected_success"]
            
            results.append({
                "test_name": test_case["name"],
                "result": result.to_dict(),
                "expected_success": test_case["expected_success"],
                "matches_expectation": matches_expectation
            })
            
        passed = sum(1 for r in results if r["matches_expectation"])
        
        return {
            "suite_name": self.name,
            "total_tests": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "results": results
        }


def create_default_test_payload(transport_type: str) -> Dict[str, Any]:
    """
    Generate default test payload based on transport type.
    
    Useful for quick testing.
    """
    if transport_type == "http":
        return {
            "test": True,
            "message": "Default HTTP test payload"
        }
    elif transport_type == "sse":
        return {
            "max_events": 3,
            "duration_seconds": 5
        }
    elif transport_type == "stdio":
        return {
            "input": "test",
            "echo": True
        }
    else:
        return {}