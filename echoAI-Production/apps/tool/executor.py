"""
Tool Executor Module

This module provides the ToolExecutor class for executing tools registered
in the EchoAI system. It handles invocation of local Python functions,
MCP server calls, and external API requests.

The executor manages the lifecycle of tool execution including input validation,
execution routing, error handling, and output validation.
"""

import asyncio
import importlib
import logging
import time
from typing import Dict, Any, Optional

from echolib.types import ToolDef, ToolType, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executor for invoking tools across different execution contexts.

    The ToolExecutor is responsible for:
    - Routing tool invocations to the appropriate handler (local/MCP/API)
    - Validating inputs and outputs against tool schemas
    - Managing async execution and timeouts
    - Error handling and result formatting

    Attributes:
        registry: Reference to ToolRegistry for resolving tool definitions
        default_timeout: Default timeout for tool execution in seconds
        _local_instances: Cache for local tool class instances
    """

    def __init__(self, registry: Any, default_timeout: int = 60, connector_manager: Any = None):
        """
        Initialize the ToolExecutor.

        Args:
            registry: ToolRegistry instance for tool lookups
            default_timeout: Default execution timeout in seconds
            connector_manager: Optional ConnectorManager for managed tool execution
        """
        self.registry = registry
        self.default_timeout = default_timeout
        self.connector_manager = connector_manager
        self._local_instances: Dict[str, Any] = {}

        logger.info(
            f"ToolExecutor initialized with default timeout={default_timeout}s"
        )

    def _get_connector_manager(self):
        """
        Get ConnectorManager instance. Uses injected instance first,
        falls back to lazy DI resolution.

        Returns:
            ConnectorManager instance or None if unavailable
        """
        if self.connector_manager:
            return self.connector_manager
        try:
            from echolib.di import container
            return container.resolve('connector.manager')
        except (KeyError, ImportError):
            return None

    async def invoke(
        self,
        tool_id: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Invoke a tool with the given input data.

        This is the main entry point for tool execution. It handles:
        - Tool lookup
        - Input validation
        - Routing to appropriate execution handler
        - Output validation
        - Error handling

        Args:
            tool_id: Unique identifier of the tool to invoke
            input_data: Input parameters for the tool
            context: Optional context (agent_id, workflow_id, etc.)

        Returns:
            ToolResult with output data and execution metadata

        Raises:
            ValueError: If tool not found or input validation fails
            TimeoutError: If execution exceeds timeout
            RuntimeError: If execution fails
        """
        start_time = time.time()
        context = context or {}

        # Get tool from registry
        tool = self.registry.get(tool_id)
        if not tool:
            logger.warning(f"Tool '{tool_id}' not found in registry")
            return ToolResult(
                name="unknown",
                tool_id=tool_id,
                output={},
                success=False,
                error=f"Tool '{tool_id}' not found",
                metadata={"context": context}
            )

        logger.info(f"Invoking tool '{tool.name}' (id={tool_id}, type={tool.tool_type})")

        try:
            # Validate input against schema
            self._validate_input(tool, input_data)

            # Route to appropriate executor based on tool type
            if tool.tool_type == ToolType.LOCAL:
                result = await self._execute_local(tool, input_data)
            elif tool.tool_type == ToolType.MCP:
                result = await self._execute_mcp(tool, input_data)
            elif tool.tool_type == ToolType.API:
                result = await self._execute_api(tool, input_data)
            elif tool.tool_type == ToolType.CREWAI:
                # CrewAI tools are handled internally by CrewAI framework
                result = await self._execute_local(tool, input_data)
            else:
                raise ValueError(f"Unknown tool type: {tool.tool_type}")

            # Validate output against schema (soft validation - log warning only)
            self._validate_output(tool, result)

            execution_time = time.time() - start_time

            logger.info(
                f"Tool '{tool.name}' executed successfully in {execution_time:.3f}s"
            )

            return ToolResult(
                name=tool.name,
                tool_id=tool_id,
                output=result,
                success=True,
                error=None,
                metadata={
                    "execution_time": execution_time,
                    "tool_type": tool.tool_type.value,
                    "tool_version": tool.version,
                    "context": context
                }
            )

        except ValueError as e:
            # Input validation or known errors
            execution_time = time.time() - start_time
            logger.warning(f"Tool '{tool.name}' validation error: {e}")
            return ToolResult(
                name=tool.name,
                tool_id=tool_id,
                output={},
                success=False,
                error=str(e),
                metadata={
                    "execution_time": execution_time,
                    "error_type": "validation_error",
                    "context": context
                }
            )

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"Tool '{tool.name}' execution timed out after {execution_time:.3f}s")
            return ToolResult(
                name=tool.name,
                tool_id=tool_id,
                output={},
                success=False,
                error=f"Execution timed out after {self.default_timeout}s",
                metadata={
                    "execution_time": execution_time,
                    "error_type": "timeout_error",
                    "context": context
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool '{tool.name}' execution failed: {e}", exc_info=True)
            return ToolResult(
                name=tool.name,
                tool_id=tool_id,
                output={},
                success=False,
                error=f"Execution failed: {str(e)}",
                metadata={
                    "execution_time": execution_time,
                    "error_type": "execution_error",
                    "context": context
                }
            )

    def _validate_input(self, tool: ToolDef, input_data: Dict[str, Any]) -> None:
        """
        Validate input data against tool's input schema.

        Args:
            tool: ToolDef containing the input schema
            input_data: Input data to validate

        Raises:
            ValueError: If validation fails with detailed error message
        """
        # If no input schema defined, skip validation
        if not tool.input_schema:
            logger.debug(f"Tool '{tool.name}' has no input schema, skipping validation")
            return

        try:
            import jsonschema
            jsonschema.validate(instance=input_data, schema=tool.input_schema)
            logger.debug(f"Input validation passed for tool '{tool.name}'")

        except jsonschema.ValidationError as e:
            error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            error_msg = f"Input validation failed at '{error_path}': {e.message}"
            logger.warning(f"Tool '{tool.name}': {error_msg}")
            raise ValueError(error_msg)

        except jsonschema.SchemaError as e:
            error_msg = f"Invalid input schema for tool '{tool.name}': {e.message}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _validate_output(self, tool: ToolDef, output_data: Dict[str, Any]) -> None:
        """
        Validate output data against tool's output schema.

        Output validation is softer than input validation - it logs warnings
        but doesn't fail the execution. This allows tools to return extra
        fields or slightly different structures.

        Args:
            tool: ToolDef containing the output schema
            output_data: Output data to validate
        """
        # If no output schema defined, skip validation
        if not tool.output_schema:
            logger.debug(f"Tool '{tool.name}' has no output schema, skipping validation")
            return

        try:
            import jsonschema
            jsonschema.validate(instance=output_data, schema=tool.output_schema)
            logger.debug(f"Output validation passed for tool '{tool.name}'")

        except jsonschema.ValidationError as e:
            # Log warning but don't fail - output validation is soft
            error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            logger.warning(
                f"Tool '{tool.name}' output validation warning at '{error_path}': {e.message}"
            )

        except jsonschema.SchemaError as e:
            logger.warning(
                f"Invalid output schema for tool '{tool.name}': {e.message}"
            )

    async def _execute_local(self, tool: ToolDef, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a local Python function tool.

        This method dynamically imports the specified module, instantiates
        the class (with caching), and calls the method with input_data.

        Args:
            tool: ToolDef containing execution_config with module, class, method
            input_data: Validated input parameters

        Returns:
            Dict containing the tool execution result

        Raises:
            ValueError: If execution_config is missing required fields
            ImportError: If module cannot be imported
            AttributeError: If class or method doesn't exist
        """
        config = tool.execution_config
        if not config:
            raise ValueError(f"Tool '{tool.name}' has no execution_config")

        module_path = config.get("module")
        class_name = config.get("class")
        method_name = config.get("method")

        if not all([module_path, class_name, method_name]):
            raise ValueError(
                f"Tool '{tool.name}' execution_config missing required fields: "
                f"module={module_path}, class={class_name}, method={method_name}"
            )

        logger.debug(
            f"Executing local tool: {module_path}.{class_name}.{method_name}"
        )

        try:
            # Dynamic import
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Cannot import module '{module_path}' for tool '{tool.name}': {e}"
            )

        try:
            # Get class
            cls = getattr(module, class_name)
        except AttributeError:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_path}'"
            )

        # Create cache key and get or create instance
        cache_key = f"{module_path}.{class_name}"

        if cache_key not in self._local_instances:
            try:
                self._local_instances[cache_key] = cls()
                logger.debug(f"Created new instance for {cache_key}")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot instantiate class '{class_name}': {e}"
                )

        instance = self._local_instances[cache_key]

        try:
            # Get method
            method = getattr(instance, method_name)
        except AttributeError:
            raise AttributeError(
                f"Method '{method_name}' not found in class '{class_name}'"
            )

        # Execute method (handle both sync and async)
        try:
            if asyncio.iscoroutinefunction(method):
                # Async method - apply timeout
                result = await asyncio.wait_for(
                    method(input_data),
                    timeout=self.default_timeout
                )
            else:
                # Sync method - run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, method, input_data),
                    timeout=self.default_timeout
                )

        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Tool '{tool.name}' execution timed out"
            )

        # Normalize result to dict
        return self._normalize_result(result)

    async def _execute_mcp(self, tool: ToolDef, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool via MCP (Model Context Protocol) server.

        This method calls the MCP connector endpoint to invoke the tool.

        Args:
            tool: ToolDef containing execution_config with connector_id
            input_data: Validated input parameters

        Returns:
            Dict containing the MCP response

        Raises:
            ValueError: If connector_id is missing
            RuntimeError: If MCP call fails
        """
        config = tool.execution_config
        if not config:
            raise ValueError(f"Tool '{tool.name}' has no execution_config")

        connector_id = config.get("connector_id")
        if not connector_id:
            raise ValueError(
                f"Tool '{tool.name}' execution_config missing 'connector_id'"
            )

        logger.debug(f"Executing MCP tool via connector '{connector_id}'")

        # MANAGED MODE: Delegate to ConnectorManager when available
        cm = self._get_connector_manager()
        if cm is not None:
            # mcp.invoke_async() is ASYNC — await directly, no thread wrapping
            result = await cm.mcp.invoke_async(
                connector_id=connector_id,
                payload=input_data
            )
            return result if isinstance(result, dict) else {"result": result}

        # Legacy fallback: direct HTTP call (only if ConnectorManager unavailable)
        payload = {
            "connector_id": connector_id,
            "payload": input_data
        }

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/connectors/mcp/invoke",
                    json=payload,
                    timeout=self.default_timeout
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    error_detail = response.text
                    raise RuntimeError(
                        f"MCP call failed with status {response.status_code}: {error_detail}"
                    )

        except httpx.TimeoutException:
            raise asyncio.TimeoutError(
                f"MCP call to connector '{connector_id}' timed out"
            )

        except httpx.HTTPError as e:
            raise RuntimeError(f"MCP HTTP error: {e}")

        except ImportError:
            raise ImportError(
                "httpx package is required for MCP tool execution. "
                "Install it with: pip install httpx"
            )

    async def _execute_api(self, tool: ToolDef, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool via direct HTTP API call.

        Args:
            tool: ToolDef containing execution_config with url, method, headers
            input_data: Validated input parameters

        Returns:
            Dict containing the API response

        Raises:
            ValueError: If URL is missing
            RuntimeError: If API call fails
        """
        config = tool.execution_config
        if not config:
            raise ValueError(f"Tool '{tool.name}' has no execution_config")

        # MANAGED MODE: Delegate to ConnectorManager when connector_id is present
        if "connector_id" in config:
            cm = self._get_connector_manager()
            if cm is not None:
                # api.invoke() is SYNC — wrap in thread to avoid blocking async event loop
                result = await asyncio.to_thread(
                    cm.api.invoke,
                    config["connector_id"],
                    input_data
                )
                return result if isinstance(result, dict) else {"result": result}
            else:
                raise RuntimeError(
                    f"Tool '{tool.name}' requires ConnectorManager for managed API execution, "
                    "but ConnectorManager is not available"
                )

        # Legacy/Manual API tool logic (Raw HTTP via httpx)
        url = config.get("url")
        if not url:
            raise ValueError(
                f"Tool '{tool.name}' execution_config missing 'url'"
            )

        http_method = config.get("method", "POST").upper()
        headers = config.get("headers", {})
        timeout = config.get("timeout", self.default_timeout)

        logger.debug(f"Executing API tool: {http_method} {url}")

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                if http_method == "GET":
                    response = await client.get(
                        url,
                        params=input_data,
                        headers=headers,
                        timeout=timeout
                    )
                else:
                    response = await client.request(
                        method=http_method,
                        url=url,
                        json=input_data,
                        headers=headers,
                        timeout=timeout
                    )

                if response.status_code >= 200 and response.status_code < 300:
                    # Try to parse as JSON, fall back to text
                    try:
                        return response.json()
                    except Exception:
                        return {"response": response.text}
                else:
                    error_detail = response.text
                    raise RuntimeError(
                        f"API call failed with status {response.status_code}: {error_detail}"
                    )

        except httpx.TimeoutException:
            raise asyncio.TimeoutError(f"API call to '{url}' timed out")

        except httpx.HTTPError as e:
            raise RuntimeError(f"API HTTP error: {e}")

        except ImportError:
            raise ImportError(
                "httpx package is required for API tool execution. "
                "Install it with: pip install httpx"
            )

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """
        Normalize tool execution result to a dictionary.

        Args:
            result: Raw result from tool method (dict, Pydantic model, or other)

        Returns:
            Dict representation of the result
        """
        if result is None:
            return {"result": None}

        if isinstance(result, dict):
            return result

        # Check for Pydantic model
        if hasattr(result, 'model_dump'):
            return result.model_dump()

        # Check for older Pydantic (v1) with dict() method
        if hasattr(result, 'dict') and callable(result.dict):
            return result.dict()

        # Wrap primitive types
        return {"result": result}

    def clear_instance_cache(self) -> int:
        """
        Clear the local tool instance cache.

        Returns:
            Number of instances cleared
        """
        count = len(self._local_instances)
        self._local_instances.clear()
        logger.info(f"Cleared {count} cached tool instances")
        return count

    def get_cached_instances(self) -> Dict[str, str]:
        """
        Get list of cached tool instances.

        Returns:
            Dict mapping cache_key to class name
        """
        return {
            key: type(instance).__name__
            for key, instance in self._local_instances.items()
        }
