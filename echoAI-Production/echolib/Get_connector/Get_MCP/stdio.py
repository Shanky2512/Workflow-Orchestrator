"""
STDIO MCP Connector implementation.
Handles subprocess execution with JSON input/output over stdin/stdout.
"""

from typing import Dict, Any, Optional, List
import asyncio
import json
import time
import sys
import shutil
from pathlib import Path
from .base import BaseMCPConnector, TransportType, ConnectorStatus
import logging

logger = logging.getLogger(__name__)


class STDIOMCPConnector(BaseMCPConnector):
    """
    STDIO transport MCP connector.
    
    Handles:
    - Launching arbitrary processes
    - JSON input via stdin
    - JSON output via stdout
    - Error handling via stderr
    - Timeout and process termination
    - Cross-platform command execution
    - Environment variable injection
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        auth_config: Dict[str, Any],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        command: str,
        args: Optional[List[str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
        timeout: int = 30,
        shell: bool = False,
        connector_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize STDIO connector.
        
        Args:
            command: Command to execute (path to executable or script)
            args: Command-line arguments
            env_vars: Environment variables to set
            working_dir: Working directory for process
            timeout: Process timeout in seconds
            shell: Whether to use shell execution (security risk if True)
        """
        super().__init__(
            name=name,
            description=description,
            transport_type=TransportType.STDIO,
            auth_config=auth_config,
            input_schema=input_schema,
            output_schema=output_schema,
            connector_id=connector_id,
            metadata=metadata
        )
        
        self.command = command
        self.args = args or []
        self.env_vars = env_vars or {}
        self.working_dir = working_dir
        self.timeout = timeout
        self.shell = shell
        
    def validate_config(self) -> tuple[bool, List[str]]:
        """Validate STDIO connector configuration"""
        errors = []
        
        # Validate command
        if not self.command:
            errors.append("command is required")
        else:
            # Check if command exists (if not using shell)
            if not self.shell:
                # Try to find command in PATH or as absolute path
                command_path = shutil.which(self.command)
                if not command_path and not Path(self.command).exists():
                    errors.append(f"command not found: {self.command}")
        
        # Validate working directory
        if self.working_dir and not Path(self.working_dir).exists():
            errors.append(f"working_dir does not exist: {self.working_dir}")
            
        # Validate timeout
        if self.timeout <= 0:
            errors.append("timeout must be positive")
            
        # Validate schemas
        if not self.input_schema:
            errors.append("input_schema is required")
        if not self.output_schema:
            errors.append("output_schema is required")
            
        # Security warning for shell execution
        if self.shell:
            logger.warning("Shell execution enabled - potential security risk")
            
        is_valid = len(errors) == 0
        if is_valid:
            self.update_status(ConnectorStatus.VALIDATED)
            
        return is_valid, errors
    
    def build_connector(self) -> Dict[str, Any]:
        """Build STDIO MCP connector specification"""
        return {
            "type": "stdio",
            "command": self.command,
            "args": self.args,
            "env": self.env_vars,
            "working_dir": self.working_dir,
            "timeout": self.timeout,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }
    
    async def _execute_process(
        self,
        stdin_data: str
    ) -> tuple[str, str, int]:
        """
        Execute subprocess and capture output.
        
        Returns:
            (stdout, stderr, return_code)
        """
        # Build command
        if self.shell:
            cmd = f"{self.command} {' '.join(self.args)}"
        else:
            cmd = [self.command] + self.args
        
        # Build environment
        import os
        env = os.environ.copy()
        env.update(self.env_vars)
        
        # Create subprocess
        try:
            process = await asyncio.create_subprocess_exec(
                *([cmd] if self.shell else cmd),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.working_dir,
                shell=self.shell
            )
            
            # Send input and wait for output with timeout
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(stdin_data.encode()),
                    timeout=self.timeout
                )
                
                stdout_str = stdout_data.decode('utf-8', errors='replace')
                stderr_str = stderr_data.decode('utf-8', errors='replace')
                return_code = process.returncode
                
            except asyncio.TimeoutError:
                # Kill process on timeout
                logger.warning(f"Process timeout after {self.timeout}s, killing")
                try:
                    process.kill()
                    await process.wait()
                except Exception as e:
                    logger.error(f"Failed to kill process: {e}")
                    
                raise TimeoutError(f"Process timeout after {self.timeout}s")
            
            return stdout_str, stderr_str, return_code
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Command not found: {self.command}")
    
    async def test(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test STDIO connector by executing process with payload.
        
        Sends payload as JSON to stdin and parses JSON from stdout.
        """
        start_time = time.time()
        
        try:
            # Convert payload to JSON for stdin
            stdin_json = json.dumps(payload)
            
            logger.info(f"Testing STDIO connector: {self.command}")
            
            # Execute process
            stdout, stderr, return_code = await self._execute_process(stdin_json)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Parse stdout line by line to extract JSON
            output_data = None
            for line in stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    output_data = json.loads(line)
                    break  # Stop at first valid JSON
                except json.JSONDecodeError:
                    continue  # Ignore non-JSON lines
            
            # If no JSON found, use raw stdout
            if output_data is None:
                output_data = stdout
            
            # Determine success
            success = return_code == 0
            
            result = {
                "success": success,
                "output": output_data,
                "error": stderr if stderr else None,
                "duration_ms": duration_ms,
                "metadata": {
                    "return_code": return_code,
                    "command": self.command,
                    "args": self.args,
                    "stderr": stderr[:500] if stderr else None  # Truncate stderr
                }
            }
            
            self.add_test_result(result)
            return result
            
        except TimeoutError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            result = {
                "success": False,
                "output": None,
                "error": str(e),
                "duration_ms": duration_ms,
                "metadata": {"timeout": self.timeout}
            }
            self.add_test_result(result)
            return result
            
        except FileNotFoundError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            result = {
                "success": False,
                "output": None,
                "error": str(e),
                "duration_ms": duration_ms,
                "metadata": {"command": self.command}
            }
            self.add_test_result(result)
            return result
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"STDIO test failed: {e}", exc_info=True)
            result = {
                "success": False,
                "output": None,
                "error": str(e),
                "duration_ms": duration_ms,
                "metadata": {"exception_type": type(e).__name__}
            }
            self.add_test_result(result)
            return result
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize STDIO connector with transport-specific fields"""
        base_data = super().serialize()
        base_data.update({
            "command": self.command,
            "args": self.args,
            "env_vars": self.env_vars,
            "working_dir": self.working_dir,
            "timeout": self.timeout,
            "shell": self.shell,
        })
        return base_data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'STDIOMCPConnector':
        """Deserialize STDIO connector from dict"""
        connector = cls(
            name=data["name"],
            description=data["description"],
            auth_config=data["auth_config"],
            input_schema=data["input_schema"],
            output_schema=data["output_schema"],
            command=data["command"],
            args=data.get("args", []),
            env_vars=data.get("env_vars", {}),
            working_dir=data.get("working_dir"),
            timeout=data.get("timeout", 30),
            shell=data.get("shell", False),
            connector_id=data["connector_id"],
            metadata=data.get("metadata", {})
        )
        
        connector.status = ConnectorStatus(data["status"])
        connector.created_at = data["created_at"]
        connector.updated_at = data["updated_at"]
        connector.test_results = data.get("test_results", [])
        
        return connector