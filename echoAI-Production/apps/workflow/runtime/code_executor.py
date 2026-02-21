"""
Multi-language code executor for workflow Code nodes.

Supports Python (via exec), JavaScript, TypeScript (via subprocess),
Java and C++ (via compile-then-run subprocess).
"""

import contextlib
import io
import json
import logging
import os
import subprocess
import tempfile
import types
from pathlib import Path

logger = logging.getLogger(__name__)

_SERIALIZABLE_TYPES = (str, int, float, bool, list, dict, tuple, type(None))


class CodeExecutor:
    SUPPORTED = {"python", "javascript", "typescript", "java", "cpp"}
    TIMEOUT = 30  # seconds

    @staticmethod
    def execute(code: str, language: str, state: dict) -> dict:
        """Execute code in the given language and return a result dict.

        Returns:
            {success, output, stdout, stderr, variables}
        """
        language = (language or "python").lower().strip()
        if language not in CodeExecutor.SUPPORTED:
            raise ValueError(
                f"Unsupported code language: {language}. "
                f"Supported: {', '.join(sorted(CodeExecutor.SUPPORTED))}"
            )

        if not code or not code.strip():
            raise ValueError("No code provided for Code node execution.")

        if language == "python":
            return CodeExecutor._exec_python(code, state)
        return CodeExecutor._exec_subprocess(code, language, state)

    @staticmethod
    def _exec_python(code: str, state: dict) -> dict:
        """Execute Python code via exec() with state injection."""
        exec_globals = {"__builtins__": __builtins__}
        exec_locals = {}

        for key, value in state.items():
            if not key.startswith("_") and key != "messages":
                exec_locals[key] = value

        exec_locals["previous_output"] = state.get("last_node_output")
        exec_locals["api_result"] = state.get("api_result")
        exec_locals["mcp_result"] = state.get("mcp_result")

        stdout_capture = io.StringIO()

        def _run_code():
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, exec_globals, exec_locals)

        _run_code()

        stdout_output = stdout_capture.getvalue()
        code_output = exec_locals.get("result", stdout_output)

        safe_vars = {}
        for k, v in exec_locals.items():
            if (
                not k.startswith("_")
                and k not in state
                and not callable(v)
                and not isinstance(v, types.ModuleType)
                and k not in ("previous_output", "api_result", "mcp_result")
                and isinstance(v, _SERIALIZABLE_TYPES)
            ):
                safe_vars[k] = v

        return {
            "success": True,
            "output": code_output,
            "stdout": stdout_output,
            "stderr": "",
            "variables": safe_vars,
        }

    @staticmethod
    def _exec_subprocess(code: str, language: str, state: dict) -> dict:
        """Execute non-Python code via subprocess."""
        state_vars = {}
        for key, value in state.items():
            if not key.startswith("_") and key != "messages":
                state_vars[key] = value
        state_vars["previous_output"] = state.get("last_node_output")
        state_vars["api_result"] = state.get("api_result")
        state_vars["mcp_result"] = state.get("mcp_result")

        # Make state JSON-safe (convert non-serializable values to strings)
        safe_state = {}
        for k, v in state_vars.items():
            try:
                json.dumps(v)
                safe_state[k] = v
            except (TypeError, ValueError):
                safe_state[k] = str(v)

        state_json = json.dumps(safe_state)

        builders = {
            "javascript": CodeExecutor._build_js_file,
            "typescript": CodeExecutor._build_ts_file,
            "java": CodeExecutor._build_java_file,
            "cpp": CodeExecutor._build_cpp_file,
        }
        build_fn = builders[language]

        tmp_dir = tempfile.mkdtemp(prefix="echo_code_")
        try:
            filepath = build_fn(code, safe_state, tmp_dir)
            return CodeExecutor._run_language(language, filepath, tmp_dir, state_json)
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _build_js_file(code: str, state_vars: dict, tmp_dir: str) -> str:
        preamble_lines = []
        for key, value in state_vars.items():
            preamble_lines.append(
                f"const {key} = JSON.parse({json.dumps(json.dumps(value))});"
            )
        preamble = "\n".join(preamble_lines)
        full_code = f"{preamble}\n\n{code}\n"
        filepath = os.path.join(tmp_dir, "code.cjs")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_code)
        return filepath

    @staticmethod
    def _build_ts_file(code: str, state_vars: dict, tmp_dir: str) -> str:
        preamble_lines = []
        for key, value in state_vars.items():
            preamble_lines.append(
                f"const {key}: any = JSON.parse({json.dumps(json.dumps(value))});"
            )
        preamble = "\n".join(preamble_lines)
        full_code = f"{preamble}\n\n{code}\n"
        filepath = os.path.join(tmp_dir, "code.ts")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_code)
        return filepath

    @staticmethod
    def _build_java_file(code: str, state_vars: dict, tmp_dir: str) -> str:
        prev_output = state_vars.get("previous_output", "")
        prev_str = json.dumps(str(prev_output)) if prev_output is not None else '""'

        if "class " not in code:
            code = (
                "import java.util.*;\n"
                "public class Main {\n"
                "    public static void main(String[] args) {\n"
                f"        String STATE_JSON = System.getenv(\"STATE_JSON\") != null ? System.getenv(\"STATE_JSON\") : \"{{}}\";\n"
                f"        String previous_output = {prev_str};\n"
                f"        {code}\n"
                "    }\n"
                "}\n"
            )

        filepath = os.path.join(tmp_dir, "Main.java")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        return filepath

    @staticmethod
    def _build_cpp_file(code: str, state_vars: dict, tmp_dir: str) -> str:
        prev_output = state_vars.get("previous_output", "")
        prev_str = json.dumps(str(prev_output)) if prev_output is not None else '""'

        if "int main" not in code:
            code = (
                "#include <iostream>\n"
                "#include <string>\n"
                "#include <cstdlib>\n"
                "int main() {\n"
                "    const char* env = std::getenv(\"STATE_JSON\");\n"
                "    std::string STATE_JSON = env ? env : \"{}\";\n"
                f"    std::string previous_output = {prev_str};\n"
                f"    {code}\n"
                "    return 0;\n"
                "}\n"
            )

        filepath = os.path.join(tmp_dir, "code.cpp")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        return filepath

    @staticmethod
    def _run_language(language: str, filepath: str, tmp_dir: str, state_json: str) -> dict:
        """Compile (if needed) and run the code file."""
        env = os.environ.copy()
        env["STATE_JSON"] = state_json

        if language in ("javascript", "typescript"):
            cmd = CodeExecutor._get_run_command(language, filepath)
            return CodeExecutor._run_process(cmd, env)

        if language == "java":
            # Compile
            compile_result = CodeExecutor._run_process(
                ["javac", filepath], env, phase="compilation"
            )
            if not compile_result["success"]:
                return compile_result
            # Run
            return CodeExecutor._run_process(
                ["java", "-cp", tmp_dir, "Main"], env
            )

        if language == "cpp":
            out_name = "code_out.exe" if os.name == "nt" else "code_out"
            out_path = os.path.join(tmp_dir, out_name)
            # Compile
            compile_result = CodeExecutor._run_process(
                ["g++", "-o", out_path, filepath], env, phase="compilation"
            )
            if not compile_result["success"]:
                return compile_result
            # Run
            return CodeExecutor._run_process([out_path], env)

        raise ValueError(f"No runner configured for language: {language}")

    @staticmethod
    def _get_run_command(language: str, filepath: str) -> list:
        commands = {
            "python": ["python", filepath],
            "javascript": ["node", filepath],
            "typescript": ["npx", "tsx", filepath],
        }
        return commands[language]

    @staticmethod
    def _run_process(cmd: list, env: dict, phase: str = "execution") -> dict:
        """Run a subprocess with timeout and capture output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=CodeExecutor.TIMEOUT,
                env=env,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if result.returncode != 0:
                error_msg = stderr.strip() or f"Process exited with code {result.returncode}"
                return {
                    "success": False,
                    "output": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "variables": {},
                    "error": f"Code {phase} failed: {error_msg}",
                }

            return {
                "success": True,
                "output": stdout.strip() if stdout.strip() else None,
                "stdout": stdout,
                "stderr": stderr,
                "variables": {},
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": None,
                "stdout": "",
                "stderr": "",
                "variables": {},
                "error": f"Code {phase} timed out after {CodeExecutor.TIMEOUT} seconds",
            }
        except FileNotFoundError as e:
            return {
                "success": False,
                "output": None,
                "stdout": "",
                "stderr": "",
                "variables": {},
                "error": f"Runtime not found: {e}. Ensure the language runtime is installed.",
            }
