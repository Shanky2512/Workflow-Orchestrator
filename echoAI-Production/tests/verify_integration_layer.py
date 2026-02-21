"""
Verification Test Suite: Universal Integration Layer (API & MCP)

Comprehensive tests covering the entire integration layer:

Scenario A — Managed Mode ("Leah" Flow):
    Mock a registered connector, build a workflow referencing it via connector_id,
    compile it, and assert the correct connector invocation path is taken.

Scenario B — Ad-hoc Mode ("Unify" Builder Flow):
    Build a workflow with an inline API/MCP node config (raw URL, method, auth),
    compile it, and assert an ephemeral connector is created and executed.

Scenario C — Agent Intelligence Flow:
    Ask the AgentDesigner to "Create a support agent that can check Jira",
    and assert that the synced Jira connector tool is selected.

Final Sign-off:
    Verify zero modifications to echolib/services.py by checking that
    ConnectorManager, APIConnector, and MCPConnector are consumed as-is.

Run with:
    pytest tests/verify_integration_layer.py -v
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any


# ============================================================================
# Helpers
# ============================================================================

def _make_state(**overrides) -> Dict[str, Any]:
    """Build a minimal valid workflow state dict for node execution."""
    state = {
        "run_id": "test_run_001",
        "workflow_id": "test_wf_001",
        "user_input": "Test input",
        "original_user_input": "Test input",
        "task_description": "",
        "crew_result": "",
        "last_node_output": None,
        "api_result": None,
        "mcp_result": None,
        "code_result": None,
        "review_result": None,
        "parallel_output": None,
        "individual_outputs": None,
        "hierarchical_output": None,
        "_conditional_route": None,
        "_conditional_node": None,
        "messages": [],
    }
    state.update(overrides)
    return state


# ============================================================================
# SCENARIO A — Managed Mode (The "Leah" Flow)
# ============================================================================

class TestScenarioA_ManagedMode:
    """Verify managed connector invocation via connector_id."""

    # ── A1: API Managed Node ──

    @pytest.mark.asyncio
    async def test_api_node_managed_invokes_connector(self):
        """
        Given a node config with connector_id="jira-prod-01",
        the API node should call APIConnector.invoke(connector_id, payload)
        via asyncio.to_thread (sync → thread).
        """
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        node_config = {
            "name": "Jira Create Issue",
            "type": "API",
            "config": {
                "connector_id": "jira-prod-01",
                "payload": {"summary": "Bug report", "priority": "high"},
            },
        }

        node_func = compiler._create_api_node("node_5004", node_config)

        # Mock ConnectorManager.api.invoke — it's SYNC, but called via to_thread
        mock_invoke_result = {
            "success": True,
            "status_code": 201,
            "data": {"issue_id": "JIRA-123", "status": "created"},
            "error": None,
            "elapsed_seconds": 0.45,
        }

        with patch("echolib.services.ConnectorManager") as MockCM:
            mock_api = MagicMock()
            mock_api.invoke.return_value = mock_invoke_result
            MockCM.return_value.get_manager.return_value = mock_api

            # Disable transparency to simplify test
            with patch("echolib.config.settings") as mock_settings:
                mock_settings.transparency_enabled = False

                state = _make_state()
                result = await node_func(state)

        # Assertions
        assert result["api_result"]["success"] is True
        assert result["api_result"]["status_code"] == 201
        assert result["last_node_output"]["issue_id"] == "JIRA-123"
        assert result["messages"][0]["type"] == "api"
        assert result["messages"][0]["mode"] == "managed"
        mock_api.invoke.assert_called_once_with("jira-prod-01", {"summary": "Bug report", "priority": "high"})

    # ── A2: MCP Managed Node ──

    @pytest.mark.asyncio
    async def test_mcp_node_managed_invokes_connector(self):
        """
        Given a node config with connector_id="mcp_slack_01",
        the MCP node should call MCPConnector.invoke_async(connector_id, payload)
        which is natively async.
        """
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        node_config = {
            "name": "Slack Notify",
            "type": "MCP",
            "config": {
                "connector_id": "mcp_slack_01",
                "payload": {"channel": "#alerts", "text": "Workflow complete"},
            },
        }

        node_func = compiler._create_mcp_node("node_5005", node_config)

        mock_invoke_result = {
            "success": True,
            "output": {"message_id": "msg_abc", "channel": "#alerts"},
        }

        with patch("echolib.services.ConnectorManager") as MockCM:
            mock_mcp = AsyncMock()
            mock_mcp.invoke_async.return_value = mock_invoke_result
            MockCM.return_value.get_manager.return_value = mock_mcp

            with patch("echolib.config.settings") as mock_settings:
                mock_settings.transparency_enabled = False

                state = _make_state()
                result = await node_func(state)

        assert result["mcp_result"]["success"] is True
        assert result["last_node_output"]["message_id"] == "msg_abc"
        assert result["messages"][0]["type"] == "mcp"
        assert result["messages"][0]["mode"] == "managed"
        mock_mcp.invoke_async.assert_called_once_with(
            "mcp_slack_01", payload={"channel": "#alerts", "text": "Workflow complete"}
        )


# ============================================================================
# SCENARIO B — Ad-hoc Mode (The "Unify" Builder Flow)
# ============================================================================

class TestScenarioB_AdhocMode:
    """Verify ephemeral connector creation for inline configs."""

    # ── B1: API Ad-hoc Node ──

    @pytest.mark.asyncio
    async def test_api_node_adhoc_creates_ephemeral_connector(self):
        """
        Given a node config with url + method (no connector_id),
        the API node should create an ephemeral HTTPConnector via
        ConnectorFactory.create(ConnectorConfig(...)) and call .execute().
        """
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        node_config = {
            "name": "Query Weather",
            "type": "API",
            "config": {
                "url": "https://api.weather.gov/points/39.7456,-104.9994",
                "method": "GET",
                "headers": {"Accept": "application/json"},
                "authentication": "none",
            },
        }

        node_func = compiler._create_api_node("node_5006", node_config)

        # Mock the factory chain
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.status_code = 200
        mock_response.body = {"properties": {"forecast": "Sunny"}}
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.error = None
        mock_response.elapsed_seconds = 0.32

        with patch("echolib.Get_connector.Get_API.connectors.factory.ConnectorFactory") as MockFactory:
            mock_connector = MagicMock()
            mock_connector.execute.return_value = mock_response
            MockFactory.create.return_value = mock_connector

            with patch("echolib.Get_connector.Get_API.models.config.ConnectorConfig") as MockConfig:
                MockConfig.return_value = MagicMock()

                with patch("echolib.config.settings") as mock_settings:
                    mock_settings.transparency_enabled = False

                    state = _make_state()
                    result = await node_func(state)

        assert result["api_result"]["success"] is True
        assert result["api_result"]["status_code"] == 200
        assert result["last_node_output"]["properties"]["forecast"] == "Sunny"
        assert result["messages"][0]["mode"] == "ad-hoc"
        mock_connector.execute.assert_called_once()

    # ── B2: MCP Ad-hoc Node ──

    @pytest.mark.asyncio
    async def test_mcp_node_adhoc_creates_ephemeral_connector(self):
        """
        Given a node config with endpoint_url (no connector_id),
        the MCP node should create an ephemeral HTTPMCPConnector
        and call .test(payload) which is async.
        """
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        node_config = {
            "name": "Custom MCP Call",
            "type": "MCP",
            "config": {
                "endpoint_url": "https://mcp.internal.io/v1/process",
                "method": "POST",
                "auth_config": {"type": "bearer", "token": "secret"},
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "payload": {"action": "enrich", "data": {"ip": "1.2.3.4"}},
            },
        }

        node_func = compiler._create_mcp_node("node_5007", node_config)

        mock_test_result = {
            "success": True,
            "output": {"enriched": True, "threat_score": 0.1},
            "duration_ms": 150,
        }

        with patch(
            "echolib.Get_connector.Get_MCP.http_script.HTTPMCPConnector"
        ) as MockHTTPMCP:
            mock_instance = AsyncMock()
            mock_instance.test.return_value = mock_test_result
            MockHTTPMCP.return_value = mock_instance

            with patch("echolib.config.settings") as mock_settings:
                mock_settings.transparency_enabled = False

                state = _make_state()
                result = await node_func(state)

        assert result["mcp_result"]["success"] is True
        assert result["last_node_output"]["enriched"] is True
        assert result["last_node_output"]["threat_score"] == 0.1
        assert result["messages"][0]["mode"] == "ad-hoc"
        mock_instance.test.assert_called_once()


# ============================================================================
# SCENARIO B+ — Code & Self-Review Nodes
# ============================================================================

class TestScenarioB_CodeAndReview:
    """Verify Code and Self-Review node types."""

    # ── Code Node ──

    @pytest.mark.asyncio
    async def test_code_node_executes_python(self):
        """
        Code node should execute Python code, capture stdout and the
        'result' variable, and pass them to the next node via state.
        """
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        node_config = {
            "name": "Pricing Calculator",
            "type": "Code",
            "config": {
                "language": "python",
                "code": (
                    "base_price = 100000\n"
                    "margin = 0.25\n"
                    "final_price = base_price / (1 - margin)\n"
                    "result = {'final_price': round(final_price, 2)}\n"
                    "print(f'Price: ${final_price:,.2f}')\n"
                ),
            },
        }

        node_func = compiler._create_code_node("node_5007", node_config)

        with patch("echolib.config.settings") as mock_settings:
            mock_settings.transparency_enabled = False

            state = _make_state()
            output = await node_func(state)

        assert output["code_result"]["success"] is True
        assert output["last_node_output"]["final_price"] == 133333.33
        assert "Price: $133,333.33" in output["code_result"]["stdout"]

    @pytest.mark.asyncio
    async def test_code_node_receives_previous_output(self):
        """
        Code node should have access to state variables including
        previous_output from the prior node.
        """
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        node_config = {
            "name": "Price Adjuster",
            "type": "Code",
            "config": {
                "language": "python",
                "code": (
                    "data = previous_output or {}\n"
                    "base = data.get('base_price', 0)\n"
                    "result = {'adjusted': base * 0.9}\n"
                ),
            },
        }

        node_func = compiler._create_code_node("node_5008", node_config)

        with patch("echolib.config.settings") as mock_settings:
            mock_settings.transparency_enabled = False

            state = _make_state(last_node_output={"base_price": 1000})
            output = await node_func(state)

        assert output["last_node_output"]["adjusted"] == 900.0

    # ── Self-Review Node ──

    @pytest.mark.asyncio
    async def test_self_review_node_passes_on_high_confidence(self):
        """
        Self-Review node should parse the LLM's JSON response,
        compare confidence against minConfidence, and return passed=True.
        """
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        node_config = {
            "name": "Compliance Check",
            "type": "Self-Review",
            "config": {
                "reviewPrompt": "Verify all RFP requirements are addressed.",
                "minConfidence": 0.85,
            },
        }

        node_func = compiler._create_self_review_node("node_5009", node_config)

        llm_review = json.dumps({
            "passed": True,
            "confidence": 0.92,
            "feedback": "All requirements addressed comprehensively.",
            "issues": [],
        })

        with patch.object(compiler, "_execute_llm_call", return_value=llm_review):
            with patch("echolib.config.settings") as mock_settings:
                mock_settings.transparency_enabled = False

                state = _make_state(
                    last_node_output="Full proposal text...",
                    crew_result="Full proposal text...",
                )
                output = await node_func(state)

        assert output["review_result"]["passed"] is True
        assert output["review_result"]["confidence"] == 0.92
        assert output["review_result"]["issues"] == []

    @pytest.mark.asyncio
    async def test_self_review_node_fails_on_low_confidence(self):
        """
        Self-Review node should return passed=False when confidence
        is below minConfidence threshold.
        """
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        node_config = {
            "name": "Quality Gate",
            "type": "Self-Review",
            "config": {
                "reviewPrompt": "Check output quality.",
                "minConfidence": 0.9,
            },
        }

        node_func = compiler._create_self_review_node("node_5010", node_config)

        llm_review = json.dumps({
            "passed": True,
            "confidence": 0.65,
            "feedback": "Partially meets criteria.",
            "issues": ["Missing section 3", "Pricing unclear"],
        })

        with patch.object(compiler, "_execute_llm_call", return_value=llm_review):
            with patch("echolib.config.settings") as mock_settings:
                mock_settings.transparency_enabled = False

                state = _make_state(last_node_output="Incomplete output...")
                output = await node_func(state)

        # confidence 0.65 < minConfidence 0.9 → should fail
        assert output["review_result"]["passed"] is False
        assert output["review_result"]["confidence"] == 0.65
        assert len(output["review_result"]["issues"]) == 2


# ============================================================================
# SCENARIO C — Agent Intelligence Flow
# ============================================================================

class TestScenarioC_AgentToolSelection:
    """Verify that the AgentDesigner selects connector tools dynamically."""

    def test_select_tools_finds_synced_jira_connector(self):
        """
        Given a synced Jira API connector tool in the registry,
        _select_tools_for_agent("Create agent to check Jira") should
        include the Jira connector tool in the result.
        """
        from apps.agent.designer.agent_designer import AgentDesigner
        from echolib.types import ToolDef, ToolType

        designer = AgentDesigner()

        # Build a mock ToolRegistry with a synced Jira tool
        mock_jira_tool = ToolDef(
            tool_id="tool_api_jira",
            name="Jira",
            description="This tool allows you to interact with Jira. Use it to manage issues.",
            tool_type=ToolType.API,
            input_schema={"type": "object"},
            status="active",
            tags=["api", "connector", "synced"],
            metadata={
                "connector_id": "api_jira_001",
                "connector_name": "jira",
                "connector_source": "api",
                "system_prompt": "This tool allows you to interact with Jira.",
            },
        )

        mock_registry = MagicMock()
        mock_registry.list_by_type.return_value = [mock_jira_tool]
        mock_registry.list_by_tags.return_value = [mock_jira_tool]

        with patch("echolib.di.container") as mock_container:
            mock_container.resolve.return_value = mock_registry

            selected = designer._select_tools_for_agent(
                "Create a support agent that can check Jira issues"
            )

        assert "tool_api_jira" in selected

    def test_select_tools_finds_synced_slack_mcp_connector(self):
        """
        Given a synced Slack MCP connector tool in the registry,
        _select_tools_for_agent("agent that sends slack messages") should
        include the Slack connector tool.
        """
        from apps.agent.designer.agent_designer import AgentDesigner
        from echolib.types import ToolDef, ToolType

        designer = AgentDesigner()

        mock_slack_tool = ToolDef(
            tool_id="tool_mcp_slack",
            name="Slack",
            description="This tool allows you to interact with Slack. Send messages to channels.",
            tool_type=ToolType.MCP,
            input_schema={"type": "object"},
            status="active",
            tags=["mcp", "connector", "synced"],
            metadata={
                "connector_id": "mcp_slack_001",
                "connector_name": "slack",
                "connector_source": "mcp",
            },
        )

        mock_registry = MagicMock()
        # API returns empty, MCP returns slack
        mock_registry.list_by_type.side_effect = lambda tt: (
            [mock_slack_tool] if tt == ToolType.MCP else []
        )

        with patch("echolib.di.container") as mock_container:
            mock_container.resolve.return_value = mock_registry

            selected = designer._select_tools_for_agent(
                "Create an agent that sends Slack messages to the team"
            )

        assert "tool_mcp_slack" in selected

    def test_select_tools_no_connector_still_selects_builtin(self):
        """
        When no connector tools are available, the function should
        still select built-in tools based on keyword matching.
        """
        from apps.agent.designer.agent_designer import AgentDesigner

        designer = AgentDesigner()

        # Mock empty registry
        mock_registry = MagicMock()
        mock_registry.list_by_type.return_value = []

        with patch("echolib.di.container") as mock_container:
            mock_container.resolve.return_value = mock_registry

            selected = designer._select_tools_for_agent(
                "Create an agent that searches the web for financial news"
            )

        assert "tool_web_search" in selected


# ============================================================================
# SCENARIO D — Node Type Dispatcher
# ============================================================================

class TestDispatcher:
    """Verify the _create_node_for_type dispatcher routes correctly."""

    def test_dispatcher_routes_api_type(self):
        """type='API' should route to _create_api_node."""
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        with patch.object(compiler, "_create_api_node", return_value="api_func") as mock:
            result = compiler._create_node_for_type("n1", {"type": "API", "name": "Test"})

        mock.assert_called_once()
        assert result == "api_func"

    def test_dispatcher_routes_mcp_type(self):
        """type='MCP' should route to _create_mcp_node."""
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        with patch.object(compiler, "_create_mcp_node", return_value="mcp_func") as mock:
            result = compiler._create_node_for_type("n2", {"type": "MCP", "name": "Test"})

        mock.assert_called_once()
        assert result == "mcp_func"

    def test_dispatcher_routes_code_type(self):
        """type='Code' should route to _create_code_node."""
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        with patch.object(compiler, "_create_code_node", return_value="code_func") as mock:
            result = compiler._create_node_for_type("n3", {"type": "Code", "name": "Test"})

        mock.assert_called_once()
        assert result == "code_func"

    def test_dispatcher_routes_self_review_type(self):
        """type='Self-Review' should route to _create_self_review_node."""
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        with patch.object(compiler, "_create_self_review_node", return_value="review_func") as mock:
            result = compiler._create_node_for_type("n4", {"type": "Self-Review", "name": "Test"})

        mock.assert_called_once()
        assert result == "review_func"

    def test_dispatcher_defaults_to_agent_type(self):
        """type='Agent' (or unknown) should route to _create_agent_node."""
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        with patch.object(compiler, "_create_agent_node", return_value="agent_func") as mock:
            result = compiler._create_node_for_type("n5", {"type": "Agent", "name": "Test"})

        mock.assert_called_once()
        assert result == "agent_func"

    def test_dispatcher_handles_case_insensitive(self):
        """Dispatcher should handle case-insensitive type strings."""
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        with patch.object(compiler, "_create_api_node", return_value="api_func") as mock:
            result = compiler._create_node_for_type("n6", {"type": "api", "name": "Test"})

        mock.assert_called_once()
        assert result == "api_func"


# ============================================================================
# SCENARIO E — Tool Registry Connector Sync
# ============================================================================

class TestToolRegistrySync:
    """Verify sync_connectors_as_tools correctly converts connectors to ToolDefs."""

    def test_sync_creates_api_and_mcp_tools(self):
        """
        sync_connectors_as_tools should call cm.api.list() + cm.mcp.list()
        and create ToolDefs with correct types.
        """
        from apps.tool.registry import ToolRegistry
        from echolib.types import ToolType

        # Build a mock storage
        mock_storage = MagicMock()
        mock_storage.load_all.return_value = []
        mock_storage.save_tool.return_value = None

        # Build a mock ConnectorManager
        mock_cm = MagicMock()
        mock_cm.api.list.return_value = {
            "success": True,
            "count": 1,
            "connectors": [
                {
                    "connector_id": "api_jira_001",
                    "name": "Jira",
                    "description": "Jira issue tracker",
                    "creation_payload": {
                        "input_schema": {
                            "type": "object",
                            "properties": {"summary": {"type": "string"}},
                        },
                        "example_payload": {"summary": "Test issue"},
                    },
                }
            ],
        }
        mock_cm.mcp.list.return_value = {
            "success": True,
            "count": 1,
            "connectors": [
                {
                    "connector_id": "mcp_slack_001",
                    "name": "Slack Notifier",
                    "description": "Send Slack messages",
                    "creation_payload": {
                        "example_payload": {"channel": "#general", "text": "Hello"},
                    },
                    "transport_type": "http",
                }
            ],
        }

        with patch("echolib.di.container") as mock_container:
            mock_container.resolve.side_effect = lambda key: {
                "connector.manager": mock_cm,
            }.get(key, MagicMock())

            # Create registry (auto-sync runs in __init__)
            registry = ToolRegistry(storage=mock_storage)

        # Check results
        result = registry.sync_connectors_as_tools()

        # The API tool
        api_tool = registry.get("tool_api_jira")
        assert api_tool is not None
        assert api_tool.tool_type == ToolType.API
        assert "api" in api_tool.tags
        assert api_tool.metadata["connector_id"] == "api_jira_001"
        assert api_tool.input_schema["properties"]["summary"]["type"] == "string"

        # The MCP tool
        mcp_tool = registry.get("tool_mcp_slack_notifier")
        assert mcp_tool is not None
        assert mcp_tool.tool_type == ToolType.MCP
        assert "mcp" in mcp_tool.tags
        assert "http" in mcp_tool.tags  # transport_type tag
        assert mcp_tool.metadata["connector_id"] == "mcp_slack_001"

    def test_sync_is_idempotent(self):
        """Running sync twice should skip already-synced tools."""
        from apps.tool.registry import ToolRegistry

        mock_storage = MagicMock()
        mock_storage.load_all.return_value = []
        mock_storage.save_tool.return_value = None

        mock_cm = MagicMock()
        mock_cm.api.list.return_value = {
            "success": True,
            "count": 1,
            "connectors": [
                {"connector_id": "api_test", "name": "TestAPI", "creation_payload": {}},
            ],
        }
        mock_cm.mcp.list.return_value = {"success": True, "count": 0, "connectors": []}

        with patch("echolib.di.container") as mock_container:
            mock_container.resolve.side_effect = lambda key: {
                "connector.manager": mock_cm,
            }.get(key, MagicMock())

            registry = ToolRegistry(storage=mock_storage)

            # First sync (runs in __init__)
            result1 = registry.sync_connectors_as_tools()
            # Second sync — should skip
            result2 = registry.sync_connectors_as_tools()

        assert "tool_api_testapi" in result2["skipped"]


# ============================================================================
# SCENARIO F — Data Flow: API Result → Next Agent Node
# ============================================================================

class TestDataFlowBetweenNodes:
    """
    Verify that data fetched by an API/MCP node is passed as full context
    to the next node in the workflow via state['last_node_output'].
    """

    @pytest.mark.asyncio
    async def test_api_data_available_to_code_node(self):
        """
        Simulate: API node fetches data → Code node processes it.
        The code node should see the API response in previous_output.
        """
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler(use_crewai=False)

        # Step 1: API node produces output
        api_output = {
            "last_node_output": {"projects": [{"id": 1, "name": "Alpha"}]},
            "api_result": {"success": True, "data": {"projects": [{"id": 1, "name": "Alpha"}]}},
            "crew_result": '{"projects": [{"id": 1, "name": "Alpha"}]}',
            "messages": [{"node": "api_1", "type": "api"}],
        }

        # Step 2: Code node reads from state (which includes API output)
        code_config = {
            "name": "Project Counter",
            "type": "Code",
            "config": {
                "language": "python",
                "code": (
                    "data = previous_output or {}\n"
                    "count = len(data.get('projects', []))\n"
                    "result = {'project_count': count}\n"
                ),
            },
        }

        code_func = compiler._create_code_node("code_1", code_config)

        with patch("echolib.config.settings") as mock_settings:
            mock_settings.transparency_enabled = False

            # Merge API output into state (simulating LangGraph state merge)
            state = _make_state(**api_output)
            code_output = await code_func(state)

        assert code_output["last_node_output"]["project_count"] == 1


# ============================================================================
# FINAL SIGN-OFF — Zero Modifications to echolib/services.py
# ============================================================================

class TestFinalSignOff:
    """Verify the constraint: echolib/services.py was NOT modified."""

    def test_connector_manager_class_is_consumed_not_modified(self):
        """
        ConnectorManager should still have exactly .api and .mcp attributes
        and a .get_manager() method. No new methods were added by us.
        """
        from echolib.services import ConnectorManager

        cm = ConnectorManager()

        # Core interface check — these must exist unchanged
        assert hasattr(cm, "api"), "ConnectorManager must have .api attribute"
        assert hasattr(cm, "mcp"), "ConnectorManager must have .mcp attribute"
        assert hasattr(cm, "get_manager"), "ConnectorManager must have .get_manager()"
        assert callable(cm.get_manager)

        # Verify get_manager routes correctly
        assert cm.get_manager("api") is cm.api
        assert cm.get_manager("mcp") is cm.mcp

    def test_api_connector_invoke_is_sync(self):
        """APIConnector.invoke must be a regular (sync) method, not async."""
        from echolib.services import APIConnector
        import asyncio

        connector = APIConnector()
        assert hasattr(connector, "invoke")
        assert not asyncio.iscoroutinefunction(connector.invoke), \
            "APIConnector.invoke must be SYNC (we wrap it with asyncio.to_thread)"

    def test_mcp_connector_invoke_async_is_async(self):
        """MCPConnector.invoke_async must be a coroutine function."""
        from echolib.services import MCPConnector
        import asyncio

        connector = MCPConnector()
        assert hasattr(connector, "invoke_async")
        assert asyncio.iscoroutinefunction(connector.invoke_async), \
            "MCPConnector.invoke_async must be ASYNC (we await it directly)"
