"""
Test Suite for CrewAI Adapter Module

Tests the integration layer between EchoAI workflows and CrewAI.
Validates that the adapter correctly creates LangGraph node functions
that execute CrewAI crews for agent collaboration.

Architecture principles tested:
- LangGraph owns workflow topology
- CrewAI executes within nodes only
- No orchestration logic in CrewAI
- State flows correctly: LangGraph → CrewAI → LangGraph
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any, List

from apps.workflow.crewai_adapter import (
    CrewAIAdapter,
    create_crewai_merge_node
)


class TestCrewAIAdapter:
    """Test suite for CrewAIAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create a fresh CrewAI adapter instance for each test."""
        return CrewAIAdapter()

    @pytest.fixture
    def sample_agent_config(self):
        """Sample agent configuration."""
        return {
            "agent_id": "agt_test_001",
            "name": "Test Agent",
            "role": "Test Executor",
            "goal": "Execute test tasks successfully",
            "description": "A test agent for unit testing",
            "input_schema": ["user_input"],
            "output_schema": ["output"],
            "llm": {
                "provider": "openrouter",
                "model": "nvidia/nemotron-3-nano-30b-a3b:free",
                "temperature": 0.7
            }
        }

    @pytest.fixture
    def sample_state(self):
        """Sample LangGraph state."""
        return {
            "user_input": "Test input data",
            "task_description": "Test task description",
            "expected_output": "Test expected output",
            "messages": []
        }

    @pytest.fixture
    def master_agent_config(self):
        """Sample master/manager agent configuration."""
        return {
            "agent_id": "agt_master_001",
            "name": "Project Manager",
            "role": "Manager",
            "goal": "Coordinate team to complete project",
            "description": "Experienced project manager",
            "llm": {
                "provider": "openrouter",
                "model": "nvidia/nemotron-3-nano-30b-a3b:free"
            }
        }

    @pytest.fixture
    def worker_agent_configs(self):
        """Sample worker agent configurations."""
        return [
            {
                "agent_id": "agt_worker_001",
                "name": "Developer",
                "role": "Software Developer",
                "goal": "Write quality code",
                "description": "Experienced developer",
                "llm": {"provider": "openrouter"}
            },
            {
                "agent_id": "agt_worker_002",
                "name": "Tester",
                "role": "QA Tester",
                "goal": "Ensure quality",
                "description": "Quality assurance specialist",
                "llm": {"provider": "openrouter"}
            }
        ]

    # ========================================================================
    # TEST: Initialization
    # ========================================================================

    def test_adapter_initialization(self, adapter):
        """Test that adapter initializes correctly."""
        assert isinstance(adapter, CrewAIAdapter)
        assert hasattr(adapter, '_llm_cache')
        assert isinstance(adapter._llm_cache, dict)
        assert len(adapter._llm_cache) == 0

    # ========================================================================
    # TEST: LLM Configuration
    # ========================================================================

    @patch('apps.workflow.crewai_adapter.ChatOpenAI')
    @patch.dict('os.environ', {
        'OPENROUTER_API_KEY': 'test_key_123',
        'OPENROUTER_BASE_URL': 'https://openrouter.ai/api/v1'
    })
    def test_get_llm_for_agent_openrouter(self, mock_chat_openai, adapter, sample_agent_config):
        """Test LLM creation for OpenRouter provider."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        llm = adapter._get_llm_for_agent(sample_agent_config)

        # Verify LLM was created with correct parameters
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs['base_url'] == 'https://openrouter.ai/api/v1'
        assert call_kwargs['api_key'] == 'test_key_123'
        assert call_kwargs['model'] == 'nvidia/nemotron-3-nano-30b-a3b:free'
        assert call_kwargs['temperature'] == 0.7

        # Verify LLM was cached
        assert llm == mock_llm
        assert len(adapter._llm_cache) == 1

    @patch('apps.workflow.crewai_adapter.ChatOpenAI')
    @patch.dict('os.environ', {
        'OPENROUTER_API_KEY': 'test_key_123'
    })
    def test_llm_caching(self, mock_chat_openai, adapter, sample_agent_config):
        """Test that LLM instances are cached and reused."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        # Get LLM twice with same config
        llm1 = adapter._get_llm_for_agent(sample_agent_config)
        llm2 = adapter._get_llm_for_agent(sample_agent_config)

        # Should only create once (cached)
        assert mock_chat_openai.call_count == 1
        assert llm1 is llm2

    @patch.dict('os.environ', {}, clear=True)
    def test_get_llm_missing_api_key(self, adapter, sample_agent_config):
        """Test that missing API key raises appropriate error."""
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
            adapter._get_llm_for_agent(sample_agent_config)

    # ========================================================================
    # TEST: Sequential Agent Node Creation
    # ========================================================================

    @patch('apps.workflow.crewai_adapter.Crew')
    @patch('apps.workflow.crewai_adapter.Agent')
    @patch('apps.workflow.crewai_adapter.Task')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'})
    def test_create_sequential_agent_node(
        self,
        mock_task_class,
        mock_agent_class,
        mock_crew_class,
        adapter,
        sample_agent_config,
        sample_state
    ):
        """Test creation of sequential agent node function."""
        # Setup mocks
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        mock_task = Mock()
        mock_task_class.return_value = mock_task

        mock_crew = Mock()
        mock_result = Mock()
        mock_result.raw = "Test output result"
        mock_crew.kickoff.return_value = mock_result
        mock_crew_class.return_value = mock_crew

        # Create node function
        node_func = adapter.create_sequential_agent_node(sample_agent_config)

        # Verify node function is callable
        assert callable(node_func)

        # Execute node function
        with patch.object(adapter, '_get_llm_for_agent', return_value=Mock()):
            result_state = node_func(sample_state)

        # Verify state was updated correctly
        assert isinstance(result_state, dict)
        assert "output" in result_state
        assert result_state["output"] == "Test output result"
        assert len(result_state["messages"]) == 1
        assert result_state["messages"][0]["role"] == "Test Executor"

        # Verify CrewAI components were created correctly
        mock_agent_class.assert_called_once()
        mock_task_class.assert_called_once()
        mock_crew_class.assert_called_once()
        mock_crew.kickoff.assert_called_once()

    # ========================================================================
    # TEST: Hierarchical Crew Node Creation
    # ========================================================================

    @patch('apps.workflow.crewai_adapter.Crew')
    @patch('apps.workflow.crewai_adapter.Agent')
    @patch('apps.workflow.crewai_adapter.Task')
    @patch('apps.workflow.crewai_adapter.Process')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'})
    def test_create_hierarchical_crew_node(
        self,
        mock_process,
        mock_task_class,
        mock_agent_class,
        mock_crew_class,
        adapter,
        master_agent_config,
        worker_agent_configs,
        sample_state
    ):
        """Test creation of hierarchical crew node with manager and workers."""
        # Setup mocks
        mock_manager = Mock()
        mock_worker1 = Mock()
        mock_worker2 = Mock()
        mock_agent_class.side_effect = [mock_manager, mock_worker1, mock_worker2]

        mock_task = Mock()
        mock_task_class.return_value = mock_task

        mock_crew = Mock()
        mock_result = Mock()
        mock_result.output = "Hierarchical execution completed"
        mock_crew.kickoff.return_value = mock_result
        mock_crew_class.return_value = mock_crew

        # Create hierarchical node function
        node_func = adapter.create_hierarchical_crew_node(
            master_agent_config=master_agent_config,
            sub_agent_configs=worker_agent_configs,
            delegation_strategy="dynamic"
        )

        # Verify node function is callable
        assert callable(node_func)

        # Execute node function
        with patch.object(adapter, '_get_llm_for_agent', return_value=Mock()):
            result_state = node_func(sample_state)

        # Verify state was updated
        assert isinstance(result_state, dict)
        assert "hierarchical_output" in result_state
        assert result_state["hierarchical_output"] == "Hierarchical execution completed"

        # Verify manager was created with delegation enabled
        manager_call = mock_agent_class.call_args_list[0]
        assert manager_call[1]['allow_delegation'] is True
        assert manager_call[1]['role'] == "Manager"

        # Verify workers were created without delegation
        worker1_call = mock_agent_class.call_args_list[1]
        worker2_call = mock_agent_class.call_args_list[2]
        assert worker1_call[1]['allow_delegation'] is False
        assert worker2_call[1]['allow_delegation'] is False

        # Verify crew was created with hierarchical process
        mock_crew_class.assert_called_once()
        crew_call_kwargs = mock_crew_class.call_args[1]
        assert len(crew_call_kwargs['agents']) == 3  # 1 manager + 2 workers

    # ========================================================================
    # TEST: Parallel Crew Node Creation
    # ========================================================================

    @patch('apps.workflow.crewai_adapter.Crew')
    @patch('apps.workflow.crewai_adapter.Agent')
    @patch('apps.workflow.crewai_adapter.Task')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'})
    def test_create_parallel_crew_node(
        self,
        mock_task_class,
        mock_agent_class,
        mock_crew_class,
        adapter,
        worker_agent_configs,
        sample_state
    ):
        """Test creation of parallel crew node for concurrent execution."""
        # Setup mocks
        mock_agents = [Mock(), Mock()]
        mock_agent_class.side_effect = mock_agents

        mock_tasks = [Mock(), Mock()]
        mock_tasks[0].output = "Output 1"
        mock_tasks[1].output = "Output 2"
        mock_task_class.side_effect = mock_tasks

        mock_crew = Mock()
        mock_crew.tasks = mock_tasks
        mock_result = Mock()
        mock_result.output = "Parallel execution completed"
        mock_crew.kickoff.return_value = mock_result
        mock_crew_class.return_value = mock_crew

        # Create parallel node function
        node_func = adapter.create_parallel_crew_node(
            agent_configs=worker_agent_configs,
            aggregation_strategy="combine"
        )

        # Verify node function is callable
        assert callable(node_func)

        # Execute node function
        with patch.object(adapter, '_get_llm_for_agent', return_value=Mock()):
            with patch.object(adapter, '_combine_results', return_value="Combined output"):
                result_state = node_func(sample_state)

        # Verify state was updated
        assert isinstance(result_state, dict)
        assert "parallel_output" in result_state
        assert "individual_outputs" in result_state
        assert len(result_state["individual_outputs"]) == 2

        # Verify multiple agents were created
        assert mock_agent_class.call_count == 2

    # ========================================================================
    # TEST: Result Aggregation Strategies
    # ========================================================================

    def test_combine_results(self, adapter):
        """Test combining multiple results."""
        results = ["Result 1", "Result 2", "Result 3"]
        combined = adapter._combine_results(results)

        assert "Result 1" in combined
        assert "Result 2" in combined
        assert "Result 3" in combined
        assert "---" in combined

    def test_combine_results_empty(self, adapter):
        """Test combining empty results."""
        results = []
        combined = adapter._combine_results(results)
        assert combined == ""

    def test_vote_on_results(self, adapter):
        """Test voting on results to select most common."""
        results = ["Option A", "Option B", "Option A", "Option A", "Option B"]
        winner = adapter._vote_on_results(results)
        assert winner == "Option A"

    # ========================================================================
    # TEST: Architectural Validation
    # ========================================================================

    def test_validate_no_orchestration_in_crewai_valid(self):
        """Test that valid CrewAI config passes validation."""
        valid_config = {
            "agents": ["agent1", "agent2"],
            "tasks": ["task1"],
            "process": "hierarchical"
        }

        # Should not raise
        assert CrewAIAdapter.validate_no_orchestration_in_crewai(valid_config) is True

    def test_validate_no_orchestration_in_crewai_invalid(self):
        """Test that config with orchestration logic fails validation."""
        invalid_configs = [
            {"next_node": "agent2"},
            {"graph.add_edge": "edge1"},
            {"workflow_control": True},
            {"decide_next": "logic"},
            {"routing_logic": "complex"}
        ]

        for config in invalid_configs:
            with pytest.raises(ValueError, match="orchestration logic"):
                CrewAIAdapter.validate_no_orchestration_in_crewai(config)

    # ========================================================================
    # TEST: Error Handling
    # ========================================================================

    @patch('apps.workflow.crewai_adapter.Crew')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'})
    def test_sequential_node_crew_failure(
        self,
        mock_crew_class,
        adapter,
        sample_agent_config,
        sample_state
    ):
        """Test that crew execution failure raises appropriate error."""
        # Setup mock to raise error
        mock_crew = Mock()
        mock_crew.kickoff.side_effect = Exception("CrewAI execution failed")
        mock_crew_class.return_value = mock_crew

        # Create node function
        node_func = adapter.create_sequential_agent_node(sample_agent_config)

        # Execute should raise RuntimeError
        with patch.object(adapter, '_get_llm_for_agent', return_value=Mock()):
            with pytest.raises(RuntimeError, match="CrewAI sequential agent node failed"):
                node_func(sample_state)


class TestCrewAIMergeNode:
    """Test suite for merge node utility function."""

    @pytest.fixture
    def parallel_agent_configs(self):
        """Sample parallel agent configurations."""
        return [
            {
                "agent_id": "agt_parallel_001",
                "name": "Analyzer 1",
                "role": "Data Analyzer",
                "output_schema": ["analysis_1"]
            },
            {
                "agent_id": "agt_parallel_002",
                "name": "Analyzer 2",
                "role": "Data Analyzer",
                "output_schema": ["analysis_2"]
            }
        ]

    @pytest.fixture
    def state_with_parallel_outputs(self):
        """State after parallel agents have executed."""
        return {
            "analysis_1": "First analysis result",
            "analysis_2": "Second analysis result",
            "messages": [
                {"agent": "agt_parallel_001", "output": "First analysis result"},
                {"agent": "agt_parallel_002", "output": "Second analysis result"}
            ]
        }

    def test_create_merge_node(self, parallel_agent_configs):
        """Test creation of merge node function."""
        merge_node = create_crewai_merge_node(
            parallel_agent_configs=parallel_agent_configs,
            merge_strategy="combine"
        )

        # Verify merge node is callable
        assert callable(merge_node)

    def test_merge_node_execution(self, parallel_agent_configs, state_with_parallel_outputs):
        """Test execution of merge node with state."""
        merge_node = create_crewai_merge_node(
            parallel_agent_configs=parallel_agent_configs,
            merge_strategy="combine"
        )

        # Execute merge node
        result_state = merge_node(state_with_parallel_outputs)

        # Verify state is returned (merge node aggregates)
        assert isinstance(result_state, dict)
        assert "aggregated_output" in result_state


# ============================================================================
# Integration-level Tests (CrewAI + LangGraph State Flow)
# ============================================================================

class TestStateFlow:
    """Test that state flows correctly between LangGraph and CrewAI."""

    @pytest.fixture
    def adapter(self):
        return CrewAIAdapter()

    @patch('apps.workflow.crewai_adapter.Crew')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'})
    def test_state_preservation(self, mock_crew_class, adapter):
        """Test that existing state keys are preserved through node execution."""
        # Setup
        agent_config = {
            "agent_id": "agt_001",
            "role": "Processor",
            "goal": "Process data",
            "input_schema": ["input"],
            "output_schema": ["output"]
        }

        initial_state = {
            "input": "test input",
            "existing_key": "existing value",
            "messages": []
        }

        mock_crew = Mock()
        mock_result = Mock()
        mock_result.raw = "processed output"
        mock_crew.kickoff.return_value = mock_result
        mock_crew_class.return_value = mock_crew

        # Execute
        node_func = adapter.create_sequential_agent_node(agent_config)

        with patch.object(adapter, '_get_llm_for_agent', return_value=Mock()):
            result_state = node_func(initial_state)

        # Verify existing keys preserved
        assert result_state["existing_key"] == "existing value"
        assert result_state["input"] == "test input"

        # Verify new output added
        assert "output" in result_state
        assert result_state["output"] == "processed output"

    def test_messages_appending(self, adapter):
        """Test that messages are appended, not overwritten."""
        agent_config = {
            "agent_id": "agt_002",
            "role": "Agent",
            "goal": "Test",
            "output_schema": ["output"]
        }

        state_with_messages = {
            "messages": [
                {"agent": "previous", "output": "previous output"}
            ]
        }

        # Mock execution
        with patch('apps.workflow.crewai_adapter.Crew') as mock_crew_class:
            mock_crew = Mock()
            mock_result = Mock()
            mock_result.raw = "new output"
            mock_crew.kickoff.return_value = mock_result
            mock_crew_class.return_value = mock_crew

            node_func = adapter.create_sequential_agent_node(agent_config)

            with patch.object(adapter, '_get_llm_for_agent', return_value=Mock()):
                with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
                    result_state = node_func(state_with_messages)

        # Verify messages were appended, not replaced
        assert len(result_state["messages"]) == 2
        assert result_state["messages"][0]["agent"] == "previous"
        assert result_state["messages"][1]["agent"] == "agt_002"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
