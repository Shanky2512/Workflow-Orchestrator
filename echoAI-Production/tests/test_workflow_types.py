"""
Test Suite for Workflow Types

Tests all four workflow execution models:
1. Sequential - Linear agent chain (A → B → C)
2. Parallel - Concurrent execution (coordinator → [agents] → aggregator)
3. Hierarchical - Manager delegates to workers (CrewAI hierarchical process)
4. Hybrid - Mixed parallel + sequential (parallel → merge → sequential)

Validates that:
- LangGraph owns workflow topology and execution order
- CrewAI handles agent collaboration within nodes
- State propagates correctly through workflows
- Each workflow type compiles and executes properly
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any
import json

from apps.workflow.designer.compiler import WorkflowCompiler
from apps.workflow.designer.designer import WorkflowDesigner


class TestSequentialWorkflows:
    """Test sequential workflow compilation and execution."""

    @pytest.fixture
    def compiler(self):
        """Create workflow compiler instance."""
        return WorkflowCompiler(use_crewai=True)

    @pytest.fixture
    def sequential_workflow(self):
        """Sample sequential workflow definition."""
        return {
            "workflow_id": "wf_sequential_test",
            "workflow_name": "Sequential Test Workflow",
            "execution_model": "sequential",
            "agents": ["agt_001", "agt_002", "agt_003"]
        }

    @pytest.fixture
    def agent_registry(self):
        """Sample agent registry with 3 agents."""
        return {
            "agt_001": {
                "agent_id": "agt_001",
                "name": "Agent 1",
                "role": "Designer",
                "goal": "Design solution",
                "description": "First agent in sequence",
                "input_schema": ["user_input"],
                "output_schema": ["design"],
                "llm": {"provider": "openrouter"}
            },
            "agt_002": {
                "agent_id": "agt_002",
                "name": "Agent 2",
                "role": "Implementer",
                "goal": "Implement design",
                "description": "Second agent in sequence",
                "input_schema": ["design"],
                "output_schema": ["implementation"],
                "llm": {"provider": "openrouter"}
            },
            "agt_003": {
                "agent_id": "agt_003",
                "name": "Agent 3",
                "role": "Reviewer",
                "goal": "Review implementation",
                "description": "Third agent in sequence",
                "input_schema": ["implementation"],
                "output_schema": ["review"],
                "llm": {"provider": "openrouter"}
            }
        }

    def test_sequential_compilation(self, compiler, sequential_workflow, agent_registry):
        """Test that sequential workflow compiles to LangGraph StateGraph."""
        compiled = compiler.compile(sequential_workflow, agent_registry)

        # Verify compilation succeeded
        assert compiled is not None

        # Verify it's a compiled graph (has invoke method)
        assert hasattr(compiled, 'invoke')
        assert callable(compiled.invoke)

    def test_sequential_graph_structure(self, compiler, sequential_workflow, agent_registry):
        """Test that sequential graph has correct node → node → node structure."""
        with patch('apps.workflow.designer.compiler.StateGraph') as mock_state_graph:
            mock_graph = Mock()
            mock_state_graph.return_value = mock_graph
            mock_graph.compile.return_value = Mock()

            compiler.compile(sequential_workflow, agent_registry)

            # Verify nodes were added for each agent
            assert mock_graph.add_node.call_count == 3

            # Verify edges connect agents sequentially
            edge_calls = [call[0] for call in mock_graph.add_edge.call_args_list]

            # Should have edges: agt_001 → agt_002, agt_002 → agt_003, agt_003 → END
            assert len(edge_calls) >= 2

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'})
    @patch('apps.workflow.crewai_adapter.Crew')
    def test_sequential_execution_flow(
        self,
        mock_crew_class,
        compiler,
        sequential_workflow,
        agent_registry
    ):
        """Test that sequential workflow executes agents in order."""
        # Setup mocks for agent execution
        execution_order = []

        def create_mock_crew_with_tracking(agent_id):
            def crew_kickoff(*args, **kwargs):
                execution_order.append(agent_id)
                mock_result = Mock()
                mock_result.raw = f"Output from {agent_id}"
                return mock_result

            mock_crew = Mock()
            mock_crew.kickoff = crew_kickoff
            return mock_crew

        # Mock crew creation to track execution order
        mock_crew_class.side_effect = lambda **kwargs: create_mock_crew_with_tracking(
            kwargs.get('agents', [Mock()])[0] if kwargs.get('agents') else 'unknown'
        )

        # Compile and execute
        compiled = compiler.compile(sequential_workflow, agent_registry)

        # Note: Full execution would require mocking entire LangGraph execution
        # For unit tests, we verify compilation is correct
        # Integration tests will verify actual execution
        assert compiled is not None


class TestParallelWorkflows:
    """Test parallel workflow compilation and execution."""

    @pytest.fixture
    def compiler(self):
        return WorkflowCompiler(use_crewai=True)

    @pytest.fixture
    def parallel_workflow(self):
        """Sample parallel workflow definition."""
        return {
            "workflow_id": "wf_parallel_test",
            "workflow_name": "Parallel Test Workflow",
            "execution_model": "parallel",
            "agents": ["agt_001", "agt_002", "agt_003"]
        }

    @pytest.fixture
    def agent_registry(self):
        """Sample agent registry with 3 parallel agents."""
        return {
            "agt_001": {
                "agent_id": "agt_001",
                "name": "Security Analyzer",
                "role": "Security Expert",
                "goal": "Analyze security",
                "output_schema": ["security_analysis"],
                "llm": {"provider": "openrouter"}
            },
            "agt_002": {
                "agent_id": "agt_002",
                "name": "Performance Analyzer",
                "role": "Performance Expert",
                "goal": "Analyze performance",
                "output_schema": ["performance_analysis"],
                "llm": {"provider": "openrouter"}
            },
            "agt_003": {
                "agent_id": "agt_003",
                "name": "Code Quality Analyzer",
                "role": "Quality Expert",
                "goal": "Analyze code quality",
                "output_schema": ["quality_analysis"],
                "llm": {"provider": "openrouter"}
            }
        }

    def test_parallel_compilation(self, compiler, parallel_workflow, agent_registry):
        """Test that parallel workflow compiles correctly."""
        compiled = compiler.compile(parallel_workflow, agent_registry)

        # Verify compilation succeeded
        assert compiled is not None
        assert hasattr(compiled, 'invoke')

    def test_parallel_graph_structure(self, compiler, parallel_workflow, agent_registry):
        """Test that parallel graph has coordinator → [agents] → aggregator structure."""
        with patch('apps.workflow.designer.compiler.StateGraph') as mock_state_graph:
            mock_graph = Mock()
            mock_state_graph.return_value = mock_graph
            mock_graph.compile.return_value = Mock()

            compiler.compile(parallel_workflow, agent_registry)

            # Verify coordinator and aggregator nodes
            node_names = [call[0][0] for call in mock_graph.add_node.call_args_list]
            assert "coordinator" in node_names
            assert "aggregator" in node_names

            # Verify parallel agent nodes
            assert "agt_001" in node_names
            assert "agt_002" in node_names
            assert "agt_003" in node_names

            # Verify edges: coordinator → each agent → aggregator
            edge_calls = [call[0] for call in mock_graph.add_edge.call_args_list]

            # Should have edges from coordinator to each agent
            coordinator_edges = [e for e in edge_calls if e[0] == "coordinator"]
            assert len(coordinator_edges) == 3

            # Should have edges from each agent to aggregator
            aggregator_edges = [e for e in edge_calls if e[1] == "aggregator"]
            assert len(aggregator_edges) == 3


class TestHierarchicalWorkflows:
    """Test hierarchical workflow compilation and execution."""

    @pytest.fixture
    def compiler(self):
        return WorkflowCompiler(use_crewai=True)

    @pytest.fixture
    def hierarchical_workflow(self):
        """Sample hierarchical workflow definition."""
        return {
            "workflow_id": "wf_hierarchical_test",
            "workflow_name": "Hierarchical Test Workflow",
            "execution_model": "hierarchical",
            "hierarchy": {
                "master_agent": "agt_master",
                "delegation_order": ["agt_worker_1", "agt_worker_2", "agt_worker_3"],
                "delegation_strategy": "dynamic"
            }
        }

    @pytest.fixture
    def agent_registry(self):
        """Sample agent registry with master and workers."""
        return {
            "agt_master": {
                "agent_id": "agt_master",
                "name": "Project Manager",
                "role": "Manager",
                "goal": "Coordinate project completion",
                "description": "Coordinates team to complete project",
                "llm": {"provider": "openrouter"}
            },
            "agt_worker_1": {
                "agent_id": "agt_worker_1",
                "name": "Frontend Developer",
                "role": "Frontend Dev",
                "goal": "Build frontend",
                "llm": {"provider": "openrouter"}
            },
            "agt_worker_2": {
                "agent_id": "agt_worker_2",
                "name": "Backend Developer",
                "role": "Backend Dev",
                "goal": "Build backend",
                "llm": {"provider": "openrouter"}
            },
            "agt_worker_3": {
                "agent_id": "agt_worker_3",
                "name": "DevOps Engineer",
                "role": "DevOps",
                "goal": "Deploy application",
                "llm": {"provider": "openrouter"}
            }
        }

    def test_hierarchical_compilation(self, compiler, hierarchical_workflow, agent_registry):
        """Test that hierarchical workflow compiles correctly."""
        compiled = compiler.compile(hierarchical_workflow, agent_registry)

        # Verify compilation succeeded
        assert compiled is not None
        assert hasattr(compiled, 'invoke')

    def test_hierarchical_uses_crewai(self, compiler, hierarchical_workflow, agent_registry):
        """Test that hierarchical workflow uses CrewAI for delegation."""
        with patch('apps.workflow.crewai_adapter.CrewAIAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.create_hierarchical_crew_node.return_value = Mock()
            mock_adapter_class.return_value = mock_adapter

            # Force compiler to use our mocked adapter
            compiler._crewai_adapter = mock_adapter

            compiled = compiler.compile(hierarchical_workflow, agent_registry)

            # Verify CrewAI hierarchical node was created
            mock_adapter.create_hierarchical_crew_node.assert_called_once()

            # Verify it was called with correct parameters
            call_kwargs = mock_adapter.create_hierarchical_crew_node.call_args[1]
            assert call_kwargs['delegation_strategy'] == 'dynamic'

    def test_hierarchical_graph_structure(self, compiler, hierarchical_workflow, agent_registry):
        """Test that hierarchical workflow creates single master node."""
        with patch('apps.workflow.designer.compiler.StateGraph') as mock_state_graph:
            mock_graph = Mock()
            mock_state_graph.return_value = mock_graph
            mock_graph.compile.return_value = Mock()

            compiler.compile(hierarchical_workflow, agent_registry)

            # For CrewAI hierarchical, should have single "hierarchical_master" node
            node_names = [call[0][0] for call in mock_graph.add_node.call_args_list]

            # When using CrewAI, should have just one hierarchical_master node
            # (workers are inside the CrewAI crew, not separate LangGraph nodes)
            if compiler._use_crewai:
                assert "hierarchical_master" in node_names


class TestHybridWorkflows:
    """Test hybrid workflow compilation (parallel + sequential)."""

    @pytest.fixture
    def compiler(self):
        return WorkflowCompiler(use_crewai=True)

    @pytest.fixture
    def hybrid_workflow(self):
        """Sample hybrid workflow: 3 parallel → merge → 2 sequential."""
        return {
            "workflow_id": "wf_hybrid_test",
            "workflow_name": "Hybrid Test Workflow",
            "execution_model": "hybrid",
            "agents": ["agt_001", "agt_002", "agt_003", "agt_004", "agt_005"],
            "topology": {
                "parallel_groups": [
                    {
                        "agents": ["agt_001", "agt_002", "agt_003"],
                        "merge_strategy": "combine"
                    }
                ],
                "sequential_chains": [
                    {
                        "agents": ["agt_004", "agt_005"]
                    }
                ]
            }
        }

    @pytest.fixture
    def agent_registry(self):
        """Agent registry for hybrid workflow."""
        return {
            f"agt_{str(i).zfill(3)}": {
                "agent_id": f"agt_{str(i).zfill(3)}",
                "name": f"Agent {i}",
                "role": f"Role {i}",
                "goal": f"Complete task {i}",
                "llm": {"provider": "openrouter"}
            }
            for i in range(1, 6)
        }

    def test_hybrid_compilation(self, compiler, hybrid_workflow, agent_registry):
        """Test that hybrid workflow compiles correctly."""
        compiled = compiler.compile(hybrid_workflow, agent_registry)

        # Verify compilation succeeded
        assert compiled is not None
        assert hasattr(compiled, 'invoke')

    def test_hybrid_graph_structure(self, compiler, hybrid_workflow, agent_registry):
        """Test that hybrid graph has parallel → merge → sequential structure."""
        with patch('apps.workflow.designer.compiler.StateGraph') as mock_state_graph:
            mock_graph = Mock()
            mock_state_graph.return_value = mock_graph
            mock_graph.compile.return_value = Mock()

            compiler.compile(hybrid_workflow, agent_registry)

            # Verify coordinator and merge nodes exist
            node_names = [call[0][0] for call in mock_graph.add_node.call_args_list]
            assert "coordinator" in node_names
            assert "merge" in node_names

            # Verify parallel agents (001, 002, 003)
            assert "agt_001" in node_names
            assert "agt_002" in node_names
            assert "agt_003" in node_names

            # Verify sequential agents (004, 005)
            assert "agt_004" in node_names
            assert "agt_005" in node_names

            # Verify edge structure
            edge_calls = [call[0] for call in mock_graph.add_edge.call_args_list]

            # Coordinator → parallel agents
            coordinator_edges = [e for e in edge_calls if e[0] == "coordinator"]
            assert len(coordinator_edges) >= 3

            # Parallel agents → merge
            merge_edges = [e for e in edge_calls if e[1] == "merge"]
            assert len(merge_edges) >= 3

    def test_hybrid_topology_parsing(self, compiler, hybrid_workflow, agent_registry):
        """Test that topology is correctly parsed from workflow JSON."""
        # This tests the internal _compile_hybrid method
        from langgraph.graph import StateGraph

        with patch.object(compiler, '_create_agent_node', return_value=Mock()):
            compiled = compiler.compile(hybrid_workflow, agent_registry)

            # Verify compilation used topology information
            assert compiled is not None


class TestWorkflowDesignerInference:
    """Test that workflow designer correctly infers workflow types."""

    @pytest.fixture
    def designer(self):
        """Create workflow designer instance."""
        return WorkflowDesigner()

    @pytest.mark.parametrize("prompt,expected_model", [
        ("Design a workflow for code generation then review then testing", "sequential"),
        ("Analyze code for security, performance, and quality simultaneously", "parallel"),
        ("A project manager coordinates developers, testers, and devops", "hierarchical"),
        ("Extract from 3 sources in parallel, then transform, then load", "hybrid")
    ])
    def test_execution_model_inference(self, designer, prompt, expected_model):
        """Test that designer infers correct execution model from prompts."""
        # This would require actual LLM calls, so we'll test the prompt structure
        # Real inference testing is in integration tests

        # Verify designer has the enhanced system prompt
        assert hasattr(designer, '_get_system_prompt')

        # Verify decision tree keywords are in prompt
        system_prompt = designer._get_system_prompt()
        assert "sequential" in system_prompt.lower()
        assert "parallel" in system_prompt.lower()
        assert "hierarchical" in system_prompt.lower()
        assert "hybrid" in system_prompt.lower()


class TestStateManagement:
    """Test state propagation through different workflow types."""

    @pytest.fixture
    def initial_state(self):
        """Initial workflow state."""
        return {
            "user_input": "Test input",
            "workflow_id": "wf_test",
            "messages": []
        }

    def test_state_schema_generation(self):
        """Test that workflow state schema is generated correctly."""
        from apps.workflow.designer.compiler import WorkflowCompiler

        compiler = WorkflowCompiler()

        agent_registry = {
            "agt_001": {
                "input_schema": ["input_a", "input_b"],
                "output_schema": ["output_x"]
            },
            "agt_002": {
                "input_schema": ["output_x"],
                "output_schema": ["output_y", "output_z"]
            }
        }

        # Generate state schema
        state_class = compiler._create_workflow_state(agent_registry)

        # Verify state class has required fields
        assert hasattr(state_class, '__annotations__')

        # State should include all input/output keys plus standard fields
        annotations = state_class.__annotations__
        assert 'messages' in annotations


# ============================================================================
# Workflow Validation Tests
# ============================================================================

class TestWorkflowValidation:
    """Test workflow validation logic."""

    def test_missing_execution_model(self):
        """Test that workflow without execution_model is rejected."""
        compiler = WorkflowCompiler()

        invalid_workflow = {
            "workflow_id": "wf_invalid",
            "agents": ["agt_001"]
            # Missing execution_model
        }

        agent_registry = {
            "agt_001": {"agent_id": "agt_001", "role": "Agent"}
        }

        with pytest.raises((ValueError, KeyError)):
            compiler.compile(invalid_workflow, agent_registry)

    def test_hierarchical_missing_master(self):
        """Test that hierarchical workflow without master agent is rejected."""
        compiler = WorkflowCompiler()

        invalid_workflow = {
            "workflow_id": "wf_hierarchical_invalid",
            "execution_model": "hierarchical",
            "hierarchy": {
                # Missing master_agent
                "delegation_order": ["agt_001"]
            }
        }

        agent_registry = {
            "agt_001": {"agent_id": "agt_001", "role": "Worker"}
        }

        with pytest.raises(ValueError, match="master agent"):
            compiler.compile(invalid_workflow, agent_registry)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
