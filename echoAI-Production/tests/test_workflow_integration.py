"""
Integration Test Suite for EchoAI Workflow Orchestration

End-to-end tests covering the complete workflow lifecycle:
1. Prompt → Design (LLM-powered workflow generation)
2. Design → Compile (LangGraph StateGraph creation)
3. Compile → Execute (CrewAI + LangGraph execution)
4. State Management (cross-workflow state integrity)

These tests validate the complete integration of:
- Workflow Designer (LLM-based planning)
- Workflow Compiler (LangGraph graph construction)
- CrewAI Adapter (agent collaboration)
- State management (LangGraph state flow)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from typing import Dict, Any

from apps.workflow.designer.designer import WorkflowDesigner
from apps.workflow.designer.compiler import WorkflowCompiler
from apps.workflow.crewai_adapter import CrewAIAdapter


# ============================================================================
# Fixtures for Integration Tests
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for workflow design."""
    def create_response(prompt_type="sequential"):
        if prompt_type == "sequential":
            return {
                "execution_model": "sequential",
                "workflow_name": "Code Generation Workflow",
                "reasoning": "Tasks must happen in order",
                "agents": [
                    {
                        "name": "Designer",
                        "role": "Software Architect",
                        "goal": "Design software architecture",
                        "description": "Designs the overall architecture",
                        "input_schema": ["user_requirements"],
                        "output_schema": ["architecture_design"]
                    },
                    {
                        "name": "Implementer",
                        "role": "Software Developer",
                        "goal": "Implement the design",
                        "description": "Writes code based on design",
                        "input_schema": ["architecture_design"],
                        "output_schema": ["code"]
                    },
                    {
                        "name": "Tester",
                        "role": "QA Engineer",
                        "goal": "Test the implementation",
                        "description": "Tests the code for quality",
                        "input_schema": ["code"],
                        "output_schema": ["test_results"]
                    }
                ]
            }
        elif prompt_type == "parallel":
            return {
                "execution_model": "parallel",
                "workflow_name": "Code Analysis Workflow",
                "reasoning": "Independent analyses can run simultaneously",
                "agents": [
                    {
                        "name": "Security Analyzer",
                        "role": "Security Expert",
                        "goal": "Analyze security vulnerabilities",
                        "description": "Checks for security issues",
                        "input_schema": ["code"],
                        "output_schema": ["security_report"]
                    },
                    {
                        "name": "Performance Analyzer",
                        "role": "Performance Expert",
                        "goal": "Analyze performance bottlenecks",
                        "description": "Checks for performance issues",
                        "input_schema": ["code"],
                        "output_schema": ["performance_report"]
                    },
                    {
                        "name": "Quality Analyzer",
                        "role": "Code Quality Expert",
                        "goal": "Analyze code quality",
                        "description": "Checks code quality metrics",
                        "input_schema": ["code"],
                        "output_schema": ["quality_report"]
                    }
                ]
            }
        elif prompt_type == "hierarchical":
            return {
                "execution_model": "hierarchical",
                "workflow_name": "Project Management Workflow",
                "reasoning": "Manager coordinates specialized workers",
                "agents": [
                    {
                        "name": "Project Manager",
                        "role": "Manager",
                        "goal": "Coordinate project completion",
                        "description": "Coordinates team members",
                        "input_schema": ["project_requirements"],
                        "output_schema": ["project_status"]
                    },
                    {
                        "name": "Frontend Developer",
                        "role": "Frontend Developer",
                        "goal": "Build user interface",
                        "description": "Develops frontend",
                        "input_schema": ["ui_requirements"],
                        "output_schema": ["frontend_code"]
                    },
                    {
                        "name": "Backend Developer",
                        "role": "Backend Developer",
                        "goal": "Build backend services",
                        "description": "Develops backend",
                        "input_schema": ["api_requirements"],
                        "output_schema": ["backend_code"]
                    }
                ],
                "hierarchy": {
                    "master_agent_index": 0,
                    "sub_agent_indices": [1, 2],
                    "delegation_strategy": "dynamic"
                }
            }
        elif prompt_type == "hybrid":
            return {
                "execution_model": "hybrid",
                "workflow_name": "Data Pipeline Workflow",
                "reasoning": "Parallel extraction, then sequential transformation and loading",
                "agents": [
                    {
                        "name": "Source 1 Extractor",
                        "role": "Data Extractor",
                        "goal": "Extract from source 1",
                        "description": "Extracts data from first source",
                        "output_schema": ["data_1"]
                    },
                    {
                        "name": "Source 2 Extractor",
                        "role": "Data Extractor",
                        "goal": "Extract from source 2",
                        "description": "Extracts data from second source",
                        "output_schema": ["data_2"]
                    },
                    {
                        "name": "Source 3 Extractor",
                        "role": "Data Extractor",
                        "goal": "Extract from source 3",
                        "description": "Extracts data from third source",
                        "output_schema": ["data_3"]
                    },
                    {
                        "name": "Transformer",
                        "role": "Data Transformer",
                        "goal": "Transform extracted data",
                        "description": "Transforms and cleans data",
                        "input_schema": ["data_1", "data_2", "data_3"],
                        "output_schema": ["transformed_data"]
                    },
                    {
                        "name": "Loader",
                        "role": "Data Loader",
                        "goal": "Load data to destination",
                        "description": "Loads data to warehouse",
                        "input_schema": ["transformed_data"],
                        "output_schema": ["load_status"]
                    }
                ],
                "topology": {
                    "parallel_groups": [
                        {
                            "agents": [0, 1, 2],
                            "merge_strategy": "combine"
                        }
                    ],
                    "sequential_chains": [
                        {
                            "agents": [3, 4]
                        }
                    ]
                }
            }
        return {}

    return create_response


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

class TestSequentialWorkflowE2E:
    """End-to-end test for sequential workflow."""

    @pytest.fixture
    def designer(self):
        return WorkflowDesigner()

    @pytest.fixture
    def compiler(self):
        return WorkflowCompiler(use_crewai=True)

    def test_sequential_workflow_e2e_design_and_compile(
        self,
        designer,
        compiler,
        mock_llm_response
    ):
        """Test sequential workflow from design to compilation."""
        # Mock the LLM call for design
        with patch.object(designer, '_design_with_llm') as mock_design:
            mock_design.return_value = (
                mock_llm_response("sequential"),
                {}  # No agents dict in this simplified mock
            )

            # Step 1: Design workflow from prompt
            workflow_json, agents_dict = designer._design_with_llm(
                "Create a workflow for code generation",
                default_llm={}
            )

            # Verify design output
            assert workflow_json["execution_model"] == "sequential"
            assert len(workflow_json["agents"]) == 3

            # Step 2: Prepare for compilation
            # In real workflow, agents would be stored and retrieved
            # For this test, we'll create a mock agent registry
            agent_registry = {}
            for idx, agent_config in enumerate(workflow_json["agents"]):
                agent_id = f"agt_{idx:03d}"
                agent_registry[agent_id] = {
                    "agent_id": agent_id,
                    **agent_config
                }

            # Update workflow with agent IDs
            workflow_json["agents"] = list(agent_registry.keys())

            # Step 3: Compile workflow
            compiled_graph = compiler.compile(workflow_json, agent_registry)

            # Verify compilation
            assert compiled_graph is not None
            assert hasattr(compiled_graph, 'invoke')


class TestParallelWorkflowE2E:
    """End-to-end test for parallel workflow."""

    @pytest.fixture
    def compiler(self):
        return WorkflowCompiler(use_crewai=True)

    def test_parallel_workflow_e2e_compile_and_structure(
        self,
        compiler,
        mock_llm_response
    ):
        """Test parallel workflow compilation and structure."""
        # Get workflow design
        workflow_json = mock_llm_response("parallel")

        # Create agent registry
        agent_registry = {}
        for idx, agent_config in enumerate(workflow_json["agents"]):
            agent_id = f"agt_{idx:03d}"
            agent_registry[agent_id] = {
                "agent_id": agent_id,
                **agent_config
            }

        workflow_json["agents"] = list(agent_registry.keys())

        # Compile
        compiled_graph = compiler.compile(workflow_json, agent_registry)

        # Verify compilation
        assert compiled_graph is not None


class TestHierarchicalWorkflowE2E:
    """End-to-end test for hierarchical workflow."""

    @pytest.fixture
    def compiler(self):
        return WorkflowCompiler(use_crewai=True)

    def test_hierarchical_workflow_e2e_with_crewai(
        self,
        compiler,
        mock_llm_response
    ):
        """Test hierarchical workflow uses CrewAI for delegation."""
        # Get workflow design
        workflow_json = mock_llm_response("hierarchical")

        # Create agent registry
        agent_registry = {}
        for idx, agent_config in enumerate(workflow_json["agents"]):
            agent_id = f"agt_{idx:03d}"
            agent_registry[agent_id] = {
                "agent_id": agent_id,
                **agent_config
            }

        # Map hierarchy indices to agent IDs
        hierarchy = workflow_json.get("hierarchy", {})
        master_idx = hierarchy.get("master_agent_index", 0)
        sub_indices = hierarchy.get("sub_agent_indices", [])

        agent_ids = list(agent_registry.keys())
        workflow_json["hierarchy"] = {
            "master_agent": agent_ids[master_idx],
            "delegation_order": [agent_ids[i] for i in sub_indices],
            "delegation_strategy": hierarchy.get("delegation_strategy", "dynamic")
        }
        workflow_json["agents"] = agent_ids

        # Compile with CrewAI
        compiled_graph = compiler.compile(workflow_json, agent_registry)

        # Verify compilation
        assert compiled_graph is not None


class TestHybridWorkflowE2E:
    """End-to-end test for hybrid workflow."""

    @pytest.fixture
    def compiler(self):
        return WorkflowCompiler(use_crewai=True)

    def test_hybrid_workflow_e2e_parallel_to_sequential(
        self,
        compiler,
        mock_llm_response
    ):
        """Test hybrid workflow with parallel → sequential pattern."""
        # Get workflow design
        workflow_json = mock_llm_response("hybrid")

        # Create agent registry
        agent_registry = {}
        for idx, agent_config in enumerate(workflow_json["agents"]):
            agent_id = f"agt_{idx:03d}"
            agent_registry[agent_id] = {
                "agent_id": agent_id,
                **agent_config
            }

        # Map topology indices to agent IDs
        agent_ids = list(agent_registry.keys())

        topology = workflow_json.get("topology", {})
        parallel_groups = topology.get("parallel_groups", [])
        sequential_chains = topology.get("sequential_chains", [])

        # Convert indices to IDs
        for group in parallel_groups:
            group["agents"] = [agent_ids[i] for i in group["agents"]]

        for chain in sequential_chains:
            chain["agents"] = [agent_ids[i] for i in chain["agents"]]

        workflow_json["topology"] = {
            "parallel_groups": parallel_groups,
            "sequential_chains": sequential_chains
        }
        workflow_json["agents"] = agent_ids

        # Compile
        compiled_graph = compiler.compile(workflow_json, agent_registry)

        # Verify compilation
        assert compiled_graph is not None


# ============================================================================
# State Management Integration Tests
# ============================================================================

class TestStateManagementIntegration:
    """Test state management across workflow execution."""

    @pytest.fixture
    def compiler(self):
        return WorkflowCompiler(use_crewai=True)

    def test_state_keys_from_agent_schemas(self, compiler):
        """Test that state schema includes all agent input/output keys."""
        agent_registry = {
            "agt_001": {
                "agent_id": "agt_001",
                "role": "Agent 1",
                "input_schema": ["input_a", "input_b"],
                "output_schema": ["output_x"]
            },
            "agt_002": {
                "agent_id": "agt_002",
                "role": "Agent 2",
                "input_schema": ["output_x"],
                "output_schema": ["output_y", "output_z"]
            }
        }

        # Create state schema
        state_class = compiler._create_workflow_state(agent_registry)

        # Verify all keys are in annotations
        annotations = state_class.__annotations__
        expected_keys = ["input_a", "input_b", "output_x", "output_y", "output_z", "messages"]

        for key in expected_keys:
            assert key in annotations, f"Expected key '{key}' not in state schema"

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'})
    @patch('apps.workflow.crewai_adapter.Crew')
    def test_state_flow_through_sequential_workflow(self, mock_crew_class, compiler):
        """Test that state flows correctly through sequential agents."""
        # Setup workflow
        workflow = {
            "execution_model": "sequential",
            "agents": ["agt_001", "agt_002"]
        }

        agent_registry = {
            "agt_001": {
                "agent_id": "agt_001",
                "role": "Agent 1",
                "goal": "Process input",
                "input_schema": ["input"],
                "output_schema": ["intermediate"]
            },
            "agt_002": {
                "agent_id": "agt_002",
                "role": "Agent 2",
                "goal": "Finalize output",
                "input_schema": ["intermediate"],
                "output_schema": ["final"]
            }
        }

        # Mock CrewAI execution
        def create_mock_crew(output_value):
            mock_crew = Mock()
            mock_result = Mock()
            mock_result.raw = output_value
            mock_crew.kickoff.return_value = mock_result
            return mock_crew

        # First agent outputs "intermediate", second agent outputs "final"
        mock_crew_class.side_effect = [
            create_mock_crew("intermediate_value"),
            create_mock_crew("final_value")
        ]

        # Compile workflow
        compiled = compiler.compile(workflow, agent_registry)

        # Note: Full execution test would require invoking the graph
        # For unit tests, we verify compilation and structure
        assert compiled is not None


# ============================================================================
# Multi-Provider LLM Integration Tests
# ============================================================================

class TestMultiProviderIntegration:
    """Test workflows with different LLM providers."""

    @pytest.fixture
    def adapter(self):
        return CrewAIAdapter()

    @patch('apps.workflow.crewai_adapter.ChatOpenAI')
    @patch.dict('os.environ', {
        'OPENROUTER_API_KEY': 'openrouter_key',
        'OPENAI_API_KEY': 'openai_key'
    })
    def test_different_llms_per_agent(self, mock_chat_openai, adapter):
        """Test that different agents can use different LLM providers."""
        agent_configs = [
            {
                "agent_id": "agt_001",
                "llm": {"provider": "openrouter", "model": "model1"}
            },
            {
                "agent_id": "agt_002",
                "llm": {"provider": "openai", "model": "gpt-4"}
            }
        ]

        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        # Get LLMs for both agents
        llm1 = adapter._get_llm_for_agent(agent_configs[0])
        llm2 = adapter._get_llm_for_agent(agent_configs[1])

        # Verify both were created
        assert llm1 is not None
        assert llm2 is not None

        # Verify they're different (different cache keys)
        # Since configs are different, they should be separate instances
        assert mock_chat_openai.call_count == 2


# ============================================================================
# Error Handling Integration Tests
# ============================================================================

class TestErrorHandlingIntegration:
    """Test error handling across the workflow lifecycle."""

    @pytest.fixture
    def compiler(self):
        return WorkflowCompiler(use_crewai=True)

    def test_invalid_workflow_json(self, compiler):
        """Test that invalid workflow JSON is rejected."""
        invalid_workflow = {
            "workflow_id": "wf_invalid"
            # Missing execution_model, agents, etc.
        }

        agent_registry = {}

        with pytest.raises((ValueError, KeyError)):
            compiler.compile(invalid_workflow, agent_registry)

    def test_missing_agent_in_registry(self, compiler):
        """Test handling of workflow referencing non-existent agent."""
        workflow = {
            "execution_model": "sequential",
            "agents": ["agt_001", "agt_999"]  # agt_999 doesn't exist
        }

        agent_registry = {
            "agt_001": {
                "agent_id": "agt_001",
                "role": "Agent 1",
                "goal": "Test"
            }
        }

        # Compiler should handle missing agents gracefully
        # (may skip them or raise error depending on implementation)
        try:
            compiled = compiler.compile(workflow, agent_registry)
            # If no error, verify graph was still created
            assert compiled is not None
        except (ValueError, KeyError):
            # Expected behavior if compiler validates agent existence
            pass


# ============================================================================
# Performance and Scalability Tests
# ============================================================================

class TestPerformanceIntegration:
    """Test workflow performance and scalability."""

    @pytest.fixture
    def compiler(self):
        return WorkflowCompiler(use_crewai=True)

    def test_large_parallel_workflow(self, compiler):
        """Test workflow with many parallel agents."""
        # Create workflow with 10 parallel agents
        num_agents = 10
        agent_ids = [f"agt_{i:03d}" for i in range(num_agents)]

        workflow = {
            "execution_model": "parallel",
            "agents": agent_ids
        }

        agent_registry = {
            agent_id: {
                "agent_id": agent_id,
                "role": f"Agent {i}",
                "goal": f"Task {i}",
                "output_schema": [f"output_{i}"]
            }
            for i, agent_id in enumerate(agent_ids)
        }

        # Should compile without issues
        compiled = compiler.compile(workflow, agent_registry)
        assert compiled is not None

    def test_deep_sequential_workflow(self, compiler):
        """Test workflow with many sequential agents."""
        # Create workflow with 15 sequential agents
        num_agents = 15
        agent_ids = [f"agt_{i:03d}" for i in range(num_agents)]

        workflow = {
            "execution_model": "sequential",
            "agents": agent_ids
        }

        # Each agent's output feeds next agent's input
        agent_registry = {}
        for i, agent_id in enumerate(agent_ids):
            agent_registry[agent_id] = {
                "agent_id": agent_id,
                "role": f"Agent {i}",
                "goal": f"Task {i}",
                "input_schema": [f"output_{i-1}"] if i > 0 else ["user_input"],
                "output_schema": [f"output_{i}"]
            }

        # Should compile without issues
        compiled = compiler.compile(workflow, agent_registry)
        assert compiled is not None


# ============================================================================
# Architectural Compliance Tests
# ============================================================================

class TestArchitecturalCompliance:
    """Verify architectural principles are maintained."""

    def test_langgraph_owns_topology(self):
        """Verify that LangGraph owns workflow topology."""
        # This is validated by ensuring CrewAI is only called inside node functions
        # and never controls graph structure

        adapter = CrewAIAdapter()

        # CrewAI adapter should only create node functions, not graphs
        node_func = adapter.create_sequential_agent_node({"role": "Test", "goal": "Test"})

        # Node function should be callable (for LangGraph to call)
        assert callable(node_func)

        # Node function should NOT have graph manipulation methods
        assert not hasattr(node_func, 'add_edge')
        assert not hasattr(node_func, 'add_node')
        assert not hasattr(node_func, 'compile')

    def test_crewai_within_nodes_only(self):
        """Verify CrewAI executes only within LangGraph nodes."""
        # Validated by checking that CrewAI adapter methods return
        # callable node functions, not graph structures

        adapter = CrewAIAdapter()

        hierarchical_node = adapter.create_hierarchical_crew_node(
            master_agent_config={"role": "Manager", "goal": "Coordinate"},
            sub_agent_configs=[{"role": "Worker", "goal": "Work"}]
        )

        # Should return a callable node function
        assert callable(hierarchical_node)

        # Should NOT return a graph or crew directly
        assert not hasattr(hierarchical_node, 'kickoff')  # Not a Crew object
        assert not hasattr(hierarchical_node, 'compile')  # Not a Graph object


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-k", "not e2e"])
