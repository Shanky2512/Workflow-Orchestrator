"""
Code Generator Service
Generates CrewAI and LangGraph code using LLM based on workflow configuration
"""
import json
import logging
from typing import Dict, Any

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    SystemMessage = None
    HumanMessage = None

logger = logging.getLogger(__name__)


class CodeGeneratorService:
    """Service for generating agentic code from workflows"""

    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            logger.warning("langchain_openai not installed. Code generation will not work. Install with: pip install langchain-openai")
            self.llm = None
            return

        # Initialize LLM with fixed configuration
        # self.llm = ChatOpenAI(
        #     base_url="http://10.188.100.130:8002/v1",
        #     model="gpt-oss:20b",
        #     api_key="ollama"
        # )
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            api_key="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"
        )

    def generate_crewai_code(self, workflow_json: Dict[str, Any]) -> str:
        """
        Generate CrewAI code from workflow configuration

        Args:
            workflow_json: Complete workflow configuration as dictionary

        Returns:
            Generated Python code using CrewAI
        """
        if not LANGCHAIN_AVAILABLE or self.llm is None:
            raise RuntimeError("langchain_openai is not installed. Please install: pip install langchain-openai")

        try:
            # Convert workflow to JSON string for task description
            task_description = f"""
Generate a complete Python script using CrewAI library based on this workflow configuration:

{json.dumps(workflow_json, indent=2)}

Requirements:
- Use the agents defined in the workflow
- Implement the execution mode: {workflow_json.get('execution_mode', 'sequential')}
- Include all agent tools and configurations
- Create proper tasks for each agent based on their goals
- Use CrewAI's Crew, Agent, Task, and Process classes
- Include proper imports and error handling
- Make the code executable and production-ready
"""

            system_prompt = SystemMessage(
                content="You are an expert Python agent developer. Generate Python code using the CrewAI library for the following task. Only output valid, executable Python code without any explanations or markdown."
            )
            human_prompt = HumanMessage(content=task_description)

            logger.info("Generating CrewAI code from workflow")
            response = self.llm.invoke([system_prompt, human_prompt])

            return response.content

        except Exception as e:
            logger.error(f"Failed to generate CrewAI code: {str(e)}")
            raise

    def generate_langgraph_code(self, workflow_json: Dict[str, Any]) -> str:
        """
        Generate LangGraph code from workflow configuration

        Args:
            workflow_json: Complete workflow configuration as dictionary

        Returns:
            Generated Python code using LangGraph
        """
        if not LANGCHAIN_AVAILABLE or self.llm is None:
            raise RuntimeError("langchain_openai is not installed. Please install: pip install langchain-openai")

        try:
            # Convert workflow to JSON string for task description
            task_description = f"""
Generate a complete Python script using LangGraph library based on this workflow configuration:

{json.dumps(workflow_json, indent=2)}

Requirements:
- Use the agents defined in the workflow
- Implement the execution mode: {workflow_json.get('execution_mode', 'sequential')}
- Create a state graph with nodes for each agent
- Define edges based on the connections
- Include all agent tools and configurations
- Use LangGraph's StateGraph, END, and node definitions
- Include proper imports and error handling
- Make the code executable and production-ready
"""

            system_prompt = SystemMessage(
                content="You are an expert Python agent developer. Generate Python code using the LangGraph library for the following task. Only output valid, executable Python code without any explanations or markdown."
            )
            human_prompt = HumanMessage(content=task_description)

            logger.info("Generating LangGraph code from workflow")
            response = self.llm.invoke([system_prompt, human_prompt])

            return response.content

        except Exception as e:
            logger.error(f"Failed to generate LangGraph code: {str(e)}")
            raise

    def generate_both_codes(self, workflow_json: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate both CrewAI and LangGraph code

        Args:
            workflow_json: Complete workflow configuration

        Returns:
            Dictionary with 'crewai_code' and 'langgraph_code' keys
        """
        logger.info("Generating both CrewAI and LangGraph code")

        crewai_code = self.generate_crewai_code(workflow_json)
        langgraph_code = self.generate_langgraph_code(workflow_json)

        return {
            "crewai_code": crewai_code,
            "langgraph_code": langgraph_code
        }
