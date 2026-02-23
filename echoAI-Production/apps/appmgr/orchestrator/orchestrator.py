"""
EchoAI Orchestrator -- Core LLM Intelligence

The orchestrator is the "brain" of the application pipeline.
Given the enhanced prompt, skill manifest, conversation history,
persona instructions, and guardrail rules, it decides:

    1. Which skills to invoke
    2. In what order (sequential / parallel / hybrid)
    3. How to chain inputs and outputs between steps
    4. Whether clarification is needed (HITL)
    5. Whether to return a fallback message (out-of-scope)

The orchestrator returns an execution plan as a structured JSON
object that the SkillExecutor can carry out.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Module-level LLM instance -- fill in before running
# ------------------------------------------------------------------
llm = ChatOpenAI(
    model="liquid/lfm-2.5-1.2b-instruct:free",          # user will fill
    base_url="https://openrouter.ai/api/v1",       # user will fill
    api_key="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",        # user will fill
    temperature=0.2,
)

# ---------------------------------------------------------------------------
# Alternative: OpenAI SDK (uncomment to use instead of LangChain)
# ---------------------------------------------------------------------------
# from openai import OpenAI
# openai_client = OpenAI(
#     api_key="sk-or-v1-your-key-here",      # replace with your key
#     base_url="https://openrouter.ai/api/v1", # or https://api.openai.com/v1
# )
# OPENAI_MODEL = "liquid/lfm-2.5-1.2b-instruct:free"  # replace with your model
#
# def _call_openai(system_prompt: str, user_message: str) -> str:
#     """Call OpenAI-compatible API directly using the openai SDK."""
#     response = openai_client.chat.completions.create(
#         model=OPENAI_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message},
#         ],
#         temperature=0.2,
#     )
#     return response.choices[0].message.content or ""
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# System Prompt Template
# ---------------------------------------------------------------------------

_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are an AI orchestrator that decides which skills (workflows and agents) to invoke based on the user's request.

{app_identity_section}

{persona_section}

{guardrail_section}

{disclaimer_section}

{tags_section}

{access_context_section}

{data_sources_section}

{document_context_section}

## Available Skills

{skill_manifest_section}

## Conversation History

{history_section}

## Instructions

1. Analyze the user's request and determine which skill(s) to invoke.
2. If the request maps to a single skill, use execution_strategy "single".
3. If multiple skills are needed in sequence (output of one feeds the next), use "sequential".
4. If multiple skills can run independently in parallel, use "parallel".
5. If a mix of dependent and independent steps is needed, use "hybrid".
6. If the user's request is unclear or ambiguous, set needs_clarification=true and provide a clarification_question.
7. If the request is completely out-of-scope (jokes, off-topic, general chat not related to any skill), set fallback_message to a helpful explanation of your capabilities.
8. Never invent skill IDs or names -- only use skills from the Available Skills list.
9. If no skills match the request at all, set fallback_message.
10. If the application has a disclaimer, be aware of it when responding and ensure your responses are consistent with it.

## Output Format

Return ONLY valid JSON (no markdown, no commentary) with this exact structure:
{{
    "reasoning": "Brief explanation of your decision",
    "execution_strategy": "single|sequential|parallel|hybrid",
    "execution_plan": [
        {{
            "step": 1,
            "skill_id": "uuid-of-the-skill",
            "skill_type": "workflow|agent",
            "skill_name": "Human-readable name",
            "depends_on": [],
            "parallel_group": null,
            "input_source": "user_input|step_N_output|merged",
            "output_key": "result_identifier"
        }}
    ],
    "final_output_from": "step_N|merged",
    "needs_clarification": false,
    "clarification_question": null,
    "fallback_message": null
}}
"""


class Orchestrator:
    """
    Core orchestrator LLM that produces an execution plan from the
    user's enhanced prompt and available skill manifest.
    """

    async def plan(
        self,
        enhanced_prompt: str,
        skill_manifest: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        persona_prompt: str,
        guardrail_rules: str,
        document_context: Optional[str] = None,
        app_tags: Optional[List[str]] = None,
        app_designations: Optional[List[str]] = None,
        app_business_units: Optional[List[str]] = None,
        app_name: Optional[str] = None,
        app_description: Optional[str] = None,
        app_disclaimer: Optional[str] = None,
        app_data_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an execution plan for the given prompt.

        Args:
            enhanced_prompt: The prompt-enhanced user message.
            skill_manifest: Dict from SkillManifestBuilder.build_manifest().
            conversation_history: List of {"role": ..., "content": ...} dicts
                (most recent last, limited to last N messages).
            persona_prompt: Persona instruction string (may be empty).
            guardrail_rules: Guardrail rule summary string (may be empty).
            document_context: Optional document context retrieved from
                session-scoped RAG index (may be None).
            app_tags: Optional list of tag names assigned to the application.
            app_designations: Optional list of designation names for access context.
            app_business_units: Optional list of business unit names for access context.
            app_name: Optional application name for identity context.
            app_description: Optional application description for identity context.
            app_disclaimer: Optional disclaimer text for compliance awareness.
            app_data_sources: Optional list of data source names connected to the app.

        Returns:
            Execution plan dict matching the JSON schema above.
        """
        fallback = self._empty_plan(
            fallback_message="I was unable to determine how to handle your request. "
            "Please try rephrasing."
        )

        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            system_prompt = self._build_system_prompt(
                skill_manifest,
                conversation_history,
                persona_prompt,
                guardrail_rules,
                document_context=document_context,
                app_tags=app_tags,
                app_designations=app_designations,
                app_business_units=app_business_units,
                app_name=app_name,
                app_description=app_description,
                app_disclaimer=app_disclaimer,
                app_data_sources=app_data_sources,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=enhanced_prompt),
            ]

            # LangChain .invoke() is synchronous -- wrap in thread
            response = await asyncio.to_thread(llm.invoke, messages)
            raw_text = response.content if hasattr(response, "content") else str(response)

            return self._parse_plan(raw_text)

        except Exception:
            logger.exception("Orchestrator LLM call failed")
            return fallback

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        skill_manifest: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        persona_prompt: str,
        guardrail_rules: str,
        document_context: Optional[str] = None,
        app_tags: Optional[List[str]] = None,
        app_designations: Optional[List[str]] = None,
        app_business_units: Optional[List[str]] = None,
        app_name: Optional[str] = None,
        app_description: Optional[str] = None,
        app_disclaimer: Optional[str] = None,
        app_data_sources: Optional[List[str]] = None,
    ) -> str:
        """Assemble the full system prompt from parts."""
        # App identity section
        app_identity_section = ""
        if app_name:
            identity_parts = [f"You are the AI assistant for the application **{app_name}**."]
            if app_description:
                identity_parts.append(f"Application purpose: {app_description}")
            app_identity_section = "## Application Identity\n\n" + " ".join(identity_parts)

        # Persona section
        persona_section = ""
        if persona_prompt:
            persona_section = f"## Persona\n\n{persona_prompt}"

        # Guardrail section
        guardrail_section = ""
        if guardrail_rules:
            guardrail_section = (
                "## Guardrail Rules\n\n"
                "You must respect these rules when selecting skills and "
                "formulating responses:\n"
                f"{guardrail_rules}"
            )

        # Disclaimer section
        disclaimer_section = ""
        if app_disclaimer:
            disclaimer_section = (
                "## Disclaimer\n\n"
                "This application has the following disclaimer that users see:\n"
                f"\"{app_disclaimer}\"\n\n"
                "Be mindful of this disclaimer in your responses. Do not contradict it."
            )

        # Tags section
        tags_section = ""
        if app_tags:
            tags_section = (
                "## Application Tags\n\n"
                f"This application is tagged as: {', '.join(app_tags)}. "
                "Consider these tags as contextual metadata when selecting "
                "skills and formulating responses."
            )

        # Access context section
        access_context_section = ""
        access_parts = []
        if app_designations:
            access_parts.append(f"Designated for: {', '.join(app_designations)}")
        if app_business_units:
            access_parts.append(f"Business units: {', '.join(app_business_units)}")
        if access_parts:
            access_context_section = (
                "## Access Context\n\n"
                f"{'. '.join(access_parts)}. "
                "Tailor responses appropriately for this audience."
            )

        # Data sources section
        data_sources_section = ""
        if app_data_sources:
            data_sources_section = (
                "## Connected Data Sources\n\n"
                f"This application has access to the following data sources: "
                f"{', '.join(app_data_sources)}. "
                "Consider these when determining which skills to invoke and "
                "what data is available for answering the user's request."
            )

        # Document context section
        document_context_section = ""
        if document_context:
            document_context_section = (
                "## Uploaded Document Context\n\n"
                "The user has uploaded documents in this conversation. The following relevant "
                "excerpts were retrieved based on their current question. Use this context "
                "when selecting skills and understand that the skills will also receive this "
                "context as input.\n\n"
                f"{document_context}"
            )

        # Skill manifest section
        skills = skill_manifest.get("skills", [])
        if skills:
            skill_lines = []
            for s in skills:
                line = (
                    f"- **{s['name']}** (id={s['skill_id']}, type={s['skill_type']}): "
                    f"{s.get('description', 'No description')}"
                )
                caps = s.get("capabilities", [])
                if caps:
                    line += f" | Capabilities: {', '.join(caps)}"
                tools = s.get("tools", [])
                if tools:
                    line += f" | Tools: {', '.join(tools)}"
                em = s.get("execution_model")
                if em:
                    line += f" | Execution model: {em}"
                skill_lines.append(line)
            skill_manifest_section = "\n".join(skill_lines)
        else:
            skill_manifest_section = "(No skills available)"

        # Conversation history section (last 8 messages = 4 exchanges max)
        # Already limited to 8 by the pipeline; defensive slice here
        history_msgs = conversation_history[-8:] if conversation_history else []
        if history_msgs:
            history_lines = []
            # Token-budget aware truncation:
            # - Recent messages (last 4): full content up to 500 chars
            # - Older messages (first 4): truncated to 200 chars
            for idx, msg in enumerate(history_msgs):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Newer messages get more space
                max_chars = 500 if idx >= len(history_msgs) - 4 else 200
                if len(content) > max_chars:
                    content = content[:max_chars] + "..."
                history_lines.append(f"[{role}]: {content}")
            history_section = "\n".join(history_lines)
        else:
            history_section = "(No prior conversation)"

        return _ORCHESTRATOR_SYSTEM_PROMPT.format(
            app_identity_section=app_identity_section,
            persona_section=persona_section,
            guardrail_section=guardrail_section,
            disclaimer_section=disclaimer_section,
            tags_section=tags_section,
            access_context_section=access_context_section,
            data_sources_section=data_sources_section,
            document_context_section=document_context_section,
            skill_manifest_section=skill_manifest_section,
            history_section=history_section,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_plan(self, raw_text: str) -> Dict[str, Any]:
        """Parse the LLM JSON response into the execution plan dict."""
        try:
            text = raw_text.strip()
            # Strip markdown code fences
            if text.startswith("```"):
                text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()

            plan = json.loads(text)

            # Validate required keys and fill defaults
            plan.setdefault("reasoning", "")
            plan.setdefault("execution_strategy", "single")
            plan.setdefault("execution_plan", [])
            plan.setdefault("final_output_from", "step_1")
            plan.setdefault("needs_clarification", False)
            plan.setdefault("clarification_question", None)
            plan.setdefault("fallback_message", None)

            # Normalize execution_plan entries
            for step in plan.get("execution_plan", []):
                step.setdefault("step", 1)
                step.setdefault("skill_id", "")
                step.setdefault("skill_type", "workflow")
                step.setdefault("skill_name", "")
                step.setdefault("depends_on", [])
                step.setdefault("parallel_group", None)
                step.setdefault("input_source", "user_input")
                step.setdefault("output_key", f"step_{step['step']}_output")

            return plan

        except (json.JSONDecodeError, AttributeError, KeyError) as exc:
            logger.warning("Failed to parse orchestrator response as JSON: %s", exc)
            # Attempt to extract useful information from raw text
            return self._empty_plan(
                fallback_message=(
                    "I encountered an issue processing your request. "
                    "Please try again or rephrase your question."
                )
            )

    @staticmethod
    def _empty_plan(
        fallback_message: Optional[str] = None,
        clarification_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a minimal valid plan."""
        return {
            "reasoning": "Unable to produce execution plan",
            "execution_strategy": "single",
            "execution_plan": [],
            "final_output_from": "none",
            "needs_clarification": clarification_question is not None,
            "clarification_question": clarification_question,
            "fallback_message": fallback_message,
        }
