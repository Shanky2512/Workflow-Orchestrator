"""
EchoAI Orchestrator -- Prompt Enhancer

Performs a single LLM call to improve the user's raw input:
    - Fixes typos and grammatical errors
    - Clarifies ambiguous intent
    - Improves readability

The enhancer does NOT answer the user's question.  It only restructures
the text and returns the enhanced version alongside detected intent
and a confidence score.

Uses a module-level ChatOpenAI instance.  Fill in the ``model``,
``base_url``, and ``api_key`` placeholders before running.

Graceful degradation: if the LLM call fails for any reason, the
original input is passed through unchanged.
"""

import asyncio
import json
import logging
from typing import Any, Dict

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

# System prompt for the enhancer LLM
_ENHANCER_SYSTEM_PROMPT = (
    "You are a prompt enhancer. Your job is to improve the user's prompt.\n"
    "Fix typos, clarify intent, improve grammar, and make the prompt clearer.\n"
    "Do NOT answer the question. Do NOT add information.\n"
    "Return ONLY valid JSON with these keys:\n"
    '  "enhanced_prompt": the improved prompt text,\n'
    '  "detected_intent": a short phrase describing what the user wants,\n'
    '  "confidence": a float between 0.0 and 1.0 indicating how confident '
    "you are about the detected intent.\n"
    "Return nothing else -- only the JSON object."
)


class PromptEnhancer:
    """
    LLM-powered prompt enhancer.

    Uses LangChain ``SystemMessage`` / ``HumanMessage`` with the
    module-level ``ChatOpenAI`` instance defined above.
    """

    async def enhance(
        self,
        user_input: str,
    ) -> Dict[str, Any]:
        """
        Enhance a user prompt via an LLM call.

        Args:
            user_input: Raw user message.

        Returns:
            Dict with keys:
                enhanced_prompt (str) -- the improved prompt
                detected_intent (str) -- short intent description
                confidence (float) -- intent confidence 0.0-1.0
        """
        fallback = {
            "enhanced_prompt": user_input,
            "detected_intent": "unknown",
            "confidence": 0.0,
        }

        if not user_input or not user_input.strip():
            return fallback

        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=_ENHANCER_SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ]

            # LangChain .invoke() is synchronous -- wrap in thread
            response = await asyncio.to_thread(llm.invoke, messages)
            raw_text = response.content if hasattr(response, "content") else str(response)

            return self._parse_response(raw_text, user_input)

        except Exception:
            logger.exception("Prompt enhancement failed; passing through original input")
            return fallback

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str, original_input: str) -> Dict[str, Any]:
        """
        Parse the LLM JSON response into the expected dict structure.

        Falls back to the original input if parsing fails.
        """
        fallback = {
            "enhanced_prompt": original_input,
            "detected_intent": "unknown",
            "confidence": 0.0,
        }

        try:
            # Strip markdown code fences if present
            text = raw_text.strip()
            if text.startswith("```"):
                # Remove opening fence (possibly ```json)
                text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()

            data = json.loads(text)

            enhanced = data.get("enhanced_prompt", original_input)
            intent = data.get("detected_intent", "unknown")
            confidence = data.get("confidence", 0.0)

            # Validate types
            if not isinstance(enhanced, str):
                enhanced = str(enhanced)
            if not isinstance(intent, str):
                intent = str(intent)
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.0

            return {
                "enhanced_prompt": enhanced,
                "detected_intent": intent,
                "confidence": confidence,
            }

        except (json.JSONDecodeError, AttributeError, KeyError):
            logger.warning("Could not parse enhancer response as JSON; using raw text")
            # If we got text but it is not JSON, use it as the enhanced prompt
            if raw_text and raw_text.strip():
                return {
                    "enhanced_prompt": raw_text.strip(),
                    "detected_intent": "unknown",
                    "confidence": 0.0,
                }
            return fallback
