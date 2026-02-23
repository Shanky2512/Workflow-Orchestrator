# Fix: Token Burn from Tool Retry Loops & Related Workflow Bugs

## Context

The DueDiligence workflow (`wf_140bb0af94bc4001804c4e2e1d395242`) burns excessive tokens during execution. The root cause is **not** a graph-level infinite loop (step_started/step_completed are balanced at 8 each). Instead, token burn originates from **agent-internal tool retry loops** where each retry appends the full error+context to the prompt, causing exponential token growth.

Seven interconnected bugs contribute to this problem. Fixes are ordered by impact.

---

## Fixes

### Fix 1: Deduplicate tool bindings in `_bind_tools_to_agent()`
**File:** `echoAI-Production/apps/workflow/crewai_adapter.py` ~line 291-327

**Problem:** The Document Classifier agent lists `tool_document_analysis` twice in its tools array (DueDeligence.json lines 60-61). `_bind_tools_to_agent()` iterates the list and creates a new wrapper for each entry with no dedup. The LLM sees duplicate tools, gets confused, and may invoke both — doubling token cost.

**Fix:** Deduplicate `tool_ids` before the loop:
```python
tool_ids = agent_config.get("tools", [])
tool_ids = list(dict.fromkeys(tool_ids))  # deduplicate, preserve order
```

---

### Fix 2: Case-insensitive provider validation in tool input
**File:** `echoAI-Production/apps/tool/executor.py` ~line 209-234

**Problem:** Web Search tool manifest defines `"enum": ["bing"]` (lowercase). When the LLM passes `"Bing"` (capitalized), `jsonschema.validate()` fails with enum mismatch. This causes a validation error returned to the agent, which retries with the same input, burning tokens on each retry.

**Fix:** Normalize string enum fields to lowercase before validation:
```python
def _validate_input(self, tool, input_data):
    if not tool.input_schema:
        return
    # Normalize string fields that have enum constraints to lowercase
    normalized = self._normalize_enum_fields(tool.input_schema, input_data)
    jsonschema.validate(instance=normalized, schema=tool.input_schema)
```

Add helper `_normalize_enum_fields()` that walks the schema, finds string properties with `"enum"` where all enum values are lowercase, and lowercases the corresponding input value.

---

### Fix 3: Make Document Analysis input validation lenient for optional fields
**File:** `echoAI-Production/apps/tool/executor.py` ~line 209-234

**Problem:** The LLM passes `policy_rules: null` but the schema requires `type: object`. Since `policy_rules` is not in the `required` list, the fix is to allow null for non-required fields.

**Fix:** Before calling `jsonschema.validate`, strip `None` values from input_data when the corresponding schema property is not in `required`:
```python
# Strip None values for non-required fields before validation
cleaned = {k: v for k, v in input_data.items()
           if v is not None or k in (tool.input_schema.get("required", []))}
```

---

### Fix 4: Fix provider/model passthrough from agent config to LLMManager
**File:** `echoAI-Production/apps/workflow/crewai_adapter.py` ~line 1123-1148

**Problem:** Two issues compound:
1. Agent configs store provider in `settings.provider` and model at top-level `model`, but `_get_llm_for_agent()` looks at `agent_config.get("llm", {})` — which is always `{}`.
2. Lines 1147-1148 force-override `provider = None; model = None` regardless.

Result: ALL agents use LLMManager defaults (openrouter/nemotron-nano), ignoring per-agent configs that specify `anthropic/claude-opus-4`.

**Fix:**
1. Extract provider/model from the correct config paths (`settings.provider`, top-level `model`)
2. Remove the force-override lines (1147-1148)
3. Only fall through to LLMManager defaults when values are actually empty/None
4. Validate that the requested provider has a real API key configured in LLMManager before using it; if not, log a warning and fall back to defaults

```python
def _get_llm_for_agent(self, agent_config):
    from llm_manager import LLMManager

    # Extract from correct config paths
    settings = agent_config.get("settings", {})
    provider = settings.get("provider") or None
    model = agent_config.get("model") or None
    temperature = settings.get("temperature")
    max_tokens = settings.get("max_token")  # note: max_token not max_tokens

    # Validate provider has a configured API key; fall back if not
    if provider and provider != "ollama":
        api_key = LLMManager.API_KEYS.get(provider, "")
        if not api_key or "YOUR_" in api_key or "_HERE" in api_key:
            logger.warning(f"Provider '{provider}' requested but API key not configured, falling back to defaults")
            provider = None
            model = None

    logger.info(f"Agent LLM: provider={provider or 'default'}, model={model or 'default'}")

    return LLMManager.get_crewai_llm(
        provider=provider, model=model,
        temperature=temperature, max_tokens=max_tokens
    )
```

---

### Fix 5: Fix `risk_score` type in state_schema and conditional evaluation
**Files:**
- `echoAI-Production/apps/workflow/designer/compiler.py` ~line 1658-1681
- `echoAI-Production/apps/workflow/crewai_adapter.py` `_coerce_value()` ~line 1188

**Problem:** `state_schema` declares `risk_score` as `"string"` (DueDeligence.json:618), so `_coerce_value("72", "string")` stores it as the string `"72"`. The conditional evaluator's coercion (compiler.py:1674) tries `int("72")` and succeeds for clean numbers, but fails for strings like `"72 (High Risk)"` or `"72/100"`, causing `'>' not supported between instances of 'str' and 'int'`.

**Fix (compiler.py — defensive):** In the conditional evaluator, wrap the eval in a try/except for TypeError and attempt numeric coercion of operands as a fallback:
```python
try:
    result = eval(normalized, safe_globals, safe_locals)
except TypeError as e:
    if "not supported between" in str(e):
        # Retry: force-coerce referenced variables to numeric
        for var in referenced_vars:
            val = safe_locals.get(var)
            if isinstance(val, str):
                num = re.search(r'-?\d+(?:\.\d+)?', val)
                if num:
                    safe_locals[var] = float(num.group()) if '.' in num.group() else int(num.group())
        result = eval(normalized, safe_globals, safe_locals)
    else:
        raise
```

This is a runtime defense. The workflow JSON should also be fixed to declare `risk_score` as `"number"`, but that's a data fix, not a code fix.

---

### Fix 6: Add SSL certificate handling for MCP/API tool execution
**File:** `echoAI-Production/apps/tool/executor.py` ~line 419-449

**Problem:** `httpx.AsyncClient()` defaults to SSL verification. In dev environments with self-signed certs, MCP and API calls fail with `CERTIFICATE_VERIFY_FAILED`, triggering retries.

**Fix:** Add a configurable `verify_ssl` parameter with environment variable override:
```python
import os
_VERIFY_SSL = os.environ.get("ECHOAI_VERIFY_SSL", "true").lower() != "false"

async with httpx.AsyncClient(verify=_VERIFY_SSL) as client:
    ...
```

Apply the same to `_execute_api()` as well.

---

### Fix 7: Cap token growth from previous_output in task descriptions
**File:** `echoAI-Production/apps/workflow/crewai_adapter.py` ~line 893-899

**Problem:** Each sequential agent receives the FULL `previous_output` (which can be thousands of tokens) and FULL `parallel_output`. During retries, the same large context is re-sent each time. Across a multi-agent pipeline, this compounds — later agents receive ALL prior outputs concatenated.

**Fix:** Truncate `previous_output` and `parallel_output` to a configurable max length:
```python
MAX_CONTEXT_CHARS = 3000  # ~750 tokens

if previous_output:
    truncated = previous_output[:MAX_CONTEXT_CHARS]
    if len(previous_output) > MAX_CONTEXT_CHARS:
        truncated += f"\n... [truncated from {len(previous_output)} chars]"
    task_parts.append(f"PREVIOUS AGENT OUTPUT:\n{truncated}")

if parallel_output and parallel_output != previous_output:
    truncated = parallel_output[:MAX_CONTEXT_CHARS]
    if len(parallel_output) > MAX_CONTEXT_CHARS:
        truncated += f"\n... [truncated from {len(parallel_output)} chars]"
    task_parts.append(f"PARALLEL EXECUTION RESULTS:\n{truncated}")
```

---

## Files to Modify

| # | File | Changes |
|---|------|---------|
| 1 | `echoAI-Production/apps/workflow/crewai_adapter.py` | Deduplicate tools (Fix 1), Fix provider/model extraction (Fix 4), Truncate context (Fix 7) |
| 2 | `echoAI-Production/apps/tool/executor.py` | Normalize enum inputs (Fix 2), Strip null non-required (Fix 3), SSL verify config (Fix 6) |
| 3 | `echoAI-Production/apps/workflow/designer/compiler.py` | TypeError fallback in conditional eval (Fix 5) |

---

## Verification

1. **Fix 1 (dedup):** Run a workflow where an agent has duplicate tools — should log binding only unique tools
2. **Fix 2 (enum):** Call web_search tool with `provider: "Bing"` — should normalize to `"bing"` and pass validation
3. **Fix 3 (null fields):** Call document_analysis with `policy_rules: null` — should strip null and pass
4. **Fix 4 (provider):** Run workflow with agents specifying `provider: "anthropic"` — log should show `provider=anthropic` not `provider=None`
5. **Fix 5 (risk_score):** Conditional with string risk_score like `"72 (High)"` — should extract `72` and evaluate `72 > 70` correctly
6. **Fix 6 (SSL):** Set `ECHOAI_VERIFY_SSL=false` and run MCP tools — should not fail on self-signed certs
7. **Fix 7 (truncation):** Run multi-agent pipeline — later agents should receive truncated prior context, not full output




# =========== DONE TILL NOW ================

All 4 fixes are implemented. Here's the summary:
                          
  What was done — all centralized, zero feature-specific patches                                                
                                                                                                     Fix:1                                 
  File: crewai_adapter.py:292                                                                                   
  What: tool_ids = list(dict.fromkeys(tool_ids))                                                                
  Why it's system-wide: _bind_tools_to_agent() is the single entry point for ALL agent types (sequential, parallel, hierarchical). Every agent auto-inherits dedup.                                                     ────────────────────────────────────────                                                           Fix:2                                                                                                        
  File: executor.py:209-244
  What: _normalize_enum_fields() static method
  Why it's system-wide: Called inside _validate_input() which is the single validation gate for ALL tool types  
    (local, MCP, API, CrewAI). Every tool auto-inherits.
  ────────────────────────────────────────
  Fix: 3
  File: executor.py:246-266
  What: _strip_null_optional_fields() static method
  Why it's system-wide: Same — called inside _validate_input(). Every tool auto-inherits.
  ────────────────────────────────────────
  Fix: 5
  File: compiler.py:1733-1767
  What: TypeError catch with numeric coercion retry
  Why it's system-wide: Inside _evaluate_condition() which is the single conditional evaluator for ALL
  workflows.
    Every conditional branch auto-inherits.

  How this stops the 10-20x tool retry loop

  The root cause: _validate_input() rejects valid-intent input (wrong case enum, null optional field) → returns 
  error to agent → agent retries with identical input → same error → 10-20 cycles burning tokens.

  Now _validate_input() silently normalizes the input before jsonschema.validate() runs, so the first call      
  succeeds. The normalizations are written back into input_data (line 298-299) so downstream execution also sees
   clean data. No tool, no workflow, no agent needs any awareness of these fixes.