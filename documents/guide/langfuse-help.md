# Langfuse Observability & Token Usage Guide

## Step 1: Open Langfuse Dashboard

Go to: https://cloud.langfuse.com\
Log in with the account that owns your API keys.

------------------------------------------------------------------------

## Step 2: What to Look For

### All Traces

Navigate to **Traces tab** → Shows every workflow run.

### Token Counts

Traces → Click any trace → Click a **Generation span**\
You will see: - input_tokens - output_tokens - total_tokens

### LLM Costs

Inside the same Generation span → Shows cost (if model pricing is
configured).

### Agent Execution Flow

Click a trace → View the span tree: LangGraph node → CrewAI → Agent →
LLM call

### Time-Travel Replay

Click a trace → Step through spans in chronological order.

### Prompt Injection Alerts

Go to **Scores tab** → Filter by `prompt_injection_risk`.

### Fine-Tuning Datasets

Go to **Datasets tab** (if using save_completion_for_finetuning()).

------------------------------------------------------------------------

## Step 3: Quick Verification Endpoint

Call:

GET http://localhost:8000/health/observability

Expected response:

{ "status": "ok", "langfuse_tracing_enabled": true, "langfuse_base_url":
"https://cloud.langfuse.com", "sample_rate": 1.0 }

------------------------------------------------------------------------

## Step 4: Configure Model Pricing (Cost Tracking)

Dashboard → Settings → Model Prices → Add your model\
Example: nvidia/nemotron-3-nano-30b-a3b:free

Without model pricing configured: - Token counts appear - Cost shows
\$0.00

------------------------------------------------------------------------

## How to Check Token Usage

Langfuse Cloud → Traces → Click a trace → Expand span tree → Click
"Generation"

Example:

Model: nvidia/nemotron-3-nano-30b-a3b:free\
Input tokens: 1,234\
Output tokens: 567\
Total tokens: 1,801\
Latency: 2.3s\
Cost: \$0.00 (free model)

------------------------------------------------------------------------

## Aggregate Usage Across All Runs

Dashboard (home page) shows: - Total tokens (input/output) over time -
Token usage per model - Cost breakdown per model - Latency percentiles

------------------------------------------------------------------------

## If Token Count Shows 0 or Missing

Possible reason: Your provider (e.g., OpenRouter free model) does not
return usage metadata in API response.

Solution: Switch to OpenAI or Azure models for automatic token tracking.

------------------------------------------------------------------------

## If Total Token Count Shows 6

This usually means:

1.  The LLM returned a very short response.
2.  Free-tier model truncated the output.
3.  The request errored early.

To verify: - Click the trace - Open the Generation span - Check input
tokens - Check output tokens - Inspect actual response text

Langfuse only reports what the provider returns.
