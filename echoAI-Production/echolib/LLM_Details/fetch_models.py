# sync_llm_provider_from_ollama_like_server.py
import json
import os
import sys
import time
from typing import List, Dict, Any
import requests
from pathlib import Path

# === Config ===
SERVER_BASE_URL = "http://10.188.100.130:8002"
TAGS_ENDPOINT = f"{SERVER_BASE_URL.rstrip('/')}/api/tags"
current_path = Path(__file__).resolve().parent
LLM_JSON_PATH = current_path/"Providers"/"llm_provider.json"

# A key to identify this server/provider in your schema (free text)
PROVIDER_KEY = os.getenv("PROVIDER_KEY", "ollama")
PROVIDER_DISPLAY_NAME = os.getenv("PROVIDER_DISPLAY_NAME", "In-house Models")

# Optional: choose a default model by name if present
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "")  # e.g., "qwen2.5:14b"

# If you insist on embedding a key in llm_provider.json (not recommended),
# set this env var. Otherwise we’ll store null and use server-side env in the app.
EMBEDDED_API_KEY = os.getenv("EMBEDDED_API_KEY", None)


def fetch_models_from_server(timeout=10) -> List[Dict[str, Any]]:
    """
    Fetch list of models from an Ollama-style /api/tags endpoint.
    Returns a list of normalized model dicts (your schema).
    """
    resp = requests.get(TAGS_ENDPOINT, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    models = payload.get("models", [])

    normalized: List[Dict[str, Any]] = []
    for m in models:
        # Typically m["name"] is the pullable model ref (e.g., "qwen2.5:14b")
        name = m.get("name") or m.get("model") or ""
        if not name:
            # Skip malformed entries
            continue

        label = name
        # You can prettify the name if you want:
        # label = name.replace(":", " ").replace("-", " ").title()

        entry = {
            "id": f"{PROVIDER_KEY}:{name}",     # unique id across providers
            "name": f"{PROVIDER_DISPLAY_NAME} - {label}",
            "provider": PROVIDER_KEY,
            "base_url": SERVER_BASE_URL,
            "api_key_env": EMBEDDED_API_KEY,        # or None; keep secrets server-side ideally
            "model_name": name,                 # the actual model ref to pass to backend
            "is_default": (name == DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME else False
        }
        normalized.append(entry)

    return normalized


def load_llm_provider_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"models": []}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"models": []}


def save_llm_provider_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def merge_models(existing: Dict[str, Any], new_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replace or upsert only models that belong to this PROVIDER_KEY,
    leaving other providers untouched.
    """
    current = existing.get("models", [])

    # Filter out old entries for this provider
    retained = [m for m in current if m.get("provider") != PROVIDER_KEY]

    # Ensure exactly one default for this provider if any entry marked default
    if any(m.get("is_default") for m in new_entries):
        # Make sure only one default within this provider (first wins)
        seen_default = False
        for m in new_entries:
            if m.get("is_default", False):
                if seen_default:
                    m["is_default"] = False
                else:
                    seen_default = True

    merged = retained + new_entries
    return {"models": merged}


def main():
    try:
        new_entries = fetch_models_from_server()
        if not new_entries:
            print("No models discovered from server; skipping update.", file=sys.stderr)
            return
        existing = load_llm_provider_json(LLM_JSON_PATH)
        updated = merge_models(existing, new_entries)
        save_llm_provider_json(LLM_JSON_PATH, updated)
        print(f"Updated {LLM_JSON_PATH} with {len(new_entries)} models from {SERVER_BASE_URL}")

    except requests.HTTPError as e:
        print(f"HTTP error while fetching models: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Network error while fetching models: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(3)
if __name__ == "__main__":
    main()