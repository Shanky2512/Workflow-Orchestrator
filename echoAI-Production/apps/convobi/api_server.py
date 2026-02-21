"""
Convobi API server entry point.

Runs the Convobi (NLP Database Interface) service standalone so the frontend
Convobi tab can use it.

Usage:
  From project root:
    python -m apps.convobi.api_server

  With custom host/port:
    CONVOBI_HOST=0.0.0.0 CONVOBI_PORT=8001 python -m apps.convobi.api_server

  With auto-reload (development):
    CONVOBI_RELOAD=1 python -m apps.convobi.api_server
"""
import os

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("CONVOBI_HOST", "0.0.0.0")
    port = int(os.getenv("CONVOBI_PORT", "8000"))
    reload = os.getenv("CONVOBI_RELOAD", "").lower() in ("1", "true", "yes")

    uvicorn.run(
        "apps.convobi.main:app",
        host=host,
        port=port,
        reload=reload,
    )
