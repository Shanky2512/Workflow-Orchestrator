"""
Main entry point for the API Connector Framework.

Supports two modes:
1. FastAPI server mode (--api)
2. CLI mode (default)
"""

import argparse
from typing import NoReturn

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import router


# =========================
# FastAPI APP (TOP LEVEL)
# =========================
app = FastAPI(
    title="API Connector Framework",
    description="Production-grade API connector generation and management platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "API Connector Framework",
        "version": "1.0.0",
        "docs": "/docs",
    }


# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}


# =========================
# SERVER RUNNER
# =========================
def run_api_server(host: str = "0.0.0.0", port: int = 8000) -> NoReturn:
    print(
        f"""
╔════════════════════════════════════════════════════════════════╗
║           API Connector Framework - FastAPI Server            ║
╚════════════════════════════════════════════════════════════════╝

Server running at: http://{host}:{port}
Swagger Docs:       http://{host}:{port}/docs
ReDoc:              http://{host}:{port}/redoc

Press CTRL+C to stop the server
"""
    )

    uvicorn.run("main:app", host=host, port=port, reload=True)


# =========================
# CLI MODE
# =========================
def run_cli() -> None:
    from cli import main as cli_main
    cli_main()


# =========================
# ENTRYPOINT
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="API Connector Framework",
    )

    parser.add_argument(
        "--api",
        action="store_true",
        help="Run FastAPI server",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind API server",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind API server",
    )

    args = parser.parse_args()

    if args.api:
        run_api_server(host=args.host, port=args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
