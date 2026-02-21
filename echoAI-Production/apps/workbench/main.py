"""
Workbench Mode -- Standalone App Entry Point

Can be run independently for development/testing, or mounted
via the gateway for production use.
"""

from fastapi import FastAPI
from . import container  # noqa: F401  -- triggers DI registration
from .routes import router

app = FastAPI(title='Workbench Service')
app.include_router(router)
