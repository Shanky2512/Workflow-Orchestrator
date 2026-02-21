import logging
from fastapi import FastAPI
from . import container
from .routes import router

# Suppress noisy LiteLLM proxy logging (apscheduler import warnings)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

app = FastAPI(title='Workflow Service')
app.include_router(router)
