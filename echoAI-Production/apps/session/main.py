
from fastapi import FastAPI
from . import container  # register providers
from .routes import router
app = FastAPI(title='Session Service')
app.include_router(router)
