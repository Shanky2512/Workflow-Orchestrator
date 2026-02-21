
from fastapi import FastAPI
from . import container
from .routes import router
app = FastAPI(title='Chat Service')
app.include_router(router)
