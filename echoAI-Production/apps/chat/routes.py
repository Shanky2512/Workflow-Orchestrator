from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.services import ChatOrchestrator
from echolib.types import Session
from echolib.utils import new_id

router = APIRouter(prefix='/chat', tags=['ChatApi'])

def orch() -> ChatOrchestrator:
    return container.resolve('chat.orchestrator')

@router.post('/chat')
async def chat(message: dict):
    sess = Session(id=new_id('sess_'), user_id='usr_dev', data={})
    return orch().orchestrate(message, sess)

@router.get('/stream')
async def stream(prompt: str):
    # use LLM stream via orchestrator's LLM (not exposed directly here)
    tokens = []
    orch().llm.stream(prompt, lambda t: tokens.append(t))
    return {'tokens': tokens}
