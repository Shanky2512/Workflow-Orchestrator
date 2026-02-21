from langchain.callbacks.base import BaseCallbackHandler


class TokenCollector(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)

    def text(self) -> str:
        return "".join(self.tokens)
