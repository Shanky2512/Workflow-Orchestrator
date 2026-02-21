import io
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import AzureChatOpenAI

from .stream_handler import TokenCollector

load_dotenv()


class CSVQueryAgent:
    def __init__(self, stream: bool = False):
        self.collector = TokenCollector() if stream else None

        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
            streaming=stream,
            callbacks=[self.collector] if stream else None,
        )

    def create_agent(self, raw_bytes: bytes, file_name: str):
        if file_name.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw_bytes))
        else:
            df = pd.read_excel(io.BytesIO(raw_bytes))

        return create_pandas_dataframe_agent(
            self.llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            include_df_in_prompt=True,
            prefix="""
You are a data analyst working with a pandas dataframe called df.
Run Python code to answer questions.

IMPORTANT:
- No Python code in final answer
- Plain language only
- No technical jargon
""",
        )
