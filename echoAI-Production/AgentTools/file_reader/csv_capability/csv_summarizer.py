import io
import os

import pandas as pd
from langchain_openai import AzureChatOpenAI


class CSVSummarizer:
    def summarize(self, raw_bytes: bytes, file_name: str, stream: bool = False):
        if file_name.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw_bytes))
        else:
            df = pd.read_excel(io.BytesIO(raw_bytes))

        stats = {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict(),
        }

        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
            streaming=stream,
        )

        prompt = f"Explain this dataset clearly:\n{stats}"
        return llm.invoke(prompt).content
