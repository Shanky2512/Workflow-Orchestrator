from typing import Iterable

from langchain_openai import ChatOpenAI
# For Azure deployment - uncomment the line below
# from langchain_openai import AzureChatOpenAI

from app.core.config import get_settings


def summarize_documents(docs: Iterable, max_chars: int = 8000) -> str:
    settings = get_settings()

    parts = []
    for doc in docs:
        content = getattr(doc, "page_content", None)
        if content is None:
            content = str(doc)
        parts.append(content)

    merged = "\n\n".join(parts)
    if len(merged) > max_chars:
        merged = merged[:max_chars]

    if not settings.OPENROUTER_API_KEY:
        return merged

    # For Azure deployment - uncomment this block
    # llm = AzureChatOpenAI(
    #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     temperature=0,
    #     max_tokens=800,
    # )

    # For local/OpenRouter - comment this when deploying to Azure
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=settings.OPENROUTER_API_KEY,
        model=settings.DEFAULT_LLM_MODEL,
        temperature=0,
        max_tokens=800,
    )

    prompt = "Summarize the following content clearly and concisely:\n" + merged
    return llm.invoke(prompt).content
