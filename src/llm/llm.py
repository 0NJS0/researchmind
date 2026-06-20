from langchain_openai import ChatOpenAI

from src.config import LLM_URL, LLM_MODEL, LLM_API_KEY


llm = ChatOpenAI(
    base_url=LLM_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    temperature=0.1,
)

def check_llm():
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_URL, api_key=LLM_API_KEY, timeout=5)
        client.models.list()
        return True, "connected"
    except Exception:
        return False, "LLM endpoint unreachable"

