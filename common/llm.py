"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    # Try to get key from NVIDIA_API_KEY first, then OPENROUTER_API_KEY
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
        raise RuntimeError("Neither NVIDIA_API_KEY nor OPENROUTER_API_KEY is set in .env")
        
    model = os.environ.get("LLM_MODEL", "meta/llama-3.1-70b-instruct")
    base_url = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )
