"""Configuration is local to this standalone project; never log credentials."""
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_TOPIC = "是否应该实行四天工作制"
ROOT = Path(__file__).resolve().parent


def load_config():
    load_dotenv(ROOT / ".env")
    return {"model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "timeout": 60, "max_retries": 1}


def create_llm(mock=False):
    settings = load_config()
    if mock:
        from tools.mock_llm import MockLLM
        return MockLLM()
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise ValueError("缺少 DEEPSEEK_API_KEY，请在项目 .env 或环境变量中配置。")
    from langchain_deepseek import ChatDeepSeek
    return ChatDeepSeek(**settings, temperature=0.5, max_tokens=4096)
