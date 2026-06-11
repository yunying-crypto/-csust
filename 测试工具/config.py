"""
Configuration management - loads API keys and model settings from .env file.
The .env file should be placed at the project root directory.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float = 120.0
    max_retries: int = 3


@dataclass
class EvalConfig:
    intern_s1: LLMConfig
    deepseek: LLMConfig


def _get_project_root() -> str:
    """Return the project root directory (parent of 测试工具/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(dotenv_path: str = None) -> EvalConfig:
    """Load .env from project root, then validate and return config."""
    if dotenv_path is None:
        dotenv_path = os.path.join(_get_project_root(), ".env")
    load_dotenv(dotenv_path, override=False)

    cfg = get_config()

    # Validate required API keys
    missing = []
    if not cfg.intern_s1.api_key or cfg.intern_s1.api_key.startswith("your_"):
        missing.append("INTERN_S1_API_KEY")
    if not cfg.deepseek.api_key or cfg.deepseek.api_key.startswith("your_"):
        missing.append("DEEPSEEK_API_KEY")
    if missing:
        raise ValueError(
            f"缺少 API Key: {', '.join(missing)}。"
            f"请复制 .env.example 为项目根目录的 .env 并填入正确的 Key。"
        )

    return cfg


def get_config() -> EvalConfig:
    return EvalConfig(
        intern_s1=LLMConfig(
            api_key=os.getenv("INTERN_S1_API_KEY", ""),
            base_url=os.getenv("INTERN_S1_BASE_URL", "https://internlm-chat.intern-ai.org.cn/puyu/api/v1"),
            model=os.getenv("INTERN_S1_MODEL", "internlm3-latest"),
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        ),
        deepseek=LLMConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        ),
    )
