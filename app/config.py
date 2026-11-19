"""Настройки сервиса. Всё читается из переменных окружения (или из файла .env)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    p = BASE_DIR / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# mock   — ответ собирается из найденных цитат, без языковой модели (офлайн, мгновенно)
# ollama — ответ формирует локальная модель через Ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "data" / "kb.sqlite3")))

# Поиск: сколько фрагментов отдаём в ответ и какая доля "веса" слов вопроса
# должна найтись во фрагменте, чтобы считать его подходящим.
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "3"))
MIN_COVERAGE = float(os.getenv("MIN_COVERAGE", "0.45"))

MAX_TITLE = 200
MAX_TEXT = 50_000
MAX_QUESTION = 500
