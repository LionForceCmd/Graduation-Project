"""ИИ-операция «ответ с источниками» (строгий JSON) и контроль качества результата.

Два режима (LLM_PROVIDER):
  mock   — без языковой модели: ответ собирается из самых подходящих предложений контекста;
  ollama — ответ формирует локальная модель; если Ollama недоступна, используется mock.

Что бы ни вернул провайдер, результат проходит одни и те же проверки (enforce):
  * структура JSON (pydantic) — иначе SCHEMA_MISMATCH / INVALID_JSON;
  * каждая цитата должна дословно присутствовать в контексте — иначе она отбрасывается;
  * нет ни одной проверенной цитаты  -> needs_review=true (NO_SOURCES);
  * confidence="low"                  -> needs_review=true (LOW_CONFIDENCE).
"""
import json
import re
import urllib.error
import urllib.request
from typing import Literal

from pydantic import BaseModel, ValidationError

from . import config, search

REFUSAL = "Данных в базе знаний недостаточно, чтобы ответить на этот вопрос. Уточните вопрос или добавьте документ."


class Source(BaseModel):
    quote: str


class AnswerOut(BaseModel):
    answer: str
    sources: list[Source]
    confidence: Literal["high", "medium", "low"]
    needs_review: bool


class LLMUnavailable(Exception):
    pass


# ---------------------------------------------------------------- mock

def mock_answer(question: str, context: str) -> dict:
    """Режим без модели: выбираем предложения контекста, лучше всего совпадающие с вопросом.

    Заголовок раздела тоже считается (пункты списка «Статусы задачи» относятся к вопросу про статусы).
    Уверенность — доля слов вопроса, которые вообще нашлись в контексте.
    """
    qs = set(search.stems(question))
    ctx_stems = set(search.stems(context))
    cov = len(qs & ctx_stems) / len(qs) if qs else 0.0
    scored = []
    for idx, (heading, sent) in enumerate(search.sentence_blocks(context)):
        m = qs & (set(search.stems(sent)) | set(search.stems(heading)))
        if m:
            scored.append((len(m), idx, sent))
    if not scored or cov == 0:
        return {"answer": REFUSAL, "sources": [], "confidence": "low", "needs_review": True}
    best = max(sc[0] for sc in scored)
    chosen = sorted([sc for sc in scored if sc[0] == best], key=lambda sc: sc[1])[:5]
    conf = "high" if cov >= 0.75 else "medium" if cov >= 0.5 else "low"
    return {
        "answer": " ".join(sc[2] for sc in chosen),
        "sources": [{"quote": sc[2]} for sc in chosen],
        "confidence": conf,
        "needs_review": conf == "low",
    }


# ---------------------------------------------------------------- ollama

SYSTEM_PROMPT = (
    "Ты помощник по базе знаний команды. Отвечай на вопрос ТОЛЬКО по переданному контексту. "
    "Ничего не выдумывай и не используй внешние знания. "
    "Верни строго JSON: {\"answer\": строка, \"sources\": [{\"quote\": строка}], "
    "\"confidence\": \"high\"|\"medium\"|\"low\", \"needs_review\": true|false}. "
    "quote — ДОСЛОВНАЯ цитата из контекста, подтверждающая ответ (1–2 цитаты). "
    "Если в контексте нет ответа: answer=\"Данных недостаточно\", sources=[], confidence=\"low\", needs_review=true. "
    "Отвечай по-русски, кратко."
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {"type": "object", "properties": {"quote": {"type": "string"}}, "required": ["quote"]},
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "needs_review": {"type": "boolean"},
    },
    "required": ["answer", "sources", "confidence", "needs_review"],
}


def _post_ollama(payload: dict) -> dict:
    req = urllib.request.Request(
        config.OLLAMA_URL + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_raw(question: str, context: str) -> str:
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "format": JSON_SCHEMA,
        # num_predict обязателен: без лимита модель может бесконечно генерировать и «повесить» запрос
        "options": {"temperature": config.LLM_TEMPERATURE, "num_ctx": 4096, "num_predict": 400},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
        ],
    }
    try:
        try:
            data = _post_ollama(payload)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            payload.pop("think")  # модель без режима «thinking» не принимает параметр
            data = _post_ollama(payload)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise LLMUnavailable(str(getattr(e, "reason", e))) from e
    return (data.get("message") or {}).get("content", "")


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError("model did not return JSON")


# ---------------------------------------------------------------- контроль качества

def enforce(parsed: dict, context: str) -> tuple[dict, list[str]]:
    """Проверяет структуру и цитаты. Возвращает (строгий результат, список причин ручной проверки)."""
    out = AnswerOut.model_validate(parsed)  # ValidationError -> SCHEMA_MISMATCH у вызывающего
    reasons: list[str] = []
    ctx = search.norm(context)
    good, dropped = [], 0
    for s in out.sources:
        q = search.norm(s.quote)
        if q and q in ctx:
            if all(q != search.norm(g["quote"]) for g in good):
                good.append({"quote": s.quote.strip()})
        else:
            dropped += 1
    needs = out.needs_review
    if out.confidence == "low":
        reasons.append("LOW_CONFIDENCE")
    if not good:
        reasons.append("NO_SOURCES")
    elif dropped:
        reasons.append("QUOTE_NOT_IN_CONTEXT")  # часть цитат выдумана — отбросили, но фиксируем
    needs = needs or "LOW_CONFIDENCE" in reasons or "NO_SOURCES" in reasons
    answer = out.answer.strip()
    if not good or not answer:
        answer = REFUSAL  # без проверенных цитат не показываем «уверенный» текст модели
        needs = True
        good = []
    return (
        {"answer": answer, "sources": good, "confidence": out.confidence, "needs_review": needs},
        reasons,
    )


def answer_with_context(question: str, context: str) -> dict:
    """Главная ИИ-операция. Всегда возвращает безопасный результат, исключений наружу не бросает.

    Ключи результата: answer, sources[{quote}], confidence, needs_review (строгий JSON)
    плюс служебные: error (причина для аудита), raw (сырой ответ модели), provider.
    """
    provider, raw, errors = config.LLM_PROVIDER, "", []
    force_review = False

    parsed = None
    if provider == "ollama":
        try:
            raw = ollama_raw(question, context)
            parsed = parse_json(raw)
        except LLMUnavailable as e:
            errors.append(f"LLM_UNAVAILABLE ({e})")
            provider = "mock (резерв: Ollama недоступна)"
        except ValueError:
            errors.append("INVALID_JSON")
            force_review = True
            provider = "mock (резерв: модель вернула не JSON)"
    else:
        provider = "mock"

    result = None
    if parsed is not None:
        try:
            result, reasons = enforce(parsed, context)
            errors += reasons
        except ValidationError:
            errors.append("SCHEMA_MISMATCH")
            force_review = True
            provider = "mock (резерв: JSON не по схеме)"
    if result is None:
        result, reasons = enforce(mock_answer(question, context), context)
        errors += reasons
    if force_review:
        result["needs_review"] = True

    # приоритет причин: то, что требует проверки, — первым
    review_codes = [e for e in errors if not e.startswith("LLM_UNAVAILABLE")]
    error = ", ".join(review_codes + [e for e in errors if e.startswith("LLM_UNAVAILABLE")]) or None
    result.update(error=error, raw=raw, provider=provider)
    return result
