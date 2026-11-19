"""Система знаний команды: FastAPI + SQLite + поиск + ИИ-ответ с источниками + аудит."""
import csv
import io
import json
import time
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from . import config, db, llm, search

STATIC = config.BASE_DIR / "app" / "static"


@asynccontextmanager
async def lifespan(_app):
    db.init()
    yield


app = FastAPI(title="Team Knowledge Base", version="1.0", lifespan=lifespan)


# ---------------------------------------------------------------- аудит

class Audited:
    """Контекст: любая команда сервиса пишет запись в audit_runs (успех, ошибка, причина проверки)."""

    def __init__(self, action: str, inp=None):
        self.action, self.inp = action, inp
        self.output = None
        self.error = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, et, ev, tb):
        ms = int((time.perf_counter() - self.t0) * 1000)
        status = "ok"
        if et is not None:
            status = "error"
            if issubclass(et, HTTPException):
                self.error = f"HTTP_{ev.status_code}: {ev.detail}"
                self.output = {"status_code": ev.status_code, "detail": ev.detail}
            else:
                self.error = f"{et.__name__}: {ev}"
                self.output = {"status_code": 500, "detail": "internal error"}
        db.audit(self.action, self.inp, self.output, status, self.error, ms)
        return False


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    try:
        body = (await request.body()).decode("utf-8", "replace")
    except Exception:
        body = ""
    detail = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()]
    db.audit(f"{request.method} {request.url.path}", body, {"status_code": 422, "detail": detail},
             "error", "VALIDATION: " + "; ".join(detail), 0)
    return JSONResponse({"detail": detail}, status_code=422)


# ---------------------------------------------------------------- модели

class DocIn(BaseModel):
    title: str
    text: str


class AskIn(BaseModel):
    question: str


class AnswerIn(BaseModel):
    question: str
    context: str


# ---------------------------------------------------------------- 1. документы

@app.post("/kb/documents")
def add_document(body: DocIn):
    with Audited("POST /kb/documents", {"title": body.title, "text_len": len(body.text)}) as a:
        title, text = body.title.strip(), body.text.strip()
        if not title or not text:
            raise HTTPException(422, "title и text не должны быть пустыми")
        if len(title) > config.MAX_TITLE:
            raise HTTPException(422, f"title длиннее {config.MAX_TITLE} символов")
        if len(text) > config.MAX_TEXT:
            raise HTTPException(422, f"text длиннее {config.MAX_TEXT} символов")
        doc_id, ts = db.new_id("doc"), db.now()
        parts = search.split_snippets(text)
        with db.connect() as c:
            c.execute("INSERT INTO documents(id, created_at, title, text) VALUES (?,?,?,?)",
                      (doc_id, ts, title, text))
            c.executemany(
                "INSERT INTO snippets(id, created_at, document_id, snippet_text) VALUES (?,?,?,?)",
                [(db.new_id("snp"), ts, doc_id, p) for p in parts],
            )
        a.output = {"status": "ok", "document_id": doc_id, "snippets": len(parts)}
        return {"status": "ok", "document_id": doc_id}


@app.get("/kb/documents")
def list_documents():
    with Audited("GET /kb/documents") as a:
        with db.connect() as c:
            rows = c.execute(
                "SELECT d.id, d.title, d.created_at, (SELECT COUNT(*) FROM snippets s WHERE s.document_id = d.id)"
                " AS snippets FROM documents d ORDER BY d.created_at DESC, d.rowid DESC"
            ).fetchall()
        res = [dict(r) for r in rows]
        a.output = {"count": len(res)}
        return res


@app.get("/kb/documents/{doc_id}")
def get_document(doc_id: str):
    """Карточка документа для панели (не пишется в аудит: это просмотр, а не команда)."""
    with db.connect() as c:
        r = c.execute("SELECT id, title, text, created_at FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not r:
        raise HTTPException(404, "документ не найден")
    return dict(r)


# ---------------------------------------------------------------- 2. вопросы

def _sources_with_docs(result_sources: list[dict], hits: list[dict]) -> list[dict]:
    out = []
    for s in result_sources:
        q = search.norm(s["quote"])
        for h in hits:
            if q in search.norm(h["text"]):
                out.append({"document_id": h["document_id"], "title": h["title"], "quote": s["quote"]})
                break
    return out


@app.post("/kb/ask")
def ask(body: AskIn):
    with Audited("POST /kb/ask", {"question": body.question}) as a:
        q = body.question.strip()
        if not q:
            raise HTTPException(422, "question не должен быть пустым")
        if len(q) > config.MAX_QUESTION:
            raise HTTPException(422, f"question длиннее {config.MAX_QUESTION} символов")
        t0 = time.perf_counter()
        with db.connect() as c:
            hits = search.search(c, q)
        if hits:
            ctx = "\n\n".join(h["text"] for h in hits)
            res = llm.answer_with_context(q, ctx)
            sources = _sources_with_docs(res["sources"], hits)
            if not sources:  # правило задания: нет источников -> ручная проверка
                res.update(answer=llm.REFUSAL, needs_review=True)
                res["error"] = ", ".join(filter(None, ["NO_SOURCES", res.get("error")]))
        else:
            res = {"answer": llm.REFUSAL, "confidence": "low", "needs_review": True,
                   "error": "NO_SOURCES", "raw": "", "provider": "retrieval (модель не вызывалась)"}
            sources = []
        res["needs_review"] = bool(res["needs_review"] or not sources)
        ms = int((time.perf_counter() - t0) * 1000)
        run_id = db.new_id("qa")
        with db.connect() as c:
            c.execute(
                "INSERT INTO qa_runs(id, created_at, question, answer, sources_json, needs_review, error,"
                " confidence, provider, raw_output, duration_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, db.now(), q, res["answer"], json.dumps(sources, ensure_ascii=False),
                 int(res["needs_review"]), res["error"], res["confidence"], res["provider"], res["raw"], ms),
            )
        resp = {
            "answer": res["answer"],
            "sources": [{"document_id": s["document_id"], "quote": s["quote"]} for s in sources],
            "needs_review": res["needs_review"],
            "run_id": run_id,
            "confidence": res["confidence"],
            "reason": res["error"],
        }
        a.output, a.error = resp, res["error"]
        return resp


@app.post("/ai/answer_with_sources")
def ai_answer(body: AnswerIn):
    """ИИ-операция отдельно (для тестов качества): строгий JSON ровно из четырёх полей."""
    with Audited("POST /ai/answer_with_sources", {"question": body.question, "context_len": len(body.context)}) as a:
        q = body.question.strip()
        if not q:
            raise HTTPException(422, "question не должен быть пустым")
        if len(body.context) > config.MAX_TEXT:
            raise HTTPException(422, f"context длиннее {config.MAX_TEXT} символов")
        res = llm.answer_with_context(q, body.context)
        out = {"answer": res["answer"], "sources": res["sources"],
               "confidence": res["confidence"], "needs_review": res["needs_review"]}
        a.output, a.error = out, res["error"]
        return out


# ---------------------------------------------------------------- история и аудит (для панели)

def _qa_row(r) -> dict:
    d = dict(r)
    d["sources"] = json.loads(d.pop("sources_json"))
    d["needs_review"] = bool(d["needs_review"])
    return d


@app.get("/qa")
def list_qa(needs_review: bool | None = None, limit: int = Query(100, ge=1, le=1000)):
    sql, args = "SELECT * FROM qa_runs", []
    if needs_review is not None:
        sql += " WHERE needs_review=?"
        args.append(int(needs_review))
    sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
    args.append(limit)
    with db.connect() as c:
        return [_qa_row(r) for r in c.execute(sql, args).fetchall()]


@app.get("/qa/{run_id}")
def get_qa(run_id: str):
    with db.connect() as c:
        r = c.execute("SELECT * FROM qa_runs WHERE id=?", (run_id,)).fetchone()
    if not r:
        raise HTTPException(404, "запись не найдена")
    return _qa_row(r)


@app.get("/audit")
def list_audit(status: str | None = None, limit: int = Query(100, ge=1, le=1000)):
    sql, args = "SELECT * FROM audit_runs", []
    if status:
        sql += " WHERE status=?"
        args.append(status)
    sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
    args.append(limit)
    with db.connect() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def _export(rows: list[dict], fmt: str, name: str) -> Response:
    if fmt == "csv":
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                            for k, v in r.items()})
        return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
    return Response(json.dumps(rows, ensure_ascii=False, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{name}.json"'})


@app.get("/export/qa")
def export_qa(format: str = "json", needs_review: bool | None = None):
    return _export(list_qa(needs_review=needs_review, limit=1000), format, "qa_history")


@app.get("/export/audit")
def export_audit(format: str = "json"):
    return _export(list_audit(limit=1000), format, "audit_runs")


@app.get("/health")
def health():
    info = {"status": "ok", "provider": config.LLM_PROVIDER, "model": None, "ollama": None}
    if config.LLM_PROVIDER == "ollama":
        info["model"] = config.OLLAMA_MODEL
        try:
            with urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=2) as r:
                names = [m["name"] for m in json.loads(r.read())["models"]]
            info["ollama"] = "ok" if config.OLLAMA_MODEL in names else "model_not_found"
        except Exception:
            info["ollama"] = "unreachable"
    return info


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
