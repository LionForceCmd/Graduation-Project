"""Автотесты API в режиме mock (без Ollama и без сети): pytest -q"""
import json
import os
import tempfile
from pathlib import Path

os.environ["LLM_PROVIDER"] = "mock"
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.sqlite3")

import pytest
from fastapi.testclient import TestClient

from app.main import app

DATA = Path(__file__).resolve().parent.parent / "tests_data"


def jsonl(name):
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        titles = {}
        for d in jsonl("kb_documents.jsonl"):
            r = c.post("/kb/documents", json=d)
            assert r.status_code == 200 and r.json()["status"] == "ok"
            titles[r.json()["document_id"]] = d["title"]
        c.titles = titles
        yield c


def test_documents_listed(client):
    docs = client.get("/kb/documents").json()
    assert len(docs) == 5
    assert {"id", "title", "created_at"} <= set(docs[0])


@pytest.mark.parametrize("q", jsonl("kb_questions.jsonl"), ids=lambda q: q["question"][:40])
def test_questions(client, q):
    r = client.post("/kb/ask", json={"question": q["question"]})
    assert r.status_code == 200
    body = r.json()
    assert body["needs_review"] == q["expected_needs_review"]
    # правила из задания: нет источников -> needs_review; needs_review=false -> источники есть
    assert (not body["sources"]) <= body["needs_review"]
    if q["expected_needs_review"]:
        assert body["sources"] == []
        assert "недостаточно" in body["answer"]
    else:
        assert any(client.titles[s["document_id"]] == q["expected_document"] for s in body["sources"])


@pytest.mark.parametrize("payload", [{"title": "", "text": "x"}, {"title": "x", "text": "  "}, {"title": "x"}])
def test_bad_document_is_422_and_audited(client, payload):
    before = len(client.get("/audit?status=error").json())
    assert client.post("/kb/documents", json=payload).status_code == 422
    assert len(client.get("/audit?status=error").json()) == before + 1


def test_empty_question_422(client):
    assert client.post("/kb/ask", json={"question": "   "}).status_code == 422


def test_unknown_document_404(client):
    assert client.get("/kb/documents/doc_nope").status_code == 404


def test_ai_endpoint_strict_json(client):
    r = client.post("/ai/answer_with_sources", json={
        "question": "Во сколько стендап?", "context": "Стендап проходит в 10:30 по Москве."}).json()
    assert set(r) == {"answer", "sources", "confidence", "needs_review"}
    assert r["sources"] and r["sources"][0]["quote"] in "Стендап проходит в 10:30 по Москве."
    empty = client.post("/ai/answer_with_sources", json={"question": "Что-то?", "context": ""}).json()
    assert empty["needs_review"] is True and empty["sources"] == []


def test_fabricated_quote_is_dropped():
    from app import llm
    res, reasons = llm.enforce(
        {"answer": "Да", "sources": [{"quote": "этого нет в тексте"}], "confidence": "high", "needs_review": False},
        "Совсем другой контекст.")
    assert res["needs_review"] is True and res["sources"] == [] and "NO_SOURCES" in reasons


def test_audit_and_export(client):
    audit = client.get("/audit?limit=1000").json()
    assert len(audit) >= 10
    assert any(a["error"] == "NO_SOURCES" for a in audit)
    assert client.get("/export/audit?format=csv").text.startswith("﻿")
    assert client.get("/export/qa?format=json").json()
