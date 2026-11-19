"""Загружает тестовые документы и прогоняет вопросы через работающий сервис.

    python scripts/eval.py                       # http://localhost:4000
    python scripts/eval.py --url http://host:4000

Печатает таблицу «вопрос -> ожидание -> факт» и возвращает код 1, если хоть одно ожидание не совпало.
Использует только стандартную библиотеку.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "tests_data"


def call(base, method, path, body=None):
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def read_jsonl(name):
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    sys.stdout.reconfigure(errors="replace")  # не падаем на символах, которых нет в кодировке консоли
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4000")
    base = ap.parse_args().url.rstrip("/")

    try:
        _, health = call(base, "GET", "/health")
    except OSError as e:
        sys.exit(f"Сервис недоступен по адресу {base}: {e}")
    print(f"Режим: {health['provider']}" + (f" / {health['model']} (Ollama: {health['ollama']})" if health["model"] else ""))

    _, existing = call(base, "GET", "/kb/documents")
    titles = {d["title"]: d["id"] for d in existing}
    for d in read_jsonl("kb_documents.jsonl"):
        if d["title"] in titles:
            continue
        code, res = call(base, "POST", "/kb/documents", d)
        titles[d["title"]] = res.get("document_id")
        print(f"  + документ «{d['title']}» -> {code}")
    id_to_title = {v: k for k, v in titles.items()}

    # проверка валидации: пустой документ должен вернуть 422 и попасть в аудит
    code, _ = call(base, "POST", "/kb/documents", {"title": "", "text": ""})
    print(f"  проверка валидации (пустой документ) -> {code} {'OK' if code == 422 else 'ОШИБКА'}")

    bad = 0 if code == 422 else 1

    # ИИ-операция отдельно: строгий JSON из четырёх полей
    code, res = call(base, "POST", "/ai/answer_with_sources",
                     {"question": "Во сколько стендап?", "context": "Стендап проходит в 10:30 по Москве."})
    ok_ai = code == 200 and set(res) == {"answer", "sources", "confidence", "needs_review"}
    print(f"  проверка /ai/answer_with_sources (строгий JSON) -> {code} {'OK' if ok_ai else 'ОШИБКА'}")
    bad += 0 if ok_ai else 1
    print(f"\n{'№':>2}  {'ожид.':<6} {'факт':<6} {'итог':<5} вопрос")
    for i, q in enumerate(read_jsonl("kb_questions.jsonl"), 1):
        code, res = call(base, "POST", "/kb/ask", {"question": q["question"]})
        ok = code == 200 and res["needs_review"] == q["expected_needs_review"]
        if ok and not q["expected_needs_review"]:
            ok = bool(res["sources"])
            exp = q.get("expected_document")
            if ok and exp:
                ok = any(id_to_title.get(s["document_id"]) == exp for s in res["sources"])
        if ok and q["expected_needs_review"]:
            ok = not res["sources"]
        bad += 0 if ok else 1
        print(f"{i:>2}  {str(q['expected_needs_review']):<6} {str(res.get('needs_review')):<6} {'OK' if ok else 'FAIL':<5} {q['question']}")
        print(f"      -> {res.get('answer', res)}")
        for s in res.get("sources", []):
            print(f"         [{id_to_title.get(s['document_id'], s['document_id'])}] «{s['quote']}»")
        if res.get("reason"):
            print(f"         причина: {res['reason']}")
    # доказательство: что записано в audit_runs
    _, audit = call(base, "GET", "/audit?limit=1000")
    by_action = {}
    for a in audit:
        k = (a["action"], a["status"])
        by_action[k] = by_action.get(k, 0) + 1
    print(f"\nАудит (audit_runs): всего записей {len(audit)}")
    for (action, status), n in sorted(by_action.items()):
        print(f"  {action:<32} {status:<6} {n}")
    reasons = sorted({a["error"].split(" ")[0].rstrip(",") for a in audit if a["error"]})
    print("  причины в поле error:", ", ".join(reasons) or "-")
    print("\nГотово:", "всё совпало с ожиданиями" if not bad else f"расхождений: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
