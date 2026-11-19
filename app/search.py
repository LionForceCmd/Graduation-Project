"""Поиск по фрагментам: нарезка на абзацы, простая русская нормализация, BM25 + покрытие слов вопроса.

Без внешних сервисов и без векторных баз: всё считается в памяти по таблице snippets
(для базы знаний команды в сотни абзацев это быстрее, чем любой сетевой вызов).
"""
import math
import re
from collections import Counter

from . import config

STOPWORDS = set(
    """и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне
    было вот от меня еще нет о из ему теперь когда даже ну ли если уже или ни быть был него до вас
    нибудь опять вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем
    была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
    совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при
    наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три
    эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда
    конечно всю между какие какое каких какую каким какому каком какого сколько нужно это эта такое
    такие какими означает значит называется называют бывают бывает даётся дается времени ответить
    который которая которое которые которого которой каждый каждая каждое свой своя наш наша наши ваш ваша ваши
    the a an is of to in and or for""".split()
)

_ENDINGS = sorted(
    """ами ями ого его ому ему ыми ими ах ях ов ев ей ой ий ый ая яя ое ее ие ые ую юю ом ем ам ям ою ею
    ть ет ут ют ит ат ят ешь ишь ете ите ла ло ли а я у ю о е ы и ь й л""".split(),
    key=len,
    reverse=True,
)
_TOKEN = re.compile(r"[a-zа-я0-9]+")


def stem(tok: str) -> str:
    if tok.isdigit() or not re.search(r"[а-я]", tok):
        return tok
    for refl in ("ся", "сь"):
        if tok.endswith(refl) and len(tok) - 2 >= 4:
            tok = tok[:-2]
            break
    for e in _ENDINGS:
        if tok.endswith(e) and len(tok) - len(e) >= 3:
            return tok[: -len(e)]
    return tok


def stems(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower().replace("ё", "е")):
        if t in STOPWORDS or (len(t) < 3 and not t.isdigit()):
            continue
        out.append(stem(t))
    return out


def norm(text: str) -> str:
    """Нормализация для проверки, что цитата реально есть в контексте."""
    text = re.sub(r"(?m)^\s*[-*•#>]+\s*", "", text)
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def split_snippets(text: str, max_len: int = 700) -> list[str]:
    """Режем по абзацам (пустая строка); слишком длинные абзацы делим по предложениям."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_len:
            out.append(para)
            continue
        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+|\n", para):
            if buf and len(buf) + len(sent) + 1 > max_len:
                out.append(buf.strip())
                buf = ""
            buf += sent + " "
        if buf.strip():
            out.append(buf.strip())
    return out


def sentence_blocks(text: str) -> list[tuple[str, str]]:
    """Список (заголовок раздела, предложение). Пункты списка и абзацы склеиваются из строк,
    маркеры («- ») убираются, заголовки (#) не попадают в ответ, но сохраняются как контекст."""
    out: list[tuple[str, str]] = []
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        heading, items = "", []
        for line in para.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                heading = line.lstrip("# ").strip()
            elif re.match(r"^[-*•]\s*", line):
                items.append(re.sub(r"^[-*•]\s*", "", line))
            elif items:
                items[-1] += " " + line  # перенос строки внутри предложения/абзаца
            else:
                items.append(line)
        for it in items:
            for s in re.split(r"(?<=[.!?])\s+", it):
                s = s.strip()
                if len(s) >= 8:
                    out.append((heading, s))
    return out


def search(conn, question: str, k: int | None = None) -> list[dict]:
    """Возвращает до k фрагментов, достаточно полно покрывающих вопрос. Пусто = «данных нет»."""
    k = k or config.SEARCH_TOP_K
    q_stems = set(stems(question))
    if not q_stems:
        return []
    rows = conn.execute(
        "SELECT s.id, s.document_id, s.snippet_text, d.title FROM snippets s"
        " JOIN documents d ON d.id = s.document_id"
    ).fetchall()
    if not rows:
        return []

    # заголовок документа тоже участвует в поиске: «термин» в вопросе найдёт абзацы из «Словаря терминов»
    docs = [Counter(stems(r["title"] + " " + r["snippet_text"])) for r in rows]
    n = len(docs)
    avg_len = (sum(sum(c.values()) for c in docs) / n) or 1.0
    df = Counter()
    for c in docs:
        for t in c:
            df[t] += 1
    # Слова вопроса, которых в базе нет совсем («даётся», «означает», «токио»), получают вес «среднего»
    # найденного слова: лишняя вода в вопросе не мешает, но если не нашлось большинство слов — отказ.
    seen = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q_stems if df[t]}
    if not seen:
        return []
    mean_w = sum(seen.values()) / len(seen)
    idf = {t: seen.get(t, mean_w) for t in q_stems}
    total_w = sum(idf.values())

    scored = []
    for r, c in zip(rows, docs):
        length = sum(c.values()) or 1
        matched = [t for t in q_stems if t in c]
        if not matched:
            continue
        coverage = sum(idf[t] for t in matched) / total_w
        bm25 = sum(
            idf[t] * c[t] * 2.5 / (c[t] + 1.5 * (0.25 + 0.75 * length / avg_len)) for t in matched
        )
        scored.append(
            {
                "snippet_id": r["id"],
                "document_id": r["document_id"],
                "title": r["title"],
                "text": r["snippet_text"],
                "coverage": round(coverage, 3),
                "score": round(bm25, 3),
            }
        )
    good = [s for s in scored if s["coverage"] >= config.MIN_COVERAGE]
    good.sort(key=lambda s: (s["coverage"], s["score"]), reverse=True)
    return good[:k]
