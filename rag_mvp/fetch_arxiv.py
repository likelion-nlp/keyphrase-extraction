"""arxiv API에서 최신 논문 메타데이터를 수집·전처리해 RAG 코퍼스를 만든다.

표준 라이브러리만 사용 (urllib + xml.etree). arxiv 권장 예절: 요청 간 3초 대기.
출력: rag_mvp/data/arxiv_corpus.csv (평면) + .jsonl (authors/categories 리스트 유지)

    python rag_mvp/fetch_arxiv.py --category cs.CL --target 800
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

_ws = re.compile(r"\s+")


def clean(text: str) -> str:
    """개행·중복 공백 정리 (arxiv title/abstract는 줄바꿈이 섞여 있음)."""
    return _ws.sub(" ", (text or "").replace("\n", " ")).strip()


def parse_entry(e) -> dict | None:
    raw_id = (e.findtext("a:id", "", NS) or "").rsplit("/", 1)[-1]  # 2501.00663v1
    m = re.match(r"(.+?)(v\d+)?$", raw_id)
    base_id, version = (m.group(1), m.group(2) or "v1") if m else (raw_id, "v1")
    title = clean(e.findtext("a:title", "", NS))
    abstract = clean(e.findtext("a:summary", "", NS))
    if not title or len(abstract) < 100:      # 너무 짧은 초록 제외
        return None
    authors = [clean(a.findtext("a:name", "", NS)) for a in e.findall("a:author", NS)]
    primary = e.find("arxiv:primary_category", NS)
    primary_cat = primary.get("term") if primary is not None else ""
    cats = [c.get("term") for c in e.findall("a:category", NS) if c.get("term")]
    published = (e.findtext("a:published", "", NS) or "")[:10]   # YYYY-MM-DD
    updated = (e.findtext("a:updated", "", NS) or "")[:10]
    return {
        "arxiv_id": base_id, "version": version,
        "title": title, "abstract": abstract,
        "authors": authors, "primary_category": primary_cat, "categories": cats,
        "published": published, "updated": updated,
        "abs_url": f"https://arxiv.org/abs/{base_id}",
        "pdf_url": f"https://arxiv.org/pdf/{base_id}",
    }


def fetch(category: str, target: int, page: int = 100) -> list[dict]:
    seen, rows = set(), []
    start = 0
    while len(rows) < target:
        q = urllib.parse.urlencode({
            "search_query": f"cat:{category}",
            "start": start, "max_results": min(page, target - len(rows) + 20),
            "sortBy": "submittedDate", "sortOrder": "descending",
        })
        url = f"{API}?{q}"
        for attempt in range(3):
            try:
                xml = urllib.request.urlopen(url, timeout=30).read()
                break
            except Exception as ex:
                print(f"  재시도 {attempt+1} ({ex})"); time.sleep(5)
        else:
            print("  요청 실패 — 중단"); break
        entries = ET.fromstring(xml).findall("a:entry", NS)
        if not entries:
            print("  더 이상 결과 없음"); break
        added = 0
        for e in entries:
            rec = parse_entry(e)
            if rec and rec["arxiv_id"] not in seen:
                seen.add(rec["arxiv_id"]); rows.append(rec); added += 1
        start += len(entries)
        print(f"  수집 {len(rows)}/{target} (이번 페이지 +{added})")
        if len(rows) >= target:
            break
        time.sleep(3)      # arxiv 예절
    return rows[:target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="cs.CL", help="arxiv 분류 (cs.CL=NLP, cs.LG=ML)")
    ap.add_argument("--target", type=int, default=800)
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    print(f"arxiv 수집: {args.category} 최신 {args.target}편")
    rows = fetch(args.category, args.target)
    print(f"\n최종 {len(rows)}편 수집 완료")

    # JSONL (리스트 필드 유지)
    with (DATA / "arxiv_corpus.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # CSV (평면 — authors/categories 는 '; ' 결합)
    with (DATA / "arxiv_corpus.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arxiv_id", "title", "abstract", "authors", "primary_category",
                    "categories", "published", "abs_url", "pdf_url"])
        for r in rows:
            w.writerow([r["arxiv_id"], r["title"], r["abstract"], "; ".join(r["authors"]),
                        r["primary_category"], "; ".join(r["categories"]),
                        r["published"], r["abs_url"], r["pdf_url"]])

    # 요약
    from collections import Counter
    dates = [r["published"] for r in rows if r["published"]]
    cats = Counter(r["primary_category"] for r in rows)
    print(f"저장: data/arxiv_corpus.csv, .jsonl")
    print(f"기간: {min(dates)} ~ {max(dates)}")
    print(f"평균 초록 길이: {sum(len(r['abstract']) for r in rows)//len(rows)}자")
    print(f"주요 분류: {dict(cats.most_common(6))}")
    print("\n샘플 3편:")
    for r in rows[:3]:
        print(f"  [{r['published']}] {r['title'][:70]}")
        print(f"     저자 {len(r['authors'])}명 · {r['primary_category']} · {r['abs_url']}")


if __name__ == "__main__":
    main()
