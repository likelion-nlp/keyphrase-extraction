"""색인 파이프라인 — arxiv 코퍼스 → P7 키프레이즈 → bge 임베딩 → Chroma 저장.

    python rag_mvp/ingest.py                 # 전체
    python rag_mvp/ingest.py --limit 30      # 테스트(앞 30편)
    python rag_mvp/ingest.py --reset         # 색인 비우고 새로
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAG = Path(__file__).resolve().parent
sys.path.insert(0, str(RAG.parent))

from rag_mvp.core.index import CHROMA_DIR, VectorIndex
from rag_mvp.core.keyphrases import KeyphraseGenerator

CORPUS = RAG / "data" / "arxiv_corpus.jsonl"


def load_arxiv(limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    rows = rows[:limit] if limit else rows
    return [{"doc_id": r["arxiv_id"], "title": r["title"], "abstract": r["abstract"],
             "meta": {"authors": "; ".join(r.get("authors", []))[:300],
                      "published": r.get("published", ""),
                      "primary_category": r.get("primary_category", ""),
                      "abs_url": r.get("abs_url", "")}} for r in rows]


def load_kp20k(limit: int | None) -> list[dict]:
    """KP20k test 부분집합 — gold·prmu를 메타에 저장(평가용)."""
    from src.data import load_kp20k as _load
    n = limit or 2000
    rows = list(_load(["test"], subset_sizes={"test": n}, seed=42)["test"])[:n]
    out = []
    for r in rows:
        out.append({"doc_id": str(r["id"]), "title": r["title"], "abstract": r["abstract"],
                    "meta": {"authors": "", "published": "", "primary_category": "kp20k",
                             "abs_url": "",
                             "gold": "; ".join(r["keyphrases"])[:500],
                             "prmu": "".join(r["prmu"])}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["arxiv", "kp20k"], default="arxiv")
    ap.add_argument("--collection", default=None, help="기본: arxiv→papers, kp20k→kp20k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32, help="키프레이즈 생성 배치(문서 수)")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    collection = args.collection or ("papers" if args.source == "arxiv" else "kp20k")

    rows = load_kp20k(args.limit) if args.source == "kp20k" else load_arxiv(args.limit)
    print(f"[{args.source}] 코퍼스 {len(rows)}편 로드 → 컬렉션 '{collection}' | 모델 로딩...", flush=True)
    t0 = time.time()
    kpgen = KeyphraseGenerator(device=args.device)
    index = VectorIndex(device=args.device, collection=collection)
    if args.reset:
        index.reset(); print("기존 색인 비움 (컬렉션 재생성)")
    print(f"[{time.time()-t0:.0f}s] 모델 로드 완료. 색인 시작 ({CHROMA_DIR})", flush=True)

    total_vec = 0
    for s in range(0, len(rows), args.batch):
        chunk = rows[s : s + args.batch]
        kps = kpgen.generate([{"title": r["title"], "abstract": r["abstract"]} for r in chunk])
        docs = [{**r, "keyphrases": kp} for r, kp in zip(chunk, kps)]
        total_vec += index.add_documents(docs)
        print(f"[{time.time()-t0:.0f}s] {s+len(chunk)}/{len(rows)}편 색인 "
              f"(누적 벡터 {total_vec}, 컬렉션 총 {index.count()})", flush=True)

    print(f"\n완료: {len(rows)}편 → 컬렉션 '{collection}' {index.count()} 벡터")


if __name__ == "__main__":
    main()
