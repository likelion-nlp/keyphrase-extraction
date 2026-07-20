"""검색 CLI (테스트/디버그용). backend는 core.index.VectorIndex를 직접 쓴다.

    python rag_mvp/search_cli.py "on-device mobile agent"
    python rag_mvp/search_cli.py "query" --mode plain   # 본문만 색인 비교
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: F401  chromadb보다 먼저
from rag_mvp.core.index import VectorIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--mode", choices=["kp", "plain"], default="kp")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    idx = VectorIndex(device=args.device)
    q = " ".join(args.query)
    print(f"색인 벡터 {idx.count()} | 질의: {q} | mode={args.mode}\n")
    for r in idx.search(q, top_k=args.top_k, mode=args.mode):
        print(f"[{r['score']}] via={r['hit_via']:9} {r['title'][:60]}")
        if r["matched_keyphrases"]:
            print(f"      매칭 키프레이즈: {', '.join(r['matched_keyphrases'])}")


if __name__ == "__main__":
    main()
