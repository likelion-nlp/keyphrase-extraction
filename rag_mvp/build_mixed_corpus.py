"""mixed 컬렉션 생성 = KP20k 전체 + arXiv 앞 N편(distractor).

이미 색인된 벡터를 복사만 하므로 P7 재생성 불필요.
    python rag_mvp/build_mixed_corpus.py --arxiv-vectors 22000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: F401  chromadb보다 먼저
from rag_mvp.core.index import VectorIndex


def copy(src: VectorIndex, dst: VectorIndex, limit: int | None = None, batch: int = 5000) -> int:
    n = min(limit, src.count()) if limit else src.count()
    off = copied = 0
    while copied < n:
        b = min(batch, n - copied)
        g = src.col.get(limit=b, offset=off, include=["embeddings", "metadatas", "documents"])
        if not g["ids"]:
            break
        dst.col.add(ids=g["ids"], embeddings=g["embeddings"],
                    metadatas=g["metadatas"], documents=g["documents"])
        copied += len(g["ids"]); off += len(g["ids"])
        print(f"  {src.collection} → {dst.collection}: {copied}/{n}", flush=True)
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv-vectors", type=int, default=22000, help="arXiv 벡터 수(~2000편×11)")
    args = ap.parse_args()

    kp = VectorIndex(collection="kp20k")
    ar = VectorIndex(collection="papers")
    mx = VectorIndex(collection="mixed"); mx.reset()
    print(f"KP20k {kp.count()} · arXiv {ar.count()} → mixed 생성", flush=True)

    copy(kp, mx)                              # KP20k 전체
    copy(ar, mx, limit=args.arxiv_vectors)    # arXiv distractor
    print(f"\n완료: mixed 총 {mx.count()} 벡터 (KP20k 전체 + arXiv {args.arxiv_vectors} 벡터)")


if __name__ == "__main__":
    main()
