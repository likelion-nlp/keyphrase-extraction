"""논지 평가 (배치판) — absent 키프레이즈로 원 논문을 찾아가나? (어휘 불일치 해결)

각 KP20k 문서의 absent gold를 질의로 self-retrieval. plain(본문만) vs kp(본문+P7 키프레이즈).
질의 임베딩을 한 번에 배치 + Chroma 배치 쿼리로 처리 (수만 질의도 1~2분).

    python rag_mvp/eval_absent_retrieval.py --limit 20000 --absent U --topk 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAG = Path(__file__).resolve().parent
sys.path.insert(0, str(RAG.parent))

import numpy as np
import torch  # noqa: F401  chromadb보다 먼저
from rag_mvp.core.index import Q_PREFIX, VectorIndex
from src.data import load_kp20k


def blended_scores(doc_meta, doc_dist, kp_meta, kp_dist, idx) -> dict[str, float]:
    doc_sim = {m["doc_id"]: 1.0 - float(d) for m, d in zip(doc_meta, doc_dist)}
    kp_best: dict[str, float] = {}
    for m, d in zip(kp_meta, kp_dist):
        did = m["doc_id"]
        kp_best[did] = max(kp_best.get(did, 0.0), 1.0 - float(d))
    out = {}
    for did in set(doc_sim) | set(kp_best):
        ds = doc_sim.get(did)
        ks = kp_best.get(did, 0.0)
        if ds is not None and did in kp_best:
            out[did] = idx.W_DOC * ds + idx.W_KP * ks
        elif ds is not None:
            out[did] = ds
        else:
            out[did] = idx.KP_ONLY_PENALTY * ks
    return out, doc_sim


def rank_of(scores: dict[str, float], target: str, topk: int) -> int:
    ordered = sorted(scores.items(), key=lambda x: -x[1])[:topk]
    for i, (did, _) in enumerate(ordered, 1):
        if did == target:
            return i
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--absent", default="U")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--pool", type=int, default=150)
    ap.add_argument("--collection", default="kp20k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--qchunk", type=int, default=500, help="Chroma 배치 쿼리 크기")
    args = ap.parse_args()
    absent_set = set(args.absent.upper())
    t0 = time.time()

    docs = list(load_kp20k(["test"], subset_sizes={"test": args.limit}, seed=42)["test"])[: args.limit]
    idx = VectorIndex(device=args.device, collection=args.collection)
    queries = [(str(r["id"]), kp) for r in docs for kp, pr in zip(r["keyphrases"], r["prmu"])
               if pr in absent_set]
    print(f"[{time.time()-t0:.0f}s] 컬렉션 '{args.collection}' {idx.count()} 벡터 · "
          f"absent({''.join(sorted(absent_set))}) 질의 {len(queries)}개", flush=True)

    # ① 질의 임베딩 일괄
    qembs = idx.embedder.encode([Q_PREFIX + q for _, q in queries], batch_size=256,
                                normalize_embeddings=True, show_progress_bar=False)
    print(f"[{time.time()-t0:.0f}s] 질의 임베딩 완료. 배치 검색 시작", flush=True)

    agg = {m: {"rr": 0.0, "h1": 0, "h10": 0} for m in ("kp", "plain")}
    N = len(queries)
    for s in range(0, N, args.qchunk):
        qe = qembs[s : s + args.qchunk].tolist()
        dres = idx.col.query(query_embeddings=qe, n_results=args.pool, where={"kind": "doc"})
        kres = idx.col.query(query_embeddings=qe, n_results=args.pool, where={"kind": "keyphrase"})
        for j in range(len(qe)):
            target = queries[s + j][0]
            kp_scores, doc_sim = blended_scores(dres["metadatas"][j], dres["distances"][j],
                                                kres["metadatas"][j], kres["distances"][j], idx)
            for mode, sc in (("kp", kp_scores), ("plain", doc_sim)):
                r = rank_of(sc, target, args.topk)
                a = agg[mode]
                a["rr"] += 1.0 / r if r else 0.0
                a["h1"] += int(r == 1)
                a["h10"] += int(0 < r <= 10)
        print(f"[{time.time()-t0:.0f}s] {min(s+args.qchunk, N)}/{N}", flush=True)

    print(f"\n=== absent 자기검색 (kp=키프레이즈 강화, plain=본문만) · {N}질의 · {len(docs)}편 · top-{args.topk} ===")
    print(f"{'지표':<8}{'plain':>10}{'kp':>10}{'개선':>12}")
    for label, key, isrr in [("MRR", "rr", True), ("Hit@1", "h1", False), ("Hit@10", "h10", False)]:
        pv = agg["plain"][key] / N
        kv = agg["kp"][key] / N
        imp = (kv - pv) / pv * 100 if pv > 0 else float("inf")
        print(f"{label:<8}{pv:>10.4f}{kv:>10.4f}{imp:>+11.1f}%")
    print(f"[{time.time()-t0:.0f}s] 완료")


if __name__ == "__main__":
    main()
