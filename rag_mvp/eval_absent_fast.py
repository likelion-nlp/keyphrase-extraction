"""빠른 absent 자기검색 평가 + 가중치 스윕 (Chroma where 우회, GPU 행렬곱).

임베딩을 컬렉션에서 한 번 로드 → 질의×문서/키프레이즈 유사도를 GPU 행렬곱 →
문서별 혼합 점수로 전체 순위(정확 rank) 계산. 여러 (본문:키프레이즈) 가중치를 한 번에 비교.

    python rag_mvp/eval_absent_fast.py --collection kp20k --limit 20000 --absent U
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from rag_mvp.core.index import Q_PREFIX, VectorIndex
from src.data import load_kp20k


def load_vectors(idx: VectorIndex, page: int = 5000):
    """컬렉션의 doc/keyphrase 임베딩을 페이지로 로드."""
    doc_emb, doc_ids, kp_emb, kp_docid = [], [], [], []
    off, total = 0, idx.count()
    while off < total:
        g = idx.col.get(limit=page, offset=off, include=["embeddings", "metadatas"])
        if not g["ids"]:
            break
        for e, m in zip(g["embeddings"], g["metadatas"]):
            if m.get("kind") == "doc":
                doc_emb.append(e); doc_ids.append(m["doc_id"])
            else:
                kp_emb.append(e); kp_docid.append(m["doc_id"])
        off += len(g["ids"])
    return (np.asarray(doc_emb, np.float32), doc_ids,
            np.asarray(kp_emb, np.float32), kp_docid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="kp20k")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--absent", default="U")
    ap.add_argument("--qchunk", type=int, default=256)
    ap.add_argument("--weights", default="0.6,0.5,0.4,0.3,0.2,0.0",
                    help="본문 가중 목록(키프레이즈 가중=1-본문). plain은 자동 포함")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    absent_set = set(args.absent.upper())
    t0 = time.time()

    docs = list(load_kp20k(["test"], subset_sizes={"test": args.limit}, seed=42)["test"])[: args.limit]
    idx = VectorIndex(device=dev, collection=args.collection)
    print(f"[{time.time()-t0:.0f}s] 벡터 로드 중 ({idx.count()})...", flush=True)
    doc_emb, doc_ids, kp_emb, kp_docid = load_vectors(idx)
    id2i = {d: i for i, d in enumerate(doc_ids)}
    Ndoc = len(doc_ids)
    kp_idx = np.array([id2i[d] for d in kp_docid], dtype=np.int64)
    print(f"[{time.time()-t0:.0f}s] doc {Ndoc} · keyphrase {len(kp_docid)} 로드 완료", flush=True)

    # 질의 (target doc_id, absent 키프레이즈)
    Q = [(str(r["id"]), kp) for r in docs for kp, pr in zip(r["keyphrases"], r["prmu"])
         if pr in absent_set and str(r["id"]) in id2i]
    qtext = [Q_PREFIX + q for _, q in Q]
    tgt = np.array([id2i[d] for d, _ in Q], dtype=np.int64)
    q_emb = idx.embedder.encode(qtext, batch_size=256, normalize_embeddings=True,
                                show_progress_bar=False).astype(np.float32)
    print(f"[{time.time()-t0:.0f}s] 질의 {len(Q)}개 임베딩 완료. GPU 랭킹 시작", flush=True)

    D = torch.tensor(doc_emb, device=dev)          # Ndoc × 384
    K = torch.tensor(kp_emb, device=dev)           # Nkp × 384
    Kidx = torch.tensor(kp_idx, device=dev)
    N = len(Q)

    settings = [("plain", None)] + [(f"kp(본문{w})", float(w)) for w in args.weights.split(",")]
    acc = {name: {"rr": 0.0, "h1": 0, "h10": 0} for name, _ in settings}

    for s in range(0, N, args.qchunk):
        qc = torch.tensor(q_emb[s : s + args.qchunk], device=dev)   # c × 384
        c = qc.shape[0]
        doc_sim = qc @ D.T                                          # c × Ndoc
        kp_sim = qc @ K.T                                           # c × Nkp
        kp_best = torch.full((c, Ndoc), -1.0, device=dev)
        kp_best.scatter_reduce_(1, Kidx.unsqueeze(0).expand(c, -1), kp_sim,
                                reduce="amax", include_self=True)   # c × Ndoc
        tg = torch.tensor(tgt[s : s + c], device=dev)
        for name, wdoc in settings:
            score = doc_sim if wdoc is None else (wdoc * doc_sim + (1 - wdoc) * kp_best)
            tgt_score = score[torch.arange(c), tg].unsqueeze(1)
            rank = (score > tgt_score).sum(dim=1) + 1              # 정확 순위
            a = acc[name]
            a["rr"] += (1.0 / rank.float()).sum().item()
            a["h1"] += (rank == 1).sum().item()
            a["h10"] += (rank <= 10).sum().item()

    print(f"\n=== absent({''.join(sorted(absent_set))}) 자기검색 · {N}질의 · {Ndoc}편 · [{time.time()-t0:.0f}s] ===")
    plain = acc["plain"]
    base_mrr = plain["rr"] / N
    print(f"{'설정':<12}{'MRR':>9}{'Hit@1':>9}{'Hit@10':>9}{'MRR개선':>10}")
    for name, _ in settings:
        a = acc[name]
        mrr, h1, h10 = a["rr"] / N, a["h1"] / N, a["h10"] / N
        imp = "" if name == "plain" else f"{(mrr-base_mrr)/base_mrr*100:+.1f}%"
        print(f"{name:<12}{mrr:>9.4f}{h1:>9.4f}{h10:>9.4f}{imp:>10}")


if __name__ == "__main__":
    main()
