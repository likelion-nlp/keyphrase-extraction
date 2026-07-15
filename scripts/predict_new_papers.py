"""외부 논문 CSV(제목+초록, gold 없음)에 전체 GERD 파이프라인을 적용하고
reference-free 지표를 계산한다.

사용:
    python scripts/predict_new_papers.py <input.csv>                    # score fusion + MMR (P2)
    python scripts/predict_new_papers.py <input.csv> --ranker pairwise  # PRMU-가중 pairwise CE (P6)

입력 CSV 필수 컬럼: title, abstract (id류 컬럼은 있으면 유지)
pairwise 모드는 실험 결과에 따라 MMR 없이 CE 순위를 그대로 쓴다 (CE top-10은 이미 다양).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data import build_source, plain_doc_text
from src.diversity import mmr_select
from src.extraction import KeyBertExtractor
from src.generation import generate_keyphrases
from src.metrics import semantic_redundancy, stem_duplicate_ratio
from src.preprocessing import classify_prmu, stem_tokens
from src.reranking import SemanticScorer, fuse_scores, merge_candidates
from src.utils import set_seed

TOP_K = 10
MMR_LAMBDA = 0.5   # validation 선택값


@torch.no_grad()
def ce_rank(ce_model, ce_tok, doc_text: str, candidates: list[dict]) -> list[dict]:
    """PRMU-가중 pairwise CE 점수로 후보를 재정렬한다 (final_score를 CE 점수로 교체)."""
    phrases = [c["phrase"] for c in candidates]
    enc = ce_tok([doc_text] * len(phrases), phrases, truncation="only_first",
                 max_length=384, padding=True, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        scores = ce_model(**enc).logits.squeeze(-1).float().cpu().numpy()
    for c, s in zip(candidates, scores):
        c["final_score"] = float(s)
    return sorted(candidates, key=lambda c: -c["final_score"])


def main(csv_path: str, ranker: str = "fusion") -> None:
    set_seed(42)
    t0 = time.time()
    df = pd.read_csv(csv_path)
    assert {"title", "abstract"} <= set(df.columns), "title/abstract 컬럼 필요"
    docs = df.to_dict("records")
    print(f"입력: {len(docs)} 문서 ({Path(csv_path).name}) | ranker={ranker}")

    # 전체 데이터로 학습한 최고 구성 로드
    ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / "keybart_full"
    assert (ckpt / "config.json").exists(), "keybart_full 체크포인트 필요"
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to("cuda").eval()
    extractor = KeyBertExtractor(device="cuda")
    scorer = SemanticScorer(device="cuda")

    ce_model = ce_tok = None
    if ranker == "pairwise":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer as _AT

        ce_dir = PROJECT_ROOT / "outputs" / "checkpoints" / "reranker_prmu_w"
        assert (ce_dir / "config.json").exists(), "reranker_prmu_w 체크포인트 필요 (pairwise_reranker.py --stage train)"
        ce_tok = _AT.from_pretrained(str(ce_dir))
        ce_model = AutoModelForSequenceClassification.from_pretrained(str(ce_dir)).to("cuda").eval()
    print(f"[{time.time()-t0:.0f}s] 모델 로드 완료 (KeyBART_full + KeyBERT + SBERT"
          + (" + PRMU-가중 CE)" if ce_model is not None else ")"))

    texts = [plain_doc_text(r["title"], r["abstract"]) for r in docs]
    sources = [build_source(r["title"], r["abstract"]) for r in docs]

    gen_pool = generate_keyphrases(model, tokenizer, sources, strategy="beam10",
                                   max_source_length=384, batch_size=4)
    ext_scored = extractor.extract_batch(texts, top_n=30)
    print(f"[{time.time()-t0:.0f}s] 후보 생성 완료 (생성 평균 "
          f"{np.mean([len(d) for d in gen_pool]):.1f}개 + 추출 30개/doc)")

    # fusion → MMR
    results, agg = [], {"dup": [], "redund": [], "n_present": 0, "n_absent": 0,
                        "doc_sim": [], "n_pred": []}
    for i, row in enumerate(docs):
        merged = merge_candidates(ext_scored[i], gen_pool[i])
        if ce_model is not None:
            ranked = ce_rank(ce_model, ce_tok, texts[i], merged)
        else:
            ranked = fuse_scores(merged, texts[i], row["title"], scorer)
        if not ranked:
            results.append({**row, "keyphrases": ""})
            continue
        emb = scorer.encode([c["phrase"] for c in ranked])
        if ce_model is not None:
            order = list(range(min(TOP_K, len(ranked))))   # CE 순위 그대로 (MMR 없음)
        else:
            rel_raw = [c["final_score"] for c in ranked]
            lo, hi = min(rel_raw), max(rel_raw)
            rel = [(s - lo) / (hi - lo) if hi > lo else 0.5 for s in rel_raw]
            order = mmr_select(emb, rel, top_k=TOP_K, lambda_=MMR_LAMBDA)
        doc_stems = stem_tokens(texts[i])
        doc_emb = scorer.encode([texts[i]])[0]

        picked = []
        for rank, j in enumerate(order, 1):
            c = ranked[j]
            tag = classify_prmu(c["phrase"], "", doc_stems=doc_stems)
            kind = "present" if tag == "P" else "absent"
            agg["n_present" if kind == "present" else "n_absent"] += 1
            picked.append({"rank": rank, "phrase": c["phrase"], "type": kind, "prmu": tag,
                           "score": round(float(c["final_score"]), 3),
                           "source": "+".join(sorted(set(c["sources"])))})
        phrases = [p["phrase"] for p in picked]
        sel_emb = np.stack([emb[j] for j in order])
        agg["dup"].append(stem_duplicate_ratio(phrases))
        agg["redund"].append(semantic_redundancy(sel_emb))
        agg["doc_sim"].append(float((sel_emb @ doc_emb).mean()))
        agg["n_pred"].append(len(phrases))
        results.append({**{k: row[k] for k in row},
                        "keyphrases": "; ".join(f"{p['phrase']}[{p['type'][0].upper()}]" for p in picked),
                        "detail": picked})

    stem = Path(csv_path).stem + ("_pairwise" if ranker == "pairwise" else "")
    pred_dir = PROJECT_ROOT / "outputs" / "predictions"

    # ① 논문별 요약 (1행 = 논문 1편): 키프레이즈 + 문서 단위 품질 지표
    summary_rows = []
    for idx, r in enumerate(results):
        detail = r.get("detail", [])
        summary_rows.append({
            **{k: v for k, v in r.items() if k not in ("detail", "abstract", "keyphrases")},
            "keyphrases_top10": "; ".join(p["phrase"] for p in detail),
            "n_present": sum(p["type"] == "present" for p in detail),
            "n_absent": sum(p["type"] == "absent" for p in detail),
            "stem_dup_ratio": round(agg["dup"][idx], 4) if idx < len(agg["dup"]) else None,
            "semantic_redundancy": round(agg["redund"][idx], 4) if idx < len(agg["redund"]) else None,
            "doc_pred_cosine": round(agg["doc_sim"][idx], 4) if idx < len(agg["doc_sim"]) else None,
        })
    summary_csv = pred_dir / f"newpapers_{stem}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # ② 키프레이즈별 상세 (1행 = 키프레이즈 1개): rank/type/score/출처
    long_rows = []
    id_cols = [c for c in df.columns if c not in ("title", "abstract")]
    for r in results:
        for p in r.get("detail", []):
            long_rows.append({
                **{c: r.get(c) for c in id_cols},
                "title": r["title"],
                "rank": p["rank"], "keyphrase": p["phrase"],
                "type": p["type"], "prmu": p["prmu"],
                "score": p["score"], "source": p["source"],
            })
    long_csv = pred_dir / f"newpapers_{stem}_keyphrases.csv"
    pd.DataFrame(long_rows).to_csv(long_csv, index=False, encoding="utf-8-sig")

    print(f"[{time.time()-t0:.0f}s] 저장:")
    print(f"  요약(논문별 1행) : {summary_csv}")
    print(f"  상세(구절별 1행) : {long_csv}")

    total = agg["n_present"] + agg["n_absent"]
    print("\n=== Reference-free 지표 (gold 없음 → F1 계산 불가) ===")
    print(f"문서당 예측 수        : {np.mean(agg['n_pred']):.2f}")
    print(f"stem 중복률           : {np.mean(agg['dup']):.4f}  (KP20k test에선 0.000)")
    print(f"semantic redundancy   : {np.mean(agg['redund']):.4f}  (KP20k P2: ~0.34)")
    print(f"문서-예측 cosine 평균 : {np.mean(agg['doc_sim']):.4f}  (관련성 proxy)")
    print(f"present : absent      : {agg['n_present']} : {agg['n_absent']} "
          f"({100*agg['n_present']/total:.1f}% : {100*agg['n_absent']/total:.1f}%)")

    print("\n=== 샘플 3건 ===")
    for r in results[:3]:
        print(f"\n[{r['title'][:75]}]")
        for p in r["detail"][:7]:
            print(f"  {p['rank']:>2}. {p['phrase']:<45} {p['type']:<8} {p['score']:.3f} ({p['source']})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(PROJECT_ROOT / "cs_papers_20232024.csv"))
    ap.add_argument("--ranker", choices=["fusion", "pairwise"], default="fusion")
    args = ap.parse_args()
    main(args.csv, ranker=args.ranker)
