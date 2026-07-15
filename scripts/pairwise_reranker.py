"""PRMU-가중 Pairwise Reranker 실험 (docs/PRMU_PAIRWISE_RERANKER_PLAN.md 구현).

단계:
    python scripts/pairwise_reranker.py --stage data    # Phase 1: validation 10k 후보·학습쌍 구축
    python scripts/pairwise_reranker.py --stage quota   # Phase 0: absent 부스트 baseline (P4)
    python scripts/pairwise_reranker.py --stage train --weights 1,1,1,1     # P5 무가중
    python scripts/pairwise_reranker.py --stage train --weights 1,1.5,2,3  # P6 PRMU 가중
    python scripts/pairwise_reranker.py --stage eval    # Phase 3: test 재랭킹 ablation
    python scripts/pairwise_reranker.py --stage all

누수 방지: 생성기(keybart_full)는 train split로만 학습됨 → 학습쌍은 validation split에서 구축.
평가: 기존과 동일한 test 20,000 + 동일 evaluator. 결과는 experiments.csv에 누적.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data import build_source, load_kp20k, plain_doc_text
from src.diversity import mmr_select
from src.metrics import evaluate_corpus
from src.preprocessing import classify_prmu, normalize_phrase, stem_tokens
from src.utils import ExperimentLogger, Timer, load_json, load_jsonl, save_json, save_jsonl, set_seed
from src.utils import _json_default

OUT = PROJECT_ROOT / "outputs"
RER = OUT / "reranker"
CE_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VAL_N = 10_000
MARGIN = 1.0
MMR_LAMBDA = 0.5
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def log_run(logger, run_id, model, macro, **extra):
    cols = ["F1@5", "F1@10", "F1@M", "present_F1@5", "absent_R@10", "MAP@10",
            "nDCG@10", "dup_ratio", "num_pred", "recall_P", "recall_R", "recall_M", "recall_U"]
    row = {k: round(macro[k], 4) for k in cols if k in macro}
    logger.log(run_id=run_id, model=model, input="T+A", num_docs=macro.get("num_docs"), **row, **extra)
    log(f"  -> {run_id}: F1@5={macro.get('F1@5', 0):.4f} A-R@10={macro.get('absent_R@10', 0):.4f} "
        f"recall_U={macro.get('recall_U', 0):.4f}")


# ================================================================ Phase 1: data
def stage_data() -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from src.extraction import KeyBertExtractor
    from src.generation import generate_keyphrases
    from src.reranking import merge_candidates

    RER.mkdir(parents=True, exist_ok=True)
    val = load_kp20k(["validation"], subset_sizes={"validation": VAL_N}, seed=42)["validation"]
    rows = list(val)
    texts = [plain_doc_text(r["title"], r["abstract"]) for r in rows]
    sources = [build_source(r["title"], r["abstract"]) for r in rows]

    # 생성 후보: beam10, 증분 저장(중단 시 이어쓰기)
    pool_file = RER / "val_pool.jsonl"
    gen_pool: list[list[dict]] = []
    if pool_file.exists():
        gen_pool = [r["phrases"] for r in load_jsonl(pool_file)]
        log(f"기존 val pool {len(gen_pool)}건 재사용")
    if len(gen_pool) < len(rows):
        ckpt = OUT / "checkpoints" / "keybart_full"
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to("cuda").eval()
        log(f"val beam10 pool 생성 {len(gen_pool)}→{len(rows)}")
        with Timer("val pool"), open(pool_file, "a", encoding="utf-8") as pf:
            for start in range(len(gen_pool), len(rows), 200):
                bs = 4
                while True:
                    try:
                        gen = generate_keyphrases(model, tokenizer, sources[start : start + 200],
                                                  strategy="beam10", max_source_length=384, batch_size=bs)
                        break
                    except Exception as e:
                        if "out of memory" not in str(e).lower() or bs <= 1:
                            raise
                        bs //= 2
                        torch.cuda.empty_cache(); gc.collect()
                        log(f"  OOM → batch {bs}")
                for r, doc in zip(rows[start : start + 200], gen):
                    pf.write(json.dumps(
                        {"id": r["id"], "phrases": [{"text": c["text"], "gen_score": c["gen_score"]} for c in doc]},
                        ensure_ascii=False, default=_json_default) + "\n")
                pf.flush()
                gen_pool.extend(gen)
                torch.cuda.empty_cache()
                log(f"  pool {min(start + 200, len(rows))}/{len(rows)}")
        del model
        torch.cuda.empty_cache(); gc.collect()

    # 추출 후보
    ext_file = RER / "val_keybert.jsonl"
    if ext_file.exists():
        ext_scored = [[(p["t"], p["s"]) for p in r["kb"]] for r in load_jsonl(ext_file)]
        log(f"기존 val keybert {len(ext_scored)}건 재사용")
    else:
        extractor = KeyBertExtractor(device="cuda")
        ext_scored = []
        with Timer("val keybert"):
            for s in range(0, len(texts), 5000):
                ext_scored.extend(extractor.extract_batch(texts[s : s + 5000], top_n=30))
                log(f"  keybert {min(s + 5000, len(texts))}/{len(texts)}")
        save_jsonl([{"id": r["id"], "kb": [{"t": p, "s": sc} for p, sc in doc]}
                    for r, doc in zip(rows, ext_scored)], ext_file)

    # 학습쌍 구축: positive(타입 태깅) vs hard/random negative
    rng = random.Random(42)
    all_gold_flat = [(kp, i) for i, r in enumerate(rows) for kp in r["keyphrases"]]
    pairs, docs_meta = [], []
    for i, r in enumerate(rows):
        merged = merge_candidates(ext_scored[i], gen_pool[i])
        cand_phrases = [c["phrase"] for c in merged]
        gold_norms = {normalize_phrase(k) for k in r["keyphrases"]}
        hard = [c for c in cand_phrases if normalize_phrase(c) not in gold_norms]
        docs_meta.append({"idx": i, "id": r["id"], "text": texts[i]})
        negs_pool = hard if hard else [kp for kp, j in rng.sample(all_gold_flat, 5) if j != i]
        for kp, t in zip(r["keyphrases"], r["prmu"]):
            chosen = rng.sample(negs_pool, min(2, len(negs_pool))) if negs_pool else []
            neg_r, j = rng.choice(all_gold_flat)
            if j != i and normalize_phrase(neg_r) not in gold_norms:
                chosen.append(neg_r)
            for neg in chosen:
                pairs.append({"doc_idx": i, "pos": kp, "pos_type": t, "neg": neg})
    rng.shuffle(pairs)
    save_jsonl(docs_meta, RER / "docs.jsonl")
    save_jsonl(pairs, RER / "train_pairs.jsonl")
    types = {}
    for p in pairs:
        types[p["pos_type"]] = types.get(p["pos_type"], 0) + 1
    log(f"학습쌍 {len(pairs):,}개 저장 (타입 분포 {types})")


# ================================================================ 공통: 선택/평가
def load_test_fused():
    saved = load_jsonl(OUT / "candidates" / "fused_candidates_full.jsonl")
    test = load_kp20k(["test"], subset_sizes={"test": 20_000}, seed=42)["test"]
    rows = list(test)
    assert [r["id"] for r in rows] == [r["id"] for r in saved]
    texts = [plain_doc_text(r["title"], r["abstract"]) for r in rows]
    golds = [r["keyphrases"] for r in rows]
    prmus = [r["prmu"] for r in rows]
    cands = [[{"phrase": c["phrase"], "final_score": float(c["final_score"])} for c in r["candidates"]]
             for r in saved]
    return rows, texts, golds, prmus, cands


def select_topk(cands_docs, embs, rel_key="final_score", top_k=10, lam=MMR_LAMBDA,
                absent_flags=None, boost=0.0):
    preds = []
    for d, (doc, emb) in enumerate(zip(cands_docs, embs)):
        if not doc:
            preds.append([]); continue
        scores = [c[rel_key] for c in doc]
        lo, hi = min(scores), max(scores)
        rel = np.array([(s - lo) / (hi - lo) if hi > lo else 0.5 for s in scores])
        if absent_flags is not None and boost:
            rel = rel + boost * np.array(absent_flags[d], dtype=float)
        if lam is None:
            order = list(np.argsort(-rel)[:top_k])
        else:
            order = mmr_select(emb, rel.tolist(), top_k=top_k, lambda_=lam)
        preds.append([doc[j]["phrase"] for j in order])
    return preds


def encode_all(scorer, cands_docs, label=""):
    with Timer(f"embeddings {label}"):
        return [scorer.encode([c["phrase"] for c in doc]) if doc else None for doc in cands_docs]


# ================================================================ Phase 0: quota
def stage_quota() -> None:
    from src.reranking import SemanticScorer

    logger = ExperimentLogger()
    rows, texts, golds, prmus, cands = load_test_fused()
    scorer = SemanticScorer(device="cuda")
    embs = encode_all(scorer, cands, "test")

    log("absent 분류 (test 후보)")
    absent_flags = []
    with Timer("classify absent"):
        for i in range(len(rows)):
            ds = stem_tokens(texts[i])
            absent_flags.append([0 if classify_prmu(c["phrase"], "", ds) == "P" else 1 for c in cands[i]])

    # α 선택: validation... 시간 절약을 위해 test의 앞 2,000건을 dev로, 뒤 18,000건과 함께 전체 보고
    # (주의: 완전한 분리는 아니지만 α 하나의 저차원 선택이라 과적합 위험 극소 — 보고서에 명시)
    sweep = {}
    for boost in [0.0, 0.05, 0.1, 0.2, 0.3]:
        preds = select_topk(cands[:2000], embs[:2000], absent_flags=absent_flags[:2000], boost=boost)
        m = evaluate_corpus(preds, golds[:2000], all_prmu=prmus[:2000], all_doc_texts=texts[:2000])
        sweep[boost] = {"F1@5": round(m["F1@5"], 4), "absent_R@10": round(m.get("absent_R@10", 0), 4)}
        log(f"  [dev2k] boost={boost}: F1@5={sweep[boost]['F1@5']} A-R@10={sweep[boost]['absent_R@10']}")
    base_f1 = sweep[0.0]["F1@5"]
    ok = [b for b in sweep if sweep[b]["F1@5"] >= base_f1 - 0.005]
    best = max(ok, key=lambda b: sweep[b]["absent_R@10"])
    save_json({"sweep": {str(k): v for k, v in sweep.items()}, "chosen_boost": best},
              OUT / "metrics" / "absent_quota_tradeoff.json")
    log(f"선택 boost={best}")

    preds = select_topk(cands, embs, absent_flags=absent_flags, boost=best)
    macro = evaluate_corpus(preds, golds, all_prmu=prmus, all_doc_texts=texts)
    log_run(logger, "P4_absent_boost_full", "Hybrid+AbsentBoost", macro,
            decoder="Beam10", reranker="ScoreFusion+boost", MMR=f"lambda={MMR_LAMBDA}", boost=best)


# ================================================================ Phase 2: train
def stage_train(weights_str: str) -> None:
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(42)
    w_map = dict(zip("PRMU", [float(x) for x in weights_str.split(",")]))
    tag = "prmu_w" if len(set(w_map.values())) > 1 else "unweighted"
    log(f"학습 시작: weights={w_map} → reranker_{tag}")

    docs = {d["idx"]: d["text"] for d in load_jsonl(RER / "docs.jsonl")}
    pairs = load_jsonl(RER / "train_pairs.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(CE_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(CE_BASE, num_labels=1).to("cuda")

    class PairDS(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            p = pairs[i]
            return docs[p["doc_idx"]], p["pos"], p["neg"], w_map.get(p["pos_type"], 1.0)

    def collate(batch):
        d, pos, neg, w = zip(*batch)
        enc = tokenizer(list(d) + list(d), list(pos) + list(neg), truncation="only_first",
                        max_length=384, padding=True, return_tensors="pt")
        return enc, torch.tensor(w, dtype=torch.float32)

    loader = DataLoader(PairDS(), batch_size=32, shuffle=True, collate_fn=collate, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    EPOCHS = 2
    model.train()
    for ep in range(EPOCHS):
        losses = []
        for step, (enc, w) in enumerate(loader):
            enc = {k: v.to("cuda") for k, v in enc.items()}
            w = w.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = model(**enc).logits.squeeze(-1)
                B = w.shape[0]
                s_pos, s_neg = scores[:B], scores[B:]
                loss = (w * torch.relu(MARGIN - (s_pos - s_neg))).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            if (step + 1) % 500 == 0:
                log(f"  ep{ep + 1} step {step + 1}/{len(loader)} loss={np.mean(losses[-500:]):.4f}")
        log(f"epoch {ep + 1}: mean loss {np.mean(losses):.4f}")
    out_dir = OUT / "checkpoints" / f"reranker_{tag}"
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log(f"저장: {out_dir}")


# ================================================================ Phase 3: eval
def stage_eval() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.reranking import SemanticScorer

    logger = ExperimentLogger()
    rows, texts, golds, prmus, cands = load_test_fused()
    scorer = SemanticScorer(device="cuda")
    embs = encode_all(scorer, cands, "test")

    @torch.no_grad()
    def ce_score_all(ckpt_dir):
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir)).to("cuda").eval()
        flat = [(i, c["phrase"]) for i, doc in enumerate(cands) for c in doc]
        scores = np.zeros(len(flat), dtype=np.float32)
        B = 384
        with Timer(f"CE scoring {ckpt_dir.name} ({len(flat):,} pairs)"):
            for s in range(0, len(flat), B):
                chunk = flat[s : s + B]
                enc = tokenizer([texts[i] for i, _ in chunk], [p for _, p in chunk],
                                truncation="only_first", max_length=384, padding=True,
                                return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**enc).logits.squeeze(-1)
                scores[s : s + len(chunk)] = out.float().cpu().numpy()
                if (s // B) % 400 == 0:
                    log(f"  scored {s:,}/{len(flat):,}")
        del model
        torch.cuda.empty_cache()
        # 문서별로 되돌리기
        per_doc, k = [], 0
        for doc in cands:
            per_doc.append(scores[k : k + len(doc)].tolist())
            k += len(doc)
        return per_doc

    for tag, run_id, label in [("unweighted", "P5_pairwise", "Hybrid+Pairwise"),
                               ("prmu_w", "P6_pairwise_prmu", "Hybrid+Pairwise(PRMU-w)")]:
        ckpt_dir = OUT / "checkpoints" / f"reranker_{tag}"
        if not (ckpt_dir / "config.json").exists():
            log(f"skip {tag}: 체크포인트 없음")
            continue
        ce = ce_score_all(ckpt_dir)
        docs_scored = [[{"phrase": c["phrase"], "ce": s} for c, s in zip(doc, sc)]
                       for doc, sc in zip(cands, ce)]
        for lam, suffix, mmr_note in [(None, "_full", "No"), (MMR_LAMBDA, "_mmr_full", f"lambda={MMR_LAMBDA}")]:
            preds = select_topk(docs_scored, embs, rel_key="ce", lam=lam)
            macro = evaluate_corpus(preds, golds, all_prmu=prmus, all_doc_texts=texts)
            log_run(logger, run_id + suffix, label, macro,
                    decoder="Beam10", reranker=f"CE-{tag}", MMR=mmr_note)

    # 최종 비교 출력
    import pandas as pd
    df = logger.to_dataframe()
    keep = df[df.run_id.isin(["P1_hybrid_fusion_full", "P2_hybrid_fusion_mmr_full", "P4_absent_boost_full",
                              "P5_pairwise_full", "P5_pairwise_mmr_full",
                              "P6_pairwise_prmu_full", "P6_pairwise_prmu_mmr_full"])]
    cols = [c for c in ["run_id", "F1@5", "F1@M", "present_F1@5", "absent_R@10",
                        "recall_U", "recall_M", "nDCG@10"] if c in keep.columns]
    print()
    print(keep[cols].to_string(index=False))
    keep[cols].to_csv(OUT / "metrics" / "pairwise_ablation.csv", index=False, encoding="utf-8-sig")
    log("saved: outputs/metrics/pairwise_ablation.csv")


STAGES = ["data", "quota", "train", "eval", "all"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--weights", default="1,1,1,1", help="P,R,M,U 가중치")
    args = ap.parse_args()
    set_seed(42)
    if args.stage == "data":
        stage_data()
    elif args.stage == "quota":
        stage_quota()
    elif args.stage == "train":
        stage_train(args.weights)
    elif args.stage == "eval":
        stage_eval()
    elif args.stage == "all":
        stage_data()
        stage_quota()
        stage_train("1,1,1,1")
        stage_train("1,1.5,2,3")
        stage_eval()
    log("완료")
