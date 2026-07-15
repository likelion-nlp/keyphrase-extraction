"""2차 노트북의 SciBERT 하이브리드 랭커를 우리 실험 조건에 그대로 이식한다 (P7).

목적: 랭커만 바꾼 순수 비교.
  - 후보 pool: 우리 것과 동일 (KeyBERT top30 ∪ keybart_full beam10) — 이미 저장된 fused_candidates 재사용
  - 평가: 우리 것과 동일 (test 20,000, 동일 evaluator)
  - 차이: 스코어러만 SciBERT + aux feature MLP, PRMU-가중 pairwise (노트북 설계)

노트북과의 차이(의도적):
  - KeyBART zero-shot → 우리 fine-tuned keybart_full (후보 품질을 우리 시스템과 동일하게)
  - train 1,000 → validation split 10,000 (P5/P6와 동일한 학습 데이터·누수 방지)

단계:
    python scripts/scibert_hybrid_ranker.py --stage train   # SciBERT 랭커 학습 (~40분)
    python scripts/scibert_hybrid_ranker.py --stage eval    # test 20,000 재랭킹 + 기록 (~30분)
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.data import load_kp20k, plain_doc_text
from src.metrics import evaluate_corpus
from src.preprocessing import classify_prmu, normalize_phrase, stem_tokens
from src.utils import ExperimentLogger, Timer, load_jsonl, save_jsonl, set_seed

OUT = PROJECT_ROOT / "outputs"
RER = OUT / "reranker"
ENCODER = "allenai/scibert_scivocab_uncased"
PRMU_WEIGHT = {"P": 1.0, "R": 1.5, "M": 2.0, "U": 2.5}   # 노트북과 동일
MARGIN, MAX_LEN, N_AUX = 0.3, 256, 3
GEN_MISSING = -5.0
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


class HybridRanker(nn.Module):
    """SciBERT [CLS] + 보조 피처(gen_score, has_gen, is_present) → MLP 스칼라 점수."""

    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER)
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h + N_AUX, 128), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask, aux):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(torch.cat([cls, aux], dim=-1)).squeeze(-1)


def make_aux(gen_score: float | None, has_gen: bool, is_present: bool) -> list[float]:
    g = GEN_MISSING if gen_score is None else float(gen_score)
    return [g / 5.0, 1.0 if has_gen else 0.0, 1.0 if is_present else 0.0]


# ---------------------------------------------------------------- 후보 재구성
def build_records(pool_file: Path, ext_file: Path, rows, texts):
    """생성/추출 후보를 노트북 레코드 형식({phrase, gen_score, source, is_present})으로 합친다."""
    from src.reranking import merge_candidates

    gen_pool = {r["id"]: r["phrases"] for r in load_jsonl(pool_file)}
    ext_pool = {r["id"]: [(p["t"], p["s"]) for p in r["kb"]] for r in load_jsonl(ext_file)}
    out = []
    for i, r in enumerate(rows):
        merged = merge_candidates(ext_pool[r["id"]], gen_pool[r["id"]])
        ds = stem_tokens(texts[i])
        recs = []
        for c in merged:
            recs.append({
                "phrase": c["phrase"],
                "gen_score": c["gen_score"],
                "source": "both" if len(c["sources"]) == 2 else c["sources"][0][:3],
                "is_present": classify_prmu(c["phrase"], "", ds) == "P",
            })
        out.append(recs)
    return out


# ---------------------------------------------------------------- 학습
def stage_train() -> None:
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(ENCODER)

    val = list(load_kp20k(["validation"], subset_sizes={"validation": 10_000}, seed=42)["validation"])
    texts = [plain_doc_text(r["title"], r["abstract"]) for r in val]
    log("validation 후보 재구성 (P5/P6와 동일한 pool)")
    recs_list = build_records(RER / "val_pool.jsonl", RER / "val_keybert.jsonl", val, texts)

    # pairwise 쌍 구성 — absent positive에 negative를 2배 배정 (노트북 absent_boost)
    rng = random.Random(42)
    pairs = []
    for i, (row, recs) in enumerate(zip(val, recs_list)):
        prmu_map = {normalize_phrase(k): t for k, t in zip(row["keyphrases"], row["prmu"])}
        gold = set(prmu_map)
        pos = [r for r in recs if normalize_phrase(r["phrase"]) in gold]
        neg = [r for r in recs if normalize_phrase(r["phrase"]) not in gold]
        if not pos or not neg:
            continue
        for p in pos:
            w = PRMU_WEIGHT.get(prmu_map.get(normalize_phrase(p["phrase"]), "P"), 1.0)
            n_neg = 4 * (2 if not p["is_present"] else 1)
            for n in rng.sample(neg, min(n_neg, len(neg))):
                pairs.append((i, p, n, w))
    rng.shuffle(pairs)
    n_abs = sum(1 for _, p, _, _ in pairs if not p["is_present"])
    log(f"학습쌍 {len(pairs):,}개 (absent positive 쌍 {n_abs:,})")

    class PairDS(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, k):
            di, p, n, w = pairs[k]
            return texts[di], p, n, w

    def collate(batch):
        t, pos, neg, w = zip(*batch)
        enc = tok(list(t) + list(t), [x["phrase"] for x in pos] + [x["phrase"] for x in neg],
                  truncation="only_first", max_length=MAX_LEN, padding=True, return_tensors="pt")
        aux = torch.tensor(
            [make_aux(x["gen_score"], x["source"] in ("gen", "both"), x["is_present"]) for x in pos]
            + [make_aux(x["gen_score"], x["source"] in ("gen", "both"), x["is_present"]) for x in neg],
            dtype=torch.float)
        return enc, aux, torch.tensor(w, dtype=torch.float)

    loader = DataLoader(PairDS(), batch_size=16, shuffle=True, collate_fn=collate)
    model = HybridRanker().to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for ep in range(2):
        losses = []
        for step, (enc, aux, w) in enumerate(loader):
            enc = {k: v.to("cuda") for k, v in enc.items()}
            aux, w = aux.to("cuda"), w.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model(enc["input_ids"], enc["attention_mask"], aux)
                B = w.shape[0]
                loss = (w * torch.clamp(MARGIN - (s[:B] - s[B:]), min=0)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            if (step + 1) % 500 == 0:
                log(f"  ep{ep+1} {step+1}/{len(loader)} loss={np.mean(losses[-500:]):.4f}")
        log(f"epoch {ep+1} 평균 loss {np.mean(losses):.4f}")
    d = OUT / "checkpoints" / "reranker_scibert"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    tok.save_pretrained(str(d))
    log(f"저장: {d}")


# ---------------------------------------------------------------- 평가
@torch.no_grad()
def stage_eval() -> None:
    from src.diversity import mmr_select
    from src.reranking import SemanticScorer

    set_seed(42)
    logger = ExperimentLogger()
    tok = AutoTokenizer.from_pretrained(str(OUT / "checkpoints" / "reranker_scibert"))
    model = HybridRanker().to("cuda")
    model.load_state_dict(torch.load(OUT / "checkpoints" / "reranker_scibert" / "model.pt"))
    model.eval()

    test = list(load_kp20k(["test"], subset_sizes={"test": 20_000}, seed=42)["test"])
    texts = [plain_doc_text(r["title"], r["abstract"]) for r in test]
    golds = [r["keyphrases"] for r in test]
    prmus = [r["prmu"] for r in test]
    saved = load_jsonl(OUT / "candidates" / "fused_candidates_full.jsonl")
    assert [r["id"] for r in test] == [r["id"] for r in saved]

    # 후보 재구성 (gen_score/source는 beam10 pool에서, is_present는 재판정)
    gen_pool = {r["id"]: {c["text"]: c["gen_score"] for c in r["phrases"]}
                for r in load_jsonl(OUT / "candidates" / "keybart_full_beam10.jsonl")}
    log("test 후보 준비")
    recs_list = []
    for i, r in enumerate(saved):
        ds = stem_tokens(texts[i])
        gmap = gen_pool.get(r["id"], {})
        recs = []
        for c in r["candidates"]:
            ph = c["phrase"]
            srcs = c.get("sources", [])
            recs.append({
                "phrase": ph,
                "gen_score": gmap.get(ph),
                "source": "both" if len(srcs) == 2 else (srcs[0][:3] if srcs else "ext"),
                "is_present": classify_prmu(ph, "", ds) == "P",
            })
        recs_list.append(recs)

    log("SciBERT 스코어링")
    all_scores = []
    B = 128
    for i, recs in enumerate(recs_list):
        sc = []
        for s in range(0, len(recs), B):
            ch = recs[s : s + B]
            enc = tok([texts[i]] * len(ch), [r["phrase"] for r in ch], truncation="only_first",
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
            aux = torch.tensor([make_aux(r["gen_score"], r["source"] in ("gen", "both"),
                                         r["is_present"]) for r in ch],
                               dtype=torch.float).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sc.extend(model(enc["input_ids"], enc["attention_mask"], aux).float().cpu().tolist())
        all_scores.append(sc)
        if (i + 1) % 2000 == 0:
            log(f"  scored {i+1}/{len(recs_list)}")
    del model
    torch.cuda.empty_cache(); gc.collect()

    scorer = SemanticScorer(device="cuda")
    embs = [scorer.encode([r["phrase"] for r in recs]) if recs else None for recs in recs_list]

    for lam, run_id, note in [(None, "P7_scibert_hybrid_full", "No"),
                              (0.5, "P7_scibert_hybrid_mmr_full", "lambda=0.5")]:
        preds = []
        for d, (recs, sc) in enumerate(zip(recs_list, all_scores)):
            if not recs:
                preds.append([]); continue
            arr = np.asarray(sc)
            lo, hi = arr.min(), arr.max()
            rel = (arr - lo) / (hi - lo) if hi > lo else np.full_like(arr, 0.5)
            order = (list(np.argsort(-rel)[:10]) if lam is None
                     else mmr_select(embs[d], rel.tolist(), top_k=10, lambda_=lam))
            preds.append([recs[j]["phrase"] for j in order])
        macro = evaluate_corpus(preds, golds, all_prmu=prmus, all_doc_texts=texts)
        cols = ["F1@5", "F1@10", "F1@M", "present_F1@5", "absent_R@10", "MAP@10", "nDCG@10",
                "dup_ratio", "recall_P", "recall_R", "recall_M", "recall_U"]
        logger.log(run_id=run_id, model="SciBERT Hybrid Ranker", input="T+A", decoder="Beam10",
                   reranker="SciBERT+aux (PRMU-w pairwise)", MMR=note, seed=42,
                   num_docs=macro["num_docs"], **{k: round(macro[k], 4) for k in cols if k in macro})
        save_jsonl([{"id": r["id"], "title": r["title"], "gold": g, "prmu": p, "pred": pr}
                    for r, g, p, pr in zip(test, golds, prmus, preds)],
                   OUT / "predictions" / f"{run_id}.jsonl")
        log(f"✓ {run_id}: F1@5={macro['F1@5']:.4f} A-R@10={macro.get('absent_R@10',0):.4f} "
            f"recall_U={macro.get('recall_U',0):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "eval", "all"], required=True)
    a = ap.parse_args()
    if a.stage in ("train", "all"):
        stage_train()
    if a.stage in ("eval", "all"):
        stage_eval()
    log("완료")
