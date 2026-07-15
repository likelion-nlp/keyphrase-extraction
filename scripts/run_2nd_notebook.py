"""2차.ipynb의 로직을 그대로 스크립트로 이식해 실행한다 (nbconvert 커널 사망 회피).

노트북과 동일: spaCy 추출 후보 + KeyBART 생성 후보 → SciBERT+aux 하이브리드 랭커
              (PRMU-가중 pairwise, margin 0.3) → 평가 + Rank CI(Holm)

사용:
    # 원본 재현 (KeyBART zero-shot, train 1000 / test 200)
    python scripts/run_2nd_notebook.py --keybart zeroshot --n-train 1000 --n-test 200

    # 간단 버전 (우리 fine-tuned KeyBART, 규모 확대)
    python scripts/run_2nd_notebook.py --keybart finetuned --n-train 5000 --n-test 2000
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
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from src.data import load_kp20k, plain_doc_text
from src.metrics import evaluate_corpus
from src.utils import ExperimentLogger, save_json, save_jsonl, set_seed

OUT = PROJECT_ROOT / "outputs"
ENCODER = "allenai/scibert_scivocab_uncased"
PRMU_WEIGHT = {"P": 1.0, "R": 1.5, "M": 2.0, "U": 2.5}
GEN_MISSING, MARGIN, N_AUX = -5.0, 0.3, 3
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- 노트북 유틸 (그대로)
import spacy
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
stemmer = PorterStemmer()


def normalize(phrase: str) -> str:
    return " ".join(stemmer.stem(w) for w in phrase.lower().split())


def get_text(ex: dict) -> str:
    return ex["title"] + ". " + ex["abstract"]


def extractive_candidates(text: str, max_len: int = 5) -> set[str]:
    doc = nlp(text)
    cands = set()
    for chunk in doc.noun_chunks:
        p = chunk.text.strip().lower()
        if 0 < len(p.split()) <= max_len:
            cands.add(p)
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop and tok.is_alpha:
            cands.add(tok.text.lower())
    return cands


def f1_at_k(pred, gold, k):
    pk = pred[:k]
    tp = len(set(pk) & set(gold))
    p = tp / max(len(pk), 1)
    r = tp / max(len(gold), 1)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def recall_at_k(pred, gold, k):
    return len(set(pred[:k]) & set(gold)) / len(gold) if gold else None


def average_precision(pred, gold):
    if not gold:
        return None
    gs, hits, ap = set(gold), 0, 0.0
    for i, p in enumerate(pred, 1):
        if p in gs:
            hits += 1
            ap += hits / i
    return ap / len(gs)


# ---------------------------------------------------------------- 모델
class HybridRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER)
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h + N_AUX, 128), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, ids, mask, aux):
        cls = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(torch.cat([cls, aux], dim=-1)).squeeze(-1)


def make_aux(rec):
    return [rec["gen_score"] / 5.0,
            1.0 if rec["source"] in ("gen", "both") else 0.0,
            1.0 if rec["is_present"] else 0.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keybart", choices=["zeroshot", "finetuned"], default="zeroshot")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    set_seed(42)
    tag = f"{args.keybart}_tr{args.n_train}_te{args.n_test}"
    log(f"설정: KeyBART={args.keybart}, train={args.n_train}, test={args.n_test}")

    ds = load_kp20k(["train", "test"],
                    subset_sizes={"train": args.n_train, "test": args.n_test}, seed=42)
    train_ds, test_ds = list(ds["train"]), list(ds["test"])

    # ---------- KeyBART 생성 후보 ----------
    kb_path = ("bloomberg/KeyBART" if args.keybart == "zeroshot"
               else str(OUT / "checkpoints" / "keybart_full"))
    kb_tok = AutoTokenizer.from_pretrained(kb_path)
    kb_model = AutoModelForSeq2SeqLM.from_pretrained(kb_path).to("cuda").eval()
    log(f"KeyBART 로드: {kb_path}")

    SEP = "<kp_sep>" if args.keybart == "finetuned" else ";"

    @torch.no_grad()
    def gen_candidates(texts, num_beams=8, num_return=4, batch=6):
        out_all = []
        for s in range(0, len(texts), batch):
            enc = kb_tok(texts[s : s + batch], truncation=True, max_length=512,
                         padding=True, return_tensors="pt").to("cuda")
            o = kb_model.generate(**enc, num_beams=num_beams, num_return_sequences=num_return,
                                  max_new_tokens=48, output_scores=True, return_dict_in_generate=True)
            seqs = kb_tok.batch_decode(o.sequences, skip_special_tokens=(args.keybart == "zeroshot"))
            sc = o.sequences_scores.cpu().tolist()
            for d in range(len(texts[s : s + batch])):
                c2s = {}
                for r in range(num_return):
                    idx = d * num_return + r
                    raw = seqs[idx]
                    for tk in ["<present>", "<absent>", "<s>", "</s>", "<pad>"]:
                        raw = raw.replace(tk, SEP)
                    for kp in raw.split(SEP):
                        kp = kp.strip().lower()
                        if kp and len(kp.split()) <= 6:
                            c2s[kp] = max(c2s.get(kp, -1e9), sc[idx])
                out_all.append(c2s)
            if (s // batch) % 20 == 0:
                log(f"  생성 {min(s + batch, len(texts))}/{len(texts)}")
        return out_all

    def build_records(dataset):
        texts = [get_text(e) for e in dataset]
        gen_maps = gen_candidates(texts)
        recs_all = []
        for ex, text, gmap in zip(dataset, texts, gen_maps):
            norm_text = normalize(text)
            ext = extractive_candidates(text)
            gold = {normalize(k) for k in ex["keyphrases"]}
            recs = []
            for c in ext | set(gmap):
                recs.append({"phrase": c,
                             "label": 1 if normalize(c) in gold else 0,
                             "is_present": normalize(c) in norm_text,
                             "gen_score": gmap.get(c, GEN_MISSING),
                             "source": "both" if (c in ext and c in gmap) else ("ext" if c in ext else "gen")})
            recs_all.append(recs)
        return recs_all

    log("train 후보 생성")
    train_recs = build_records(train_ds)
    log("test 후보 생성")
    test_recs = build_records(test_ds)
    del kb_model
    torch.cuda.empty_cache(); gc.collect()

    # ---------- 후보 pool의 absent 상한선 ----------
    n_abs_gold = sum(1 for e in train_ds for c in e["prmu"] if c in ("M", "U"))
    n_abs_hit = sum(1 for e, rs in zip(train_ds, train_recs)
                    for r in rs if r["label"] == 1 and not r["is_present"])
    ceiling = n_abs_hit / max(n_abs_gold, 1)
    log(f"후보 생성 단계 absent recall(상한선): {ceiling:.4f}  ({n_abs_hit}/{n_abs_gold})")

    # ---------- pairwise 학습 ----------
    enc_tok = AutoTokenizer.from_pretrained(ENCODER)
    rng = random.Random(42)
    pairs = []
    for ex, recs in zip(train_ds, train_recs):
        pmap = {normalize(k): t for k, t in zip(ex["keyphrases"], ex["prmu"])}
        text = get_text(ex)
        pos = [r for r in recs if r["label"] == 1]
        neg = [r for r in recs if r["label"] == 0]
        if not pos or not neg:
            continue
        for p in pos:
            w = PRMU_WEIGHT.get(pmap.get(normalize(p["phrase"])), 1.0)
            n_neg = 4 * (2 if not p["is_present"] else 1)
            for n in rng.sample(neg, min(n_neg, len(neg))):
                pairs.append((text, p, n, w))
    rng.shuffle(pairs)
    log(f"학습쌍 {len(pairs):,}개 (absent positive {sum(1 for _, p, _, _ in pairs if not p['is_present']):,})")

    class PairDS(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            return pairs[i]

    def collate(b):
        t, pos, neg, w = zip(*b)
        enc = enc_tok(list(t) + list(t), [x["phrase"] for x in pos] + [x["phrase"] for x in neg],
                      truncation="only_first", max_length=256, padding=True, return_tensors="pt")
        aux = torch.tensor([make_aux(x) for x in pos] + [make_aux(x) for x in neg], dtype=torch.float)
        return enc, aux, torch.tensor(w, dtype=torch.float)

    loader = DataLoader(PairDS(), batch_size=16, shuffle=True, collate_fn=collate)
    ranker = HybridRanker().to("cuda")
    opt = torch.optim.AdamW(ranker.parameters(), lr=2e-5)
    ranker.train()
    for ep in range(args.epochs):
        losses = []
        for step, (enc, aux, w) in enumerate(loader):
            enc = {k: v.to("cuda") for k, v in enc.items()}
            aux, w = aux.to("cuda"), w.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = ranker(enc["input_ids"], enc["attention_mask"], aux)
                B = w.shape[0]
                loss = (w * torch.clamp(MARGIN - (s[:B] - s[B:]), min=0)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
            if (step + 1) % 300 == 0:
                log(f"  ep{ep+1} {step+1}/{len(loader)} loss={np.mean(losses[-300:]):.4f}")
        log(f"epoch {ep+1} 평균 loss {np.mean(losses):.4f}")

    # ---------- 평가 ----------
    @torch.no_grad()
    def score_recs(text, recs, bs=64):
        ranker.eval()
        out = []
        for s in range(0, len(recs), bs):
            ch = recs[s : s + bs]
            enc = enc_tok([text] * len(ch), [r["phrase"] for r in ch], truncation="only_first",
                          max_length=256, padding=True, return_tensors="pt").to("cuda")
            aux = torch.tensor([make_aux(r) for r in ch], dtype=torch.float).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.extend(ranker(enc["input_ids"], enc["attention_mask"], aux).float().cpu().tolist())
        return out

    def freq_scores(text, recs):
        tl = text.lower()
        return [tl.count(r["phrase"].lower())
                + (1.0 if 0 <= tl.find(r["phrase"].lower()) < len(tl) * 0.3 else 0.0) for r in recs]

    rows, matrix, abs_matrix = [], [], []
    our_preds: list[list[str]] = []   # 우리 evaluator용 top-10 (원표기 유지)
    MODELS = ["Hybrid ranker", "Extractive-only", "KeyBART raw", "Frequency"]
    log("평가")
    for ex, recs in zip(test_ds, test_recs):
        if not recs:
            our_preds.append([])
            continue
        text = get_text(ex)
        nt = normalize(text)
        gold = [normalize(k) for k in ex["keyphrases"]]
        g_pres = [g for g in gold if g in nt]
        g_abs = [g for g in gold if g not in nt]
        sc = score_recs(text, recs)
        ranked = [r for _, r in sorted(zip(sc, recs), key=lambda x: -x[0])]
        our_preds.append([r["phrase"] for r in ranked[:10]])   # ← 우리 채점표용
        r_all = [normalize(r["phrase"]) for r in ranked]
        r_pres = [normalize(r["phrase"]) for r in ranked if r["is_present"]]
        r_abs = [normalize(r["phrase"]) for r in ranked if not r["is_present"]]
        rows.append({"f1@5": f1_at_k(r_all, gold, 5), "f1@10": f1_at_k(r_all, gold, 10),
                     "present_f1@10": f1_at_k(r_pres, g_pres, 10) if g_pres else None,
                     "absent_R@10": recall_at_k(r_abs, g_abs, 10),
                     "absent_R@50": recall_at_k(r_abs, g_abs, 50),
                     "absent_MAP": average_precision(r_abs, g_abs)})

        # Rank CI용 4개 모델 F1@10
        row = [f1_at_k(r_all, gold, 10)]
        ext_p = [(s, r) for s, r in zip(sc, recs) if r["source"] != "gen"]
        row.append(f1_at_k([normalize(r["phrase"]) for _, r in sorted(ext_p, key=lambda x: -x[0])], gold, 10))
        gen_p = [(r["gen_score"], r) for r in recs if r["source"] in ("gen", "both")]
        row.append(f1_at_k([normalize(r["phrase"]) for _, r in sorted(gen_p, key=lambda x: -x[0])], gold, 10))
        fs = freq_scores(text, recs)
        row.append(f1_at_k([normalize(r["phrase"]) for _, r in sorted(zip(fs, recs), key=lambda x: -x[0])], gold, 10))
        matrix.append(row)

        if g_abs:
            arow = [recall_at_k(r_abs, g_abs, 10) or 0.0]
            eo = [(s, r) for s, r in zip(sc, recs) if r["source"] != "gen" and not r["is_present"]]
            arow.append(recall_at_k([normalize(r["phrase"]) for _, r in sorted(eo, key=lambda x: -x[0])], g_abs, 10) or 0.0)
            go = [(r["gen_score"], r) for r in recs if r["source"] in ("gen", "both") and not r["is_present"]]
            arow.append(recall_at_k([normalize(r["phrase"]) for _, r in sorted(go, key=lambda x: -x[0])], g_abs, 10) or 0.0)
            fo = [(s, r) for s, r in zip(fs, recs) if not r["is_present"]]
            arow.append(recall_at_k([normalize(r["phrase"]) for _, r in sorted(fo, key=lambda x: -x[0])], g_abs, 10) or 0.0)
            abs_matrix.append(arow)

    df = pd.DataFrame(rows)
    print("\n=== 노트북 자체 지표 (Hybrid ranker) ===")
    print(df.mean(numeric_only=True).round(4).to_string())

    # ---------- 우리 채점표와 동일 조건 평가 (experiments.csv에 누적) ----------
    run_id = f"N1_notebook_zeroshot" if args.keybart == "zeroshot" else "N2_notebook_finetuned"
    run_id += f"_te{args.n_test}"
    our_texts = [plain_doc_text(e["title"], e["abstract"]) for e in test_ds]
    our_golds = [e["keyphrases"] for e in test_ds]
    our_prmus = [e["prmu"] for e in test_ds]
    macro = evaluate_corpus(our_preds, our_golds, all_prmu=our_prmus, all_doc_texts=our_texts)
    print("\n=== 우리 evaluator 기준 (P1~P7과 직접 비교 가능) ===")
    for k in ["F1@5", "F1@10", "F1@M", "present_F1@5", "absent_R@10", "MAP@10", "nDCG@10",
              "dup_ratio", "recall_P", "recall_U"]:
        if k in macro:
            print(f"  {k:>13}: {macro[k]:.4f}")

    cols = ["F1@5", "F1@10", "F1@M", "present_F1@5", "absent_R@10", "MAP@10", "nDCG@10",
            "dup_ratio", "num_pred", "recall_P", "recall_R", "recall_M", "recall_U"]
    ExperimentLogger().log(
        run_id=run_id, model=f"2차 노트북 (SciBERT+aux, KeyBART {args.keybart})",
        input="T+A", decoder="Beam8x4", reranker="SciBERT+aux (PRMU-w pairwise)", MMR="No",
        train_subset=args.n_train, seed=42, num_docs=macro["num_docs"],
        **{k: round(macro[k], 4) for k in cols if k in macro})
    save_jsonl([{"id": e["id"], "title": e["title"], "gold": g, "prmu": p, "pred": pr}
                for e, g, p, pr in zip(test_ds, our_golds, our_prmus, our_preds)],
               OUT / "predictions" / f"{run_id}.jsonl")
    log(f"기록: experiments.csv [{run_id}] + predictions/{run_id}.jsonl")

    def rank_ci(mat, names, alpha=0.05):
        n = len(names)
        pv = {}
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                d = mat[:, j] - mat[:, k]
                if np.allclose(d, 0):
                    pv[(j, k)] = 1.0
                    continue
                t, p2 = stats.ttest_1samp(d, 0)
                pv[(j, k)] = p2 / 2 if t > 0 else 1 - p2 / 2
        keys = list(pv)
        rej = dict(zip(keys, multipletests([pv[k] for k in keys], alpha=alpha, method="holm")[0]))
        return {names[j]: (1 + sum(1 for k in range(n) if k != j and rej[(j, k)]),
                           n - sum(1 for k in range(n) if k != j and rej[(k, j)])) for j in range(n)}

    mat = np.array(matrix)
    print("\n=== 모델별 평균 F1@10 ===")
    for nm, m in zip(MODELS, mat.mean(axis=0)):
        print(f"  {nm:<18}: {m:.4f}")
    ci = rank_ci(mat, MODELS)
    print("Rank CI (F1@10):", ci)

    amat = np.array(abs_matrix)
    print(f"\n=== Absent R@10 (absent 정답 있는 {len(amat)}문서) ===")
    for nm, m in zip(MODELS, amat.mean(axis=0)):
        print(f"  {nm:<18}: {m:.4f}")
    aci = rank_ci(amat, MODELS)
    print("Absent Rank CI:", aci)

    save_json({
        "config": vars(args), "absent_candidate_ceiling": round(ceiling, 4),
        "hybrid_metrics": {k: (round(v, 4) if pd.notna(v) else None)
                           for k, v in df.mean(numeric_only=True).items()},
        "f1@10_by_model": dict(zip(MODELS, [round(float(x), 4) for x in mat.mean(axis=0)])),
        "rank_ci_f1@10": {k: list(v) for k, v in ci.items()},
        "absent_r@10_by_model": dict(zip(MODELS, [round(float(x), 4) for x in amat.mean(axis=0)])),
        "rank_ci_absent": {k: list(v) for k, v in aci.items()},
    }, OUT / "metrics" / f"notebook2_{tag}.json")
    log(f"저장: outputs/metrics/notebook2_{tag}.json")


if __name__ == "__main__":
    main()
