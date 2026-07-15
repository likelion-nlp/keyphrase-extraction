"""인코더 토너먼트 + 최종 앙상블 (docs/ENCODER_ABLATION_PLAN.md 구현).

P7 방법론(aux 피처 + PRMU-가중 pairwise margin 0.3) 고정, 인코더만 교체:
    P8  allenai/specter2_base                    (과학 특화, CLS)
    P9  microsoft/deberta-v3-base                (범용 클래식, CLS)
    P10 Alibaba-NLP/gte-reranker-modernbert-base (2026 리랭킹 상위권, CLS)
    P11 Qwen/Qwen3-Reranker-0.6B                 (LLM 혈통, last-token)
마지막에 P13 = RRF 앙상블 (멤버 선택은 validation 홀드아웃 1,000편).

단계:
    python scripts/encoder_tournament.py --stage smoke
    python scripts/encoder_tournament.py --stage run --run-id P8
    python scripts/encoder_tournament.py --stage p7scores          # P7 전 후보 점수 npz
    python scripts/encoder_tournament.py --stage ensdev            # 앙상블 홀드아웃 후보 구축
    python scripts/encoder_tournament.py --stage ensemble          # P13 선택+평가
    python scripts/encoder_tournament.py --stage grand             # 전 모델 통합 CSV
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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.data import build_source, load_kp20k, plain_doc_text
from src.metrics import evaluate_corpus, precision_recall_f1_at_k
from src.preprocessing import classify_prmu, normalize_phrase, stem_tokens
from src.utils import ExperimentLogger, Timer, load_jsonl, save_json, save_jsonl, set_seed
from src.utils import _json_default

OUT = PROJECT_ROOT / "outputs"
RER = OUT / "reranker"
SCORES = RER / "scores"
PRMU_WEIGHT = {"P": 1.0, "R": 1.5, "M": 2.0, "U": 2.5}
GEN_MISSING, MARGIN, MAX_LEN, N_AUX = -5.0, 0.3, 256, 3
T0 = time.time()

ROSTER = {
    "P8": {"encoder": "allenai/specter2_base", "pooling": "cls", "batch": 16, "accum": 1},
    # DeBERTa-v3: bf16에서 분리 어텐션 NaN 발산(알려진 이슈) → lr 인하 + clip
    "P9": {"encoder": "microsoft/deberta-v3-base", "pooling": "cls", "batch": 16, "accum": 1,
           "lr": 1e-5, "clip": 1.0},
    "P10": {"encoder": "Alibaba-NLP/gte-reranker-modernbert-base", "pooling": "cls", "batch": 16, "accum": 1},
    "P11": {"encoder": "Qwen/Qwen3-Reranker-0.6B", "pooling": "last", "batch": 8, "accum": 2,
            "grad_ckpt": True},
    # Gemma 계열 (게이트 모델 — HF 라이선스 동의 + 토큰 로그인 후 실행 가능)
    "P12": {"encoder": "google/embeddinggemma-300m", "pooling": "mean", "batch": 8, "accum": 2,
            "grad_ckpt": True},
    # 과학 도메인 특화 추가 분과 (앙상블 다양성 강화)
    "P14": {"encoder": "KISTI-AI/scideberta-cs", "pooling": "cls", "batch": 16, "accum": 1},
    "P15": {"encoder": "allenai/cs_roberta_base", "pooling": "cls", "batch": 16, "accum": 1},
}
ENSEMBLE_POOL = ["P2", "P7", "P8", "P9", "P10", "P11", "P12", "P14", "P15"]


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ================================================================ 모델
class HybridRanker(nn.Module):
    """인코더 + aux 3피처 → MLP 스칼라 (P7 구조, 인코더·풀링만 가변)."""

    def __init__(self, encoder_name: str, pooling: str = "cls"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.pooling = pooling
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h + N_AUX, 128), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask, aux):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            h = out[:, 0]
        elif self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(out.dtype)
            h = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        elif self.pooling == "last":
            idx = attention_mask.sum(1) - 1
            h = out[torch.arange(out.shape[0], device=out.device), idx]
        else:
            raise ValueError(self.pooling)
        return self.head(torch.cat([h, aux.to(out.dtype)], dim=-1)).squeeze(-1).float()


def make_aux(gen_score, has_gen: bool, is_present: bool) -> list[float]:
    g = GEN_MISSING if gen_score is None else float(gen_score)
    return [g / 5.0, 1.0 if has_gen else 0.0, 1.0 if is_present else 0.0]


def get_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ================================================================ 데이터 재구성 (P7과 동일)
def build_records(pool_file: Path, ext_file: Path, rows, texts):
    from src.reranking import merge_candidates

    gen_pool = {r["id"]: r["phrases"] for r in load_jsonl(pool_file)}
    ext_pool = {r["id"]: [(p["t"], p["s"]) for p in r["kb"]] for r in load_jsonl(ext_file)}
    out = []
    for i, r in enumerate(rows):
        merged = merge_candidates(ext_pool[r["id"]], gen_pool[r["id"]])
        ds = stem_tokens(texts[i])
        out.append([{
            "phrase": c["phrase"], "gen_score": c["gen_score"],
            "source": "both" if len(c["sources"]) == 2 else c["sources"][0][:3],
            "is_present": classify_prmu(c["phrase"], "", ds) == "P",
        } for c in merged])
    return out


def build_train_pairs():
    """P7과 완전히 동일한 학습쌍 (seed 42 고정 → 모델 간 동일 데이터)."""
    val = list(load_kp20k(["validation"], subset_sizes={"validation": 10_000}, seed=42)["validation"])
    texts = [plain_doc_text(r["title"], r["abstract"]) for r in val]
    recs_list = build_records(RER / "val_pool.jsonl", RER / "val_keybert.jsonl", val, texts)
    rng = random.Random(42)
    pairs = []
    for i, (row, recs) in enumerate(zip(val, recs_list)):
        pmap = {normalize_phrase(k): t for k, t in zip(row["keyphrases"], row["prmu"])}
        gold = set(pmap)
        pos = [r for r in recs if normalize_phrase(r["phrase"]) in gold]
        neg = [r for r in recs if normalize_phrase(r["phrase"]) not in gold]
        if not pos or not neg:
            continue
        for p in pos:
            w = PRMU_WEIGHT.get(pmap.get(normalize_phrase(p["phrase"]), "P"), 1.0)
            n_neg = 4 * (2 if not p["is_present"] else 1)
            for n in rng.sample(neg, min(n_neg, len(neg))):
                pairs.append((i, p, n, w))
    rng.shuffle(pairs)
    return texts, pairs


def load_test_assets():
    test = list(load_kp20k(["test"], subset_sizes={"test": 20_000}, seed=42)["test"])
    texts = [plain_doc_text(r["title"], r["abstract"]) for r in test]
    saved = load_jsonl(OUT / "candidates" / "fused_candidates_full.jsonl")
    assert [r["id"] for r in test] == [r["id"] for r in saved]
    gen_pool = {r["id"]: {c["text"]: c["gen_score"] for c in r["phrases"]}
                for r in load_jsonl(OUT / "candidates" / "keybart_full_beam10.jsonl")}
    recs_list, fusion_scores, sources_map = [], [], []
    for i, r in enumerate(saved):
        ds = stem_tokens(texts[i])
        gmap = gen_pool.get(r["id"], {})
        recs, fsc, smap = [], [], {}
        for c in r["candidates"]:
            ph = c["phrase"]
            srcs = c.get("sources", [])
            recs.append({"phrase": ph, "gen_score": gmap.get(ph),
                         "source": "both" if len(srcs) == 2 else (srcs[0][:3] if srcs else "ext"),
                         "is_present": classify_prmu(ph, "", ds) == "P"})
            fsc.append(float(c["final_score"]))
            smap[ph] = "+".join(sorted(srcs)) if srcs else ""
        recs_list.append(recs)
        fusion_scores.append(fsc)
        sources_map.append(smap)
    return test, texts, recs_list, fusion_scores, sources_map


# ================================================================ CSV 내보내기 (기존 포맷)
def export_csvs(run_id: str, test, texts, preds, score_lookup, sources_map):
    doc_rows, long_rows = [], []
    stem_cache = [stem_tokens(t) for t in texts]
    for i, r in enumerate(test):
        gold, prmu, pr = r["keyphrases"], r["prmu"], preds[i]
        gn = {normalize_phrase(g) for g in gold}
        marked = ["[O] " + p if normalize_phrase(p) in gn else p for p in pr]
        m5 = precision_recall_f1_at_k(pr, gold, 5)
        m10 = precision_recall_f1_at_k(pr, gold, 10)
        doc_rows.append({"id": r["id"], "title": r["title"], "gold": "; ".join(gold),
                         "gold_prmu": "".join(prmu), "pred_top10": "; ".join(marked),
                         "tp@10": m10["tp"], "F1@5": round(m5["f1"], 4),
                         "R@10": round(m10["recall"], 4)})
        for rank, p in enumerate(pr, 1):
            tag = classify_prmu(p, "", stem_cache[i])
            long_rows.append({"id": r["id"], "title": r["title"], "rank": rank, "keyphrase": p,
                              "type": "present" if tag == "P" else "absent", "prmu": tag,
                              "score": round(float(score_lookup[i].get(p, 0.0)), 4),
                              "source": sources_map[i].get(p, ""),
                              "is_correct": int(normalize_phrase(p) in gn),
                              "gold_all": "; ".join(gold)})
    pd.DataFrame(doc_rows).to_csv(OUT / "predictions" / f"kp20k_test_{run_id}_full.csv",
                                  index=False, encoding="utf-8-sig")
    pd.DataFrame(long_rows).to_csv(OUT / "predictions" / f"kp20k_test_{run_id}_full_keyphrases.csv",
                                   index=False, encoding="utf-8-sig")
    log(f"  CSV 저장: kp20k_test_{run_id}_full(.csv/_keyphrases.csv)")


def evaluate_and_record(run_id: str, model_label: str, reranker_label: str,
                        test, texts, recs_list, scores_docs, sources_map):
    preds, lookups = [], []
    for recs, sc in zip(recs_list, scores_docs):
        order = np.argsort(-np.asarray(sc))[:10]
        preds.append([recs[j]["phrase"] for j in order])
        lookups.append({recs[j]["phrase"]: sc[j] for j in range(len(recs))})
    golds = [r["keyphrases"] for r in test]
    prmus = [r["prmu"] for r in test]
    macro = evaluate_corpus(preds, golds, all_prmu=prmus, all_doc_texts=texts)
    cols = ["F1@5", "F1@10", "F1@M", "present_F1@5", "absent_R@10", "MAP@10", "nDCG@10",
            "dup_ratio", "num_pred", "recall_P", "recall_R", "recall_M", "recall_U"]
    ExperimentLogger().log(run_id=f"{run_id}_full" if not run_id.endswith("_full") else run_id,
                           model=model_label, input="T+A", decoder="Beam10",
                           reranker=reranker_label, MMR="No", seed=42, num_docs=macro["num_docs"],
                           **{k: round(macro[k], 4) for k in cols if k in macro})
    save_jsonl([{"id": r["id"], "title": r["title"], "gold": g, "prmu": p, "pred": pr}
                for r, g, p, pr in zip(test, golds, prmus, preds)],
               OUT / "predictions" / f"{run_id}_full.jsonl")
    export_csvs(run_id, test, texts, preds, lookups, sources_map)
    log(f"✓ {run_id}: F1@5={macro['F1@5']:.4f} A-R@10={macro.get('absent_R@10', 0):.4f} "
        f"recall_U={macro.get('recall_U', 0):.4f} nDCG@10={macro['nDCG@10']:.4f}")
    return macro


# ================================================================ stage: smoke
def stage_smoke() -> None:
    set_seed(42)
    for rid, cfg in ROSTER.items():
        name = cfg["encoder"]
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            tok = get_tokenizer(name)
            enc = tok(["doc text about neural networks"] * 4, ["neural network"] * 4,
                      truncation="only_first", max_length=64, padding=True, return_tensors="pt")
            model = HybridRanker(name, cfg["pooling"]).to("cuda")
            if cfg.get("grad_ckpt"):
                model.encoder.gradient_checkpointing_enable()
            opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
            aux = torch.zeros(4, N_AUX)
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s = model(enc["input_ids"].to("cuda"), enc["attention_mask"].to("cuda"),
                              aux.to("cuda"))
                    loss = s.mean()
                opt.zero_grad(); loss.backward(); opt.step()
            n_p = sum(p.numel() for p in model.parameters()) / 1e6
            vram = torch.cuda.max_memory_allocated() / 1024**3
            log(f"SMOKE OK {rid} {name} | {n_p:.0f}M | pooling={cfg['pooling']} | VRAM {vram:.2f}GB")
            del model, opt
        except Exception as e:
            log(f"SMOKE FAIL {rid} {name} — {type(e).__name__}: {str(e)[:150]}")
        torch.cuda.empty_cache(); gc.collect()


# ================================================================ stage: run (모델 1개)
def stage_run(run_id: str) -> None:
    cfg = ROSTER[run_id]
    set_seed(42)
    log(f"=== {run_id}: {cfg['encoder']} (pooling={cfg['pooling']}) ===")
    tok = get_tokenizer(cfg["encoder"])
    texts, pairs = build_train_pairs()
    log(f"학습쌍 {len(pairs):,}개 (P7과 동일)")

    class PairDS(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, k):
            di, p, n, w = pairs[k]
            return texts[di], p, n, w

    def collate(b):
        t, pos, neg, w = zip(*b)
        enc = tok(list(t) + list(t), [x["phrase"] for x in pos] + [x["phrase"] for x in neg],
                  truncation="only_first", max_length=MAX_LEN, padding=True, return_tensors="pt")
        aux = torch.tensor([make_aux(x["gen_score"], x["source"] in ("gen", "both"), x["is_present"])
                            for x in list(pos) + list(neg)], dtype=torch.float)
        return enc, aux, torch.tensor(w, dtype=torch.float)

    model = HybridRanker(cfg["encoder"], cfg["pooling"]).to("cuda")
    if cfg.get("grad_ckpt"):
        model.encoder.gradient_checkpointing_enable()
        log("gradient checkpointing ON")
    loader = DataLoader(PairDS(), batch_size=cfg["batch"], shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-5))
    accum = cfg.get("accum", 1)
    clip = cfg.get("clip")
    model.train()
    nan_streak = 0
    for ep in range(2):
        losses = []
        opt.zero_grad()
        for step, (enc, aux, w) in enumerate(loader):
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model(enc["input_ids"], enc["attention_mask"], aux.to("cuda"))
                B = w.shape[0]
                loss = (w.to("cuda") * torch.clamp(MARGIN - (s[:B] - s[B:]), min=0)).mean() / accum
            if not torch.isfinite(loss):
                nan_streak += 1
                opt.zero_grad()
                if nan_streak >= 20:
                    raise RuntimeError("loss NaN 연속 20회 — 학습 발산, 설정(lr/clip) 조정 필요")
                continue
            nan_streak = 0
            loss.backward()
            if (step + 1) % accum == 0:
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step(); opt.zero_grad()
            losses.append(float(loss) * accum)
            if (step + 1) % 500 == 0:
                log(f"  ep{ep + 1} {step + 1}/{len(loader)} loss={np.mean(losses[-500:]):.4f}")
        log(f"epoch {ep + 1} 평균 loss {np.mean(losses):.4f}")
    ck = OUT / "checkpoints" / f"reranker_{run_id}"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck / "model.pt")
    tok.save_pretrained(str(ck))
    save_json({"encoder": cfg["encoder"], "pooling": cfg["pooling"]}, ck / "tournament_meta.json")

    # ---- test 전 후보 스코어링 + npz 저장 + 평가/CSV ----
    test, t_texts, recs_list, _, sources_map = load_test_assets()
    scores_docs = score_all(model, tok, t_texts, recs_list, batch=256 if cfg["batch"] >= 16 else 128)
    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCORES / f"{run_id}.npz",
                        scores=np.concatenate([np.asarray(s, dtype=np.float32) for s in scores_docs]),
                        lens=np.array([len(s) for s in scores_docs]))
    del model
    torch.cuda.empty_cache(); gc.collect()
    evaluate_and_record(run_id, f"Tournament {cfg['encoder'].split('/')[-1]}",
                        f"{cfg['encoder'].split('/')[-1]}+aux (PRMU-w pairwise)",
                        test, t_texts, recs_list, scores_docs, sources_map)


@torch.no_grad()
def score_all(model, tok, texts, recs_list, batch=256):
    model.eval()
    flat = [(i, r["phrase"], r) for i, recs in enumerate(recs_list) for r in recs]
    sc = np.zeros(len(flat), dtype=np.float32)
    for s in range(0, len(flat), batch):
        ch = flat[s : s + batch]
        enc = tok([texts[i] for i, _, _ in ch], [p for _, p, _ in ch], truncation="only_first",
                  max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        aux = torch.tensor([make_aux(r["gen_score"], r["source"] in ("gen", "both"),
                                     r["is_present"]) for _, _, r in ch],
                           dtype=torch.float).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sc[s : s + len(ch)] = model(enc["input_ids"], enc["attention_mask"], aux).cpu().numpy()
        if (s // batch) % 400 == 0:
            log(f"  scoring {s:,}/{len(flat):,}")
    out, k = [], 0
    for recs in recs_list:
        out.append(sc[k : k + len(recs)].tolist())
        k += len(recs)
    return out


def load_ranker_from_ckpt(ck: Path):
    meta_p = ck / "tournament_meta.json"
    if meta_p.exists():
        meta = json.load(open(meta_p, encoding="utf-8"))
        enc_name, pooling = meta["encoder"], meta["pooling"]
    else:  # P7 (SciBERT)
        enc_name, pooling = "allenai/scibert_scivocab_uncased", "cls"
    model = HybridRanker(enc_name, pooling).to("cuda")
    model.load_state_dict(torch.load(ck / "model.pt"))
    tok = get_tokenizer(str(ck))
    return model, tok


# ================================================================ stage: p7scores
def stage_p7scores() -> None:
    set_seed(42)
    test, texts, recs_list, _, _ = load_test_assets()
    model, tok = load_ranker_from_ckpt(OUT / "checkpoints" / "reranker_scibert")
    scores_docs = score_all(model, tok, texts, recs_list)
    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCORES / "P7.npz",
                        scores=np.concatenate([np.asarray(s, dtype=np.float32) for s in scores_docs]),
                        lens=np.array([len(s) for s in scores_docs]))
    log("P7 전 후보 점수 저장 완료")


# ================================================================ stage: ensdev
def stage_ensdev() -> None:
    """앙상블 홀드아웃: validation 셔플 10,000~11,000번째 문서의 후보 구축."""
    from transformers import AutoModelForSeq2SeqLM

    from src.extraction import KeyBertExtractor
    from src.generation import generate_keyphrases

    set_seed(42)
    val_all = load_kp20k(["validation"], seed=42)["validation"].shuffle(seed=42)
    dev = list(val_all.select(range(10_000, 11_000)))
    texts = [plain_doc_text(r["title"], r["abstract"]) for r in dev]
    sources = [build_source(r["title"], r["abstract"]) for r in dev]

    pool_file = RER / "ensdev_pool.jsonl"
    done = len(load_jsonl(pool_file)) if pool_file.exists() else 0
    if done < len(dev):
        ck = OUT / "checkpoints" / "keybart_full"
        kb_tok = AutoTokenizer.from_pretrained(str(ck))
        kb = AutoModelForSeq2SeqLM.from_pretrained(str(ck)).to("cuda").eval()
        with open(pool_file, "a", encoding="utf-8") as pf:
            for s in range(done, len(dev), 200):
                gen = generate_keyphrases(kb, kb_tok, sources[s : s + 200], strategy="beam10",
                                          max_source_length=384, batch_size=4)
                for r, doc in zip(dev[s : s + 200], gen):
                    pf.write(json.dumps({"id": r["id"], "phrases": [
                        {"text": c["text"], "gen_score": c["gen_score"]} for c in doc]},
                        ensure_ascii=False, default=_json_default) + "\n")
                pf.flush()
                torch.cuda.empty_cache()
                log(f"  ensdev pool {min(s + 200, len(dev))}/{len(dev)}")
        del kb
        torch.cuda.empty_cache(); gc.collect()

    ext_file = RER / "ensdev_keybert.jsonl"
    if not ext_file.exists():
        ex = KeyBertExtractor(device="cuda")
        kb30 = ex.extract_batch(texts, top_n=30)
        save_jsonl([{"id": r["id"], "kb": [{"t": p, "s": sc} for p, sc in doc]}
                    for r, doc in zip(dev, kb30)], ext_file)
    save_jsonl([{"id": r["id"], "title": r["title"], "keyphrases": r["keyphrases"],
                 "prmu": r["prmu"], "abstract": r["abstract"]} for r in dev],
               RER / "ensdev_docs.jsonl")
    log("ensdev 후보 구축 완료 (1,000편)")


# ================================================================ stage: ensemble
def stage_ensemble() -> None:
    from itertools import combinations

    from src.reranking import SemanticScorer, fuse_scores, merge_candidates

    set_seed(42)
    dev = load_jsonl(RER / "ensdev_docs.jsonl")
    d_texts = [plain_doc_text(r["title"], r["abstract"]) for r in dev]
    d_recs = build_records(RER / "ensdev_pool.jsonl", RER / "ensdev_keybert.jsonl", dev, d_texts)
    d_golds = [r["keyphrases"] for r in dev]

    # 멤버별 dev 점수
    dev_scores: dict[str, list[list[float]]] = {}
    scorer = SemanticScorer(device="cuda")
    gen_pool = {r["id"]: r["phrases"] for r in load_jsonl(RER / "ensdev_pool.jsonl")}
    ext_pool = {r["id"]: [(p["t"], p["s"]) for p in r["kb"]]
                for r in load_jsonl(RER / "ensdev_keybert.jsonl")}
    log("P2(fusion) dev 점수")
    p2 = []
    for i, r in enumerate(dev):
        merged = merge_candidates(ext_pool[r["id"]], gen_pool[r["id"]])
        ranked = fuse_scores(merged, d_texts[i], r["title"], scorer)
        smap = {c["phrase"]: c["final_score"] for c in ranked}
        p2.append([smap.get(x["phrase"], 0.0) for x in d_recs[i]])
    dev_scores["P2"] = p2

    ckpts = {"P7": OUT / "checkpoints" / "reranker_scibert"}
    for rid in ROSTER:
        ckpts[rid] = OUT / "checkpoints" / f"reranker_{rid}"
    for rid, ck in ckpts.items():
        if not (ck / "model.pt").exists():
            log(f"skip {rid} (체크포인트 없음)")
            continue
        log(f"{rid} dev 점수")
        model, tok = load_ranker_from_ckpt(ck)
        dev_scores[rid] = score_all(model, tok, d_texts, d_recs, batch=128)
        del model
        torch.cuda.empty_cache(); gc.collect()

    members_avail = [m for m in ENSEMBLE_POOL if m in dev_scores]
    log(f"앙상블 후보 멤버: {members_avail}")

    def rrf_predict(recs_list, member_scores, members, top_k=10):
        preds = []
        for d in range(len(recs_list)):
            n = len(recs_list[d])
            if n == 0:
                preds.append([]); continue
            rrf = np.zeros(n)
            for m in members:
                order = np.argsort(-np.asarray(member_scores[m][d]))
                ranks = np.empty(n)
                ranks[order] = np.arange(1, n + 1)
                rrf += 1.0 / (60 + ranks)
            top = np.argsort(-rrf)[:top_k]
            preds.append([recs_list[d][j]["phrase"] for j in top])
        return preds

    # dev에서 조합 탐색 (크기 2~4) — P12(Gemma)는 필수 멤버 (사용자 지정)
    required = "P12" if "P12" in members_avail else None
    if required is None:
        log("경고: P12(Gemma) dev 점수 없음 — 필수 제약 없이 탐색")
    results = []
    for k in (2, 3, 4):
        for combo in combinations(members_avail, k):
            if required is not None and required not in combo:
                continue
            preds = rrf_predict(d_recs, dev_scores, combo)
            f1 = evaluate_corpus(preds, d_golds, ks=(5,))["F1@5"]
            results.append((f1, combo))
    singles = []  # 단독 참조 (선택 대상 아님 — 비교 로그·기록용)
    for m in members_avail:
        preds = rrf_predict(d_recs, dev_scores, (m,))
        singles.append((evaluate_corpus(preds, d_golds, ks=(5,))["F1@5"], (m,)))
    results.sort(reverse=True)
    singles.sort(reverse=True)
    for f1, combo in results[:8]:
        log(f"  [dev] {'+'.join(combo):<20} F1@5={f1:.4f}")
    for f1, (m,) in singles[:3]:
        log(f"  [dev/단독참조] {m:<14} F1@5={f1:.4f}")
    best_combo = results[0][1]
    save_json({"dev_results": [{"members": list(c), "F1@5": round(f, 4)} for f, c in results],
               "dev_singles": [{"member": c[0], "F1@5": round(f, 4)} for f, c in singles],
               "required_member": required,
               "best": list(best_combo)}, OUT / "metrics" / "ensemble_selection.json")
    log(f"선택된 앙상블: {best_combo}")

    # ---- test 적용 ----
    test, t_texts, t_recs, t_fusion, t_sources = load_test_assets()
    t_scores = {"P2": t_fusion}
    for rid in best_combo:
        if rid == "P2":
            continue
        z = np.load(SCORES / f"{rid}.npz")
        flat, lens = z["scores"], z["lens"]
        out, k = [], 0
        for L in lens:
            out.append(flat[k : k + L].tolist()); k += L
        t_scores[rid] = out
    preds = rrf_predict(t_recs, t_scores, best_combo)
    lookups = []
    for d in range(len(t_recs)):
        n = len(t_recs[d])
        rrf = np.zeros(n)
        for m in best_combo:
            order = np.argsort(-np.asarray(t_scores[m][d]))
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            rrf += 1.0 / (60 + ranks)
        lookups.append({t_recs[d][j]["phrase"]: float(rrf[j]) for j in range(n)})
    golds = [r["keyphrases"] for r in test]
    prmus = [r["prmu"] for r in test]
    macro = evaluate_corpus(preds, golds, all_prmu=prmus, all_doc_texts=t_texts)
    cols = ["F1@5", "F1@10", "F1@M", "present_F1@5", "absent_R@10", "MAP@10", "nDCG@10",
            "dup_ratio", "num_pred", "recall_P", "recall_R", "recall_M", "recall_U"]
    ExperimentLogger().log(run_id="P13_ensemble_full", model=f"RRF({'+'.join(best_combo)})",
                           input="T+A", decoder="Beam10", reranker="RRF ensemble", MMR="No",
                           seed=42, num_docs=macro["num_docs"],
                           **{k: round(macro[k], 4) for k in cols if k in macro})
    save_jsonl([{"id": r["id"], "title": r["title"], "gold": g, "prmu": p, "pred": pr}
                for r, g, p, pr in zip(test, golds, prmus, preds)],
               OUT / "predictions" / "P13_ensemble_full.jsonl")
    export_csvs("P13_ensemble", test, t_texts, preds, lookups, t_sources)
    log(f"✓ P13_ensemble({'+'.join(best_combo)}): F1@5={macro['F1@5']:.4f} "
        f"A-R@10={macro.get('absent_R@10', 0):.4f} nDCG@10={macro['nDCG@10']:.4f}")


# ================================================================ stage: grand
def stage_grand() -> None:
    exp = pd.read_csv(OUT / "metrics" / "experiments.csv")
    order = ["B0_tfidf_full", "B1_keybert_full", "B2_keybert_mmr_full", "B3_bart_beam5_full",
             "B4_keybart_beam5_full", "P1_hybrid_fusion_full", "P2_hybrid_fusion_mmr_full",
             "P5_pairwise_full", "P6_pairwise_prmu_full", "P7_scibert_hybrid_full",
             "P8_full", "P9_full", "P10_full", "P11_full", "P12_full", "P14_full",
             "P15_full", "P13_ensemble_full", "P16_team_ensemble_full",
             "N1_notebook_zeroshot_te2000"]
    d = exp[exp.run_id.isin(order)].copy()
    d["__o"] = d.run_id.map({r: i for i, r in enumerate(order)})
    d = d.sort_values("__o")
    cols = ["run_id", "model", "F1@5", "F1@10", "F1@M", "present_F1@5", "absent_R@10",
            "MAP@10", "nDCG@10", "dup_ratio", "recall_P", "recall_R", "recall_M", "recall_U",
            "num_docs"]
    view = d[[c for c in cols if c in d.columns]]
    view.to_csv(OUT / "metrics" / "grand_comparison_all_models.csv",
                index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print()
    print(view.to_string(index=False))
    log("저장: outputs/metrics/grand_comparison_all_models.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "run", "p7scores", "ensdev", "ensemble", "grand"])
    ap.add_argument("--run-id", choices=list(ROSTER), default=None)
    args = ap.parse_args()
    set_seed(42)
    if args.stage == "smoke":
        stage_smoke()
    elif args.stage == "run":
        assert args.run_id, "--run-id 필요"
        stage_run(args.run_id)
    elif args.stage == "p7scores":
        stage_p7scores()
    elif args.stage == "ensdev":
        stage_ensdev()
    elif args.stage == "ensemble":
        stage_ensemble()
    elif args.stage == "grand":
        stage_grand()
    log("완료")
