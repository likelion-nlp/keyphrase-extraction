"""기본 실험 오케스트레이터 — 노트북 04~12와 동일 로직을 스크립트로 실행한다.

사용:
    python scripts/run_experiments.py --stage extract   # B0 TF-IDF, B1/B2 KeyBERT(±MMR)
    python scripts/run_experiments.py --stage bart      # B3 BART-base 학습+평가
    python scripts/run_experiments.py --stage keybart   # B4 KeyBART 학습+평가
    python scripts/run_experiments.py --stage hybrid    # P1 fusion, P2 +MMR (λ 스윕 포함)
    python scripts/run_experiments.py --stage table     # 최종 비교표 출력/저장
    python scripts/run_experiments.py --stage all

프로파일 (--profile, 기본 dev):
    dev  — Stage B: train 10k / eval test 2,000 (config yaml 그대로)
    full — Stage D: train 530,809 전체 / eval test 20,000 전체, 2 epochs.
           배치는 RTX 5080 16GB 프로빙 결과로 고정 (BART b32/acc1 238 samp/s,
           KeyBART b8/acc4/ckpt-off 71 samp/s; 생성은 batch 8 — 그 이상은
           Windows WDDM이 시스템 RAM으로 spill해 7배 느려짐).
           run_id·산출물 파일명에 "_full" 접미사가 붙어 dev 결과와 분리 보존된다.

모든 결과는 outputs/metrics/experiments.csv 에 run_id 기준 누적 기록된다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.utils import ExperimentLogger, Timer, load_config, save_json, save_jsonl, set_seed

OUT = PROJECT_ROOT / "outputs"
SEED = 42
T0 = time.time()

# --profile이 설정한다 (__main__ 참조)
PROFILE = "dev"
EVAL_N = 2000
SUFFIX = ""
KEYBERT_CHUNK = 5000          # 대규모 배치 시 vocabulary RAM 폭증 방지

# full 프로파일의 학습 오버라이드 (프로빙 근거, effective batch 32는 dev와 동일 유지)
FULL_TRAIN_OVERRIDES = {
    "bart": {"per_device_train_batch_size": 32, "gradient_accumulation_steps": 1,
             "gradient_checkpointing": False, "num_train_epochs": 2},
    "keybart": {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 4,
                "gradient_checkpointing": False, "num_train_epochs": 2},
}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def log_run(logger: ExperimentLogger, run_id: str, model: str, macro: dict, **extra) -> None:
    cols = [
        "F1@5", "F1@10", "F1@M", "present_F1@5", "absent_R@10",
        "MAP@10", "nDCG@10", "dup_ratio", "num_pred",
        "recall_P", "recall_R", "recall_M", "recall_U",
    ]
    row = {k: round(macro[k], 4) for k in cols if k in macro}
    logger.log(run_id=run_id, model=model, input="T+A",
               num_docs=macro.get("num_docs"), **row, **extra)
    log(f"  -> logged {run_id}: F1@5={macro.get('F1@5', 0):.4f} F1@M={macro.get('F1@M', 0):.4f} "
        f"P-F1@5={macro.get('present_F1@5', 0):.4f} A-R@10={macro.get('absent_R@10', 0):.4f} "
        f"dup={macro.get('dup_ratio', 0):.4f}")


def load_eval_set():
    from src.data import load_kp20k, plain_doc_text

    test = load_kp20k(["test"], subset_sizes={"test": EVAL_N}, seed=SEED)["test"]
    rows = list(test)
    texts = [plain_doc_text(r["title"], r["abstract"]) for r in rows]
    golds = [r["keyphrases"] for r in rows]
    prmus = [r["prmu"] for r in rows]
    return rows, texts, golds, prmus


def save_preds(rows, golds, prmus, preds, filename: str) -> None:
    save_jsonl(
        [{"id": r["id"], "title": r["title"], "gold": g, "prmu": p, "pred": pr}
         for r, g, p, pr in zip(rows, golds, prmus, preds)],
        OUT / "predictions" / filename,
    )
    log(f"  -> saved predictions: {filename}")


# ---------------------------------------------------------------- stage: extract
def stage_extract() -> None:
    from src.data import load_kp20k, plain_doc_text
    from src.extraction import KeyBertExtractor, TfidfExtractor
    from src.metrics import evaluate_corpus

    cfg = load_config(PROJECT_ROOT / "configs" / "baseline.yaml")
    logger = ExperimentLogger()
    rows, texts, golds, prmus = load_eval_set()
    log(f"eval set: {len(rows)} docs")

    # B0 --------------------------------------------------------- TF-IDF
    log("B0 TF-IDF: vocabulary 구축")
    # full: 530k 전체로 1~3gram vocab을 만들면 pruning 전 중간 dict가 RAM을 터뜨린다
    # (플랜 Notebook 04 주의사항) → 100k subset + max_features로 제한
    fit_n = 100_000 if PROFILE == "full" else cfg["data"]["tfidf_fit_subset"]
    train_sub = load_kp20k(["train"], subset_sizes={"train": fit_n}, seed=SEED)["train"]
    fit_texts = [plain_doc_text(r["title"], r["abstract"]) for r in train_sub]
    tfidf = TfidfExtractor(
        ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
        max_features=cfg["tfidf"]["max_features"],
        min_df=cfg["tfidf"]["min_df"], max_df=cfg["tfidf"]["max_df"],
    )
    with Timer("tfidf fit"):
        tfidf.fit(fit_texts)
    with Timer("tfidf extract"):
        scored = tfidf.extract_batch(texts, top_n=cfg["tfidf"]["top_n"])
    preds = [[p for p, _ in doc] for doc in scored]
    macro = evaluate_corpus(preds, golds, all_prmu=prmus, all_doc_texts=texts)
    log_run(logger, f"B0_tfidf{SUFFIX}", "TF-IDF", macro, decoder="-", reranker="-", MMR="No",
            seed=SEED, tfidf_fit=fit_n)
    save_preds(rows, golds, prmus, preds, f"B0_tfidf{SUFFIX}.jsonl")

    # B1/B2 ------------------------------------------------------ KeyBERT
    def keybert_chunked(extractor, texts_, **kw):
        # 20k 문서를 한 번에 넣으면 KeyBERT가 코퍼스 전체 n-gram vocabulary를
        # 한꺼번에 임베딩한다 → chunk로 나눠 RAM을 상수로 유지 (결과는 문서별이라 동일)
        out = []
        for s in range(0, len(texts_), KEYBERT_CHUNK):
            out.extend(extractor.extract_batch(texts_[s : s + KEYBERT_CHUNK], **kw))
            if len(texts_) > KEYBERT_CHUNK:
                log(f"  keybert {min(s + KEYBERT_CHUNK, len(texts_))}/{len(texts_)}")
        return out

    log("B1 KeyBERT (MMR off)")
    extractor = KeyBertExtractor(model_name=cfg["keybert"]["model"], device="cuda")
    with Timer("keybert no-mmr"):
        scored_off = keybert_chunked(extractor, texts, top_n=cfg["keybert"]["top_n"], use_mmr=False)
    preds_off = [[p for p, _ in doc] for doc in scored_off]
    macro_off = evaluate_corpus(preds_off, golds, all_prmu=prmus, all_doc_texts=texts)
    log_run(logger, f"B1_keybert{SUFFIX}", "KeyBERT", macro_off, decoder="-", reranker="-", MMR="No", seed=SEED)
    save_preds(rows, golds, prmus, preds_off, f"B1_keybert{SUFFIX}.jsonl")

    log("B2 KeyBERT (MMR on, diversity=0.5)")
    with Timer("keybert mmr"):
        scored_on = keybert_chunked(
            extractor, texts, top_n=cfg["keybert"]["top_n"], use_mmr=True,
            diversity=cfg["keybert"]["diversity"],
        )
    preds_on = [[p for p, _ in doc] for doc in scored_on]
    macro_on = evaluate_corpus(preds_on, golds, all_prmu=prmus, all_doc_texts=texts)
    log_run(logger, f"B2_keybert_mmr{SUFFIX}", "KeyBERT+MMR", macro_on, decoder="-", reranker="-", MMR="Yes", seed=SEED)
    save_preds(rows, golds, prmus, preds_on, f"B2_keybert_mmr{SUFFIX}.jsonl")


# ---------------------------------------------------------------- stage: seq2seq 공통
def train_and_eval_seq2seq(config_name: str, run_id: str, model_label: str, ckpt_name: str) -> None:
    import pandas as pd

    from src.data import build_source, load_kp20k
    from src.generation import generate_keyphrases, load_seq2seq, make_seq2seq_dataset, train_seq2seq
    from src.metrics import evaluate_corpus

    cfg = load_config(PROJECT_ROOT / "configs" / f"{config_name}.yaml")
    logger = ExperimentLogger()
    set_seed(cfg["seed"])

    if PROFILE == "full":
        cfg["data"]["train_subset"] = None            # None → 530,809 전체
        cfg["training"].update(FULL_TRAIN_OVERRIDES[config_name])
        log(f"{run_id}: full 프로파일 오버라이드 적용 — {FULL_TRAIN_OVERRIDES[config_name]}")
    run_id = run_id + SUFFIX
    ckpt_name = ckpt_name + SUFFIX

    train_n = cfg["data"]["train_subset"]
    log(f"{run_id}: 데이터 로드 (train {'전체 530,809' if train_n is None else format(train_n, ',')})")
    ds = load_kp20k(
        ["train", "validation"],
        subset_sizes={"train": train_n, "validation": 300},
        seed=cfg["seed"],
    )
    MAX_SRC = cfg["tokenization"]["max_source_length"]
    MAX_TGT = cfg["tokenization"]["max_target_length"]

    log(f"{run_id}: 모델 로드 — {cfg['model_name']}")
    model, tokenizer = load_seq2seq(cfg["model_name"], device="cuda")

    with Timer("tokenize"):
        train_tok = make_seq2seq_dataset(ds["train"], tokenizer, MAX_SRC, MAX_TGT)
        val_tok = make_seq2seq_dataset(ds["validation"], tokenizer, MAX_SRC, MAX_TGT)

    ckpt_dir = OUT / "checkpoints" / ckpt_name
    trainer = train_seq2seq(
        model, tokenizer, train_tok, eval_dataset=val_tok,
        output_dir=str(ckpt_dir),
        learning_rate=float(cfg["training"]["learning_rate"]),
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False),
        seed=cfg["seed"],
        logging_steps=50,
        disable_tqdm=True,
    )
    eff_batch = cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]
    n_steps = int(len(train_tok) / eff_batch * cfg["training"]["num_train_epochs"])
    log(f"{run_id}: 학습 시작 — {len(train_tok):,} samples, effective batch {eff_batch}, ~{n_steps} steps")
    with Timer(f"{run_id} train"):
        result = trainer.train()
    log(f"{run_id}: train loss {result.training_loss:.4f}")
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    pd.DataFrame(trainer.state.log_history).to_json(
        OUT / "metrics" / f"{run_id}_log_history.json", orient="records", indent=1
    )

    # 평가 (공통 eval set)
    rows, texts, golds, prmus = load_eval_set()
    sources = [build_source(r["title"], r["abstract"]) for r in rows]
    strategy = cfg["decoding"]["strategy"]
    log(f"{run_id}: 생성 시작 — {strategy}, {len(sources)} docs")
    preds = []
    CHUNK = 200
    with Timer(f"{run_id} generate"):
        for start in range(0, len(sources), CHUNK):
            gen = generate_keyphrases(
                model, tokenizer, sources[start : start + CHUNK],
                strategy=strategy, max_source_length=MAX_SRC,
                max_new_tokens=cfg["decoding"]["max_new_tokens"],
                batch_size=16 if "base" in cfg["model_name"] else 8,
            )
            preds.extend([[g["text"] for g in doc] for doc in gen])
            log(f"  generated {min(start + CHUNK, len(sources))}/{len(sources)}")
    macro = evaluate_corpus(preds, golds, all_prmu=prmus, all_doc_texts=texts)
    log_run(logger, run_id, model_label, macro,
            decoder=strategy, reranker="-", MMR="No",
            train_subset=train_n if train_n is not None else 530_809, seed=cfg["seed"],
            epochs=cfg["training"]["num_train_epochs"])
    save_preds(rows, golds, prmus, preds, f"{run_id}.jsonl")


def stage_bart() -> None:
    train_and_eval_seq2seq("bart", "B3_bart_beam5", "BART", "bart_base")


def stage_keybart() -> None:
    train_and_eval_seq2seq("keybart", "B4_keybart_beam5", "KeyBART", "keybart")


# ---------------------------------------------------------------- stage: hybrid
def stage_hybrid() -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from src.data import build_source
    from src.diversity import mmr_select
    from src.extraction import KeyBertExtractor
    from src.generation import generate_keyphrases
    from src.metrics import candidate_recall, evaluate_corpus, semantic_redundancy
    from src.reranking import SemanticScorer, fuse_scores, merge_candidates

    logger = ExperimentLogger()
    rows, texts, golds, prmus = load_eval_set()
    titles = [r["title"] for r in rows]

    # 생성기: 현재 프로파일의 fine-tuned KeyBART 우선
    ckpt = next(
        (c for c in [OUT / "checkpoints" / f"keybart{SUFFIX}", OUT / "checkpoints" / f"bart_base{SUFFIX}"]
         if (c / "config.json").exists()),
        None,
    )
    assert ckpt is not None, "체크포인트가 없습니다 — bart/keybart stage를 먼저 실행하세요"
    log(f"hybrid: 생성기 로드 — {ckpt.name}")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to("cuda").eval()

    # 후보 pool: beam10 (모든 시퀀스 분해·통합)
    # 공유 VRAM 환경(브라우저 등과 동시 사용) 대응:
    #  - 청크마다 즉시 파일에 append → 크래시해도 생성분 보존, 재실행 시 이어서 진행
    #  - OOM 발생 시 배치를 절반으로 줄여 재시도 (4→2→1)
    import gc
    import json as _json

    from src.utils import _json_default, load_jsonl

    sources = [build_source(r["title"], r["abstract"]) for r in rows]
    CHUNK = 200
    pool_file = OUT / "candidates" / f"{ckpt.name}_beam10.jsonl"

    gen_pool: list[list[dict]] = []
    if pool_file.exists():
        existing = load_jsonl(pool_file)
        for i, r in enumerate(existing):
            assert r["id"] == rows[i]["id"], f"partial pool 정렬 불일치 (doc {i})"
        gen_pool = [r["phrases"] for r in existing]
        log(f"hybrid: 기존 pool {len(gen_pool)}건 발견 — 이어서 생성")

    def gen_chunk_with_retry(chunk_sources, batch_size=4):
        bs = batch_size
        while True:
            try:
                return generate_keyphrases(
                    model, tokenizer, chunk_sources,
                    strategy="beam10", max_source_length=384, batch_size=bs,
                )
            except Exception as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                gc.collect()
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                log(f"  OOM → batch {bs}로 축소 재시도")

    log(f"hybrid: beam10 후보 생성 — {len(sources)} docs (완료 {len(gen_pool)})")
    with Timer("beam10 pool"), open(pool_file, "a", encoding="utf-8") as pf:
        for start in range(len(gen_pool), len(sources), CHUNK):
            gen = gen_chunk_with_retry(sources[start : start + CHUNK])
            for r, g, p, doc in zip(rows[start : start + CHUNK],
                                    golds[start : start + CHUNK],
                                    prmus[start : start + CHUNK], gen):
                row = {"id": r["id"], "title": r["title"], "gold": g, "prmu": p,
                       "phrases": [{"text": c["text"], "gen_score": c["gen_score"],
                                     "seq_id": c.get("seq_id", 0)} for c in doc]}
                pf.write(_json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            pf.flush()
            gen_pool.extend(gen)
            torch.cuda.empty_cache()
            log(f"  pool {min(start + CHUNK, len(sources))}/{len(sources)}")

    # pool 완료 후 생성기 VRAM 반납 → 이후 단계(KeyBERT/SBERT)에 여유 확보
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 추출 후보: KeyBERT top30 (chunk 단위 — vocabulary RAM 상수 유지)
    log("hybrid: KeyBERT top30")
    extractor = KeyBertExtractor(device="cuda")
    ext_scored = []
    with Timer("keybert top30"):
        for s in range(0, len(texts), KEYBERT_CHUNK):
            ext_scored.extend(extractor.extract_batch(texts[s : s + KEYBERT_CHUNK], top_n=30))
            if len(texts) > KEYBERT_CHUNK:
                log(f"  keybert {min(s + KEYBERT_CHUNK, len(texts))}/{len(texts)}")

    # Candidate recall (플랜 11.2)
    ext_cands = [[p for p, _ in doc] for doc in ext_scored]
    gen_cands = [[c["text"] for c in doc] for doc in gen_pool]

    def interleave(a, b):
        # union recall@K는 순서 의존 — 한쪽을 앞에 몰면 @20이 그쪽 recall로 왜곡된다
        out = []
        for x, y in zip(a, b):
            out.extend([x, y])
        out.extend(a[len(b):] if len(a) > len(b) else b[len(a):])
        return out

    union_cands = [interleave(e, g) for e, g in zip(ext_cands, gen_cands)]
    cr = {
        "KeyBERT": candidate_recall(ext_cands, golds),
        "Generator": candidate_recall(gen_cands, golds),
        "Union": candidate_recall(union_cands, golds),
    }
    save_json(cr, OUT / "metrics" / f"candidate_recall{SUFFIX}.json")
    for name, vals in cr.items():
        log(f"  cand recall [{name}]: " + " ".join(f"{k.split('@')[1]}={v:.3f}" for k, v in vals.items()))

    # P1: merge + score fusion
    log("P1: score fusion")
    scorer = SemanticScorer(device="cuda")
    fused_docs = []
    with Timer("fusion"):
        for i in range(len(rows)):
            merged = merge_candidates(ext_scored[i], gen_pool[i])
            ranked = fuse_scores(merged, texts[i], titles[i], scorer)
            fused_docs.append(ranked)
            if (i + 1) % 500 == 0:
                log(f"  fused {i + 1}/{len(rows)}")
    fused_preds = [[c["phrase"] for c in doc] for doc in fused_docs]
    macro_p1 = evaluate_corpus(fused_preds, golds, all_prmu=prmus, all_doc_texts=texts)
    log_run(logger, f"P1_hybrid_fusion{SUFFIX}", "Hybrid", macro_p1,
            decoder="Beam10", reranker="ScoreFusion", MMR="No", seed=SEED)
    save_preds(rows, golds, prmus, fused_preds, f"P1_hybrid_fusion{SUFFIX}.jsonl")
    save_jsonl(
        [{"id": r["id"], "title": t, "gold": g, "prmu": p,
          "candidates": [{"phrase": c["phrase"], "final_score": c["final_score"], "sources": c["sources"]}
                          for c in doc]}
         for r, t, g, p, doc in zip(rows, titles, golds, prmus, fused_docs)],
        OUT / "candidates" / f"fused_candidates{SUFFIX}.jsonl",
    )

    # P2: MMR λ 스윕 (임베딩 1회 캐시)
    log("P2: MMR λ 스윕")
    with Timer("candidate embeddings"):
        all_embs = [scorer.encode([c["phrase"] for c in doc]) if doc else None for doc in fused_docs]

    def select(lam, top_k=10):
        out = []
        for doc, emb in zip(fused_docs, all_embs):
            if not doc:
                out.append([])
                continue
            scores = [c["final_score"] for c in doc]
            lo, hi = min(scores), max(scores)
            rel = [(s - lo) / (hi - lo) if hi > lo else 0.5 for s in scores]
            if lam is None:
                order = list(np.argsort(-np.asarray(rel))[:top_k])
            else:
                order = mmr_select(emb, rel, top_k=top_k, lambda_=lam)
            out.append([doc[j]["phrase"] for j in order])
        return out

    sweep = {}
    for lam in [None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = select(lam)
        macro = evaluate_corpus(preds, golds, all_prmu=prmus, all_doc_texts=texts)
        reds = []
        for doc, emb, p in list(zip(fused_docs, all_embs, preds))[:400]:
            if len(p) >= 2 and emb is not None:
                lut = {c["phrase"]: emb[j] for j, c in enumerate(doc)}
                sel = np.stack([lut[x] for x in p if x in lut])
                reds.append(semantic_redundancy(sel))
        key = "no_mmr" if lam is None else f"lambda_{lam}"
        sweep[key] = {
            "F1@5": round(macro["F1@5"], 4), "F1@10": round(macro["F1@10"], 4),
            "semantic_redundancy": round(float(np.mean(reds)), 4),
            "dup_ratio": round(macro["dup_ratio"], 4),
        }
        log(f"  {key}: F1@5={sweep[key]['F1@5']} redundancy={sweep[key]['semantic_redundancy']}")
    save_json(sweep, OUT / "metrics" / f"mmr_sweep{SUFFIX}.json")
    # P2의 λ 선택·기록은 tune_mmr stage가 validation 기준으로 수행한다 (test 선택 편향 방지)


# ---------------------------------------------------------------- stage: tune_mmr
def stage_tune_mmr() -> None:
    """MMR λ를 validation에서 선택한 뒤 test에 적용해 P2를 다시 기록한다 (플랜 7.7).

    test에서 λ를 고르면 test 과적합이므로, validation 500건으로
    확장 그리드 {0.3..0.9}를 탐색하고 best λ로 test를 재평가한다.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from src.data import build_source, load_kp20k, plain_doc_text
    from src.diversity import mmr_select
    from src.extraction import KeyBertExtractor
    from src.generation import generate_keyphrases
    from src.metrics import evaluate_corpus, semantic_redundancy
    from src.reranking import SemanticScorer, fuse_scores, merge_candidates
    from src.utils import load_json, load_jsonl

    logger = ExperimentLogger()
    scorer = SemanticScorer(device="cuda")

    def select(fused_docs, embs, lam, top_k=10):
        out = []
        for doc, emb in zip(fused_docs, embs):
            if not doc:
                out.append([])
                continue
            scores = [c["final_score"] for c in doc]
            lo, hi = min(scores), max(scores)
            rel = [(s - lo) / (hi - lo) if hi > lo else 0.5 for s in scores]
            if lam is None:
                order = list(np.argsort(-np.asarray(rel))[:top_k])
            else:
                order = mmr_select(emb, rel, top_k=top_k, lambda_=lam)
            out.append([doc[j]["phrase"] for j in order])
        return out

    # --- validation에서 λ 선택 (프로파일별 파일; 이미 탐색한 결과가 있으면 재사용) ---
    sel_file = OUT / "metrics" / f"mmr_val_selection{SUFFIX}.json"
    if sel_file.exists():
        best_lam = float(load_json(sel_file)["best_lambda"])
        log(f"tune_mmr: 저장된 validation 선택 λ = {best_lam} 재사용 ({sel_file.name})")
    else:
        ckpt = OUT / "checkpoints" / f"keybart{SUFFIX}"
        if not (ckpt / "config.json").exists():
            ckpt = OUT / "checkpoints" / f"bart_base{SUFFIX}"
        assert (ckpt / "config.json").exists(), "keybart/bart 체크포인트 필요"
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to("cuda").eval()
        extractor = KeyBertExtractor(device="cuda")

        VAL_N = 500
        val = load_kp20k(["validation"], subset_sizes={"validation": VAL_N}, seed=SEED)["validation"]
        v_rows = list(val)
        v_texts = [plain_doc_text(r["title"], r["abstract"]) for r in v_rows]
        v_golds = [r["keyphrases"] for r in v_rows]
        v_sources = [build_source(r["title"], r["abstract"]) for r in v_rows]

        log(f"tune_mmr: validation {VAL_N}건 beam10 pool 생성")
        v_pool = []
        with Timer("val beam10"):
            for start in range(0, len(v_sources), 250):
                v_pool.extend(generate_keyphrases(
                    model, tokenizer, v_sources[start : start + 250],
                    strategy="beam10", max_source_length=384, batch_size=6))
                log(f"  val pool {min(start + 250, len(v_sources))}/{len(v_sources)}")
        with Timer("val keybert"):
            v_ext = extractor.extract_batch(v_texts, top_n=30)

        log("tune_mmr: validation fusion")
        v_fused = []
        with Timer("val fusion"):
            for i in range(len(v_rows)):
                merged = merge_candidates(v_ext[i], v_pool[i])
                v_fused.append(fuse_scores(merged, v_texts[i], v_rows[i]["title"], scorer))
        v_embs = [scorer.encode([c["phrase"] for c in doc]) if doc else None for doc in v_fused]

        grid = [None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        val_sweep = {}
        for lam in grid:
            preds = select(v_fused, v_embs, lam)
            macro = evaluate_corpus(preds, v_golds, ks=(5, 10))
            key = "no_mmr" if lam is None else f"lambda_{lam}"
            val_sweep[key] = {"F1@5": round(macro["F1@5"], 4), "F1@10": round(macro["F1@10"], 4)}
            log(f"  [val] {key}: F1@5={val_sweep[key]['F1@5']}")
        best_key = max((k for k in val_sweep if k != "no_mmr"), key=lambda k: val_sweep[k]["F1@5"])
        best_lam = float(best_key.split("_")[1])
        save_json({"grid": val_sweep, "best_lambda": best_lam}, sel_file)
        log(f"tune_mmr: validation 선택 λ = {best_lam}")

    # --- 저장된 test fusion 후보에 best λ 적용 ---
    fused_file = OUT / "candidates" / f"fused_candidates{SUFFIX}.jsonl"
    assert fused_file.exists(), "hybrid stage 산출물이 필요합니다"
    t_saved = load_jsonl(fused_file)
    rows, texts, golds, prmus = load_eval_set()
    assert [r["id"] for r in rows] == [r["id"] for r in t_saved], "test 후보 정렬 불일치"
    # float() 방어: 과거 버그로 문자열로 저장된 점수도 안전하게 복원
    t_fused = [
        [{"phrase": c["phrase"], "final_score": float(c["final_score"])} for c in r["candidates"]]
        for r in t_saved
    ]
    with Timer("test candidate embeddings"):
        t_embs = [scorer.encode([c["phrase"] for c in doc]) if doc else None for doc in t_fused]

    preds = select(t_fused, t_embs, best_lam)
    macro = evaluate_corpus(preds, golds, all_prmu=prmus, all_doc_texts=texts)
    log_run(logger, f"P2_hybrid_fusion_mmr{SUFFIX}", "Hybrid+MMR", macro,
            decoder="Beam10", reranker="ScoreFusion",
            MMR=f"lambda={best_lam} (val-selected)", seed=SEED)
    save_preds(rows, golds, prmus, preds, f"P2_hybrid_mmr{SUFFIX}.jsonl")


# ---------------------------------------------------------------- stage: table
def stage_table() -> None:
    import pandas as pd

    logger = ExperimentLogger()
    df = logger.to_dataframe()
    if df.empty:
        log("기록된 실험이 없습니다")
        return
    order = ["B0_tfidf", "B1_keybert", "B2_keybert_mmr", "B3_bart_beam5",
             "B4_keybart_beam5", "P1_hybrid_fusion", "P2_hybrid_fusion_mmr"]

    def sort_key(run_id: str) -> tuple:
        base = run_id.removesuffix("_full")
        idx = order.index(base) if base in order else 99
        return (0 if run_id.endswith("_full") else 1, idx)  # full 결과를 위쪽 그룹으로

    df["__o"] = df.run_id.map(lambda r: sort_key(str(r)))
    df = df.sort_values("__o").drop(columns="__o")
    cols = ["run_id", "model", "train_subset", "F1@5", "F1@10", "F1@M", "present_F1@5",
            "absent_R@10", "MAP@10", "nDCG@10", "dup_ratio", "recall_P", "recall_U"]
    view = df[[c for c in cols if c in df.columns]]
    pd.set_option("display.width", 220)
    print()
    print(view.to_string(index=False))
    view.to_csv(OUT / "metrics" / "final_experiment_table.csv", index=False)
    log("saved: outputs/metrics/final_experiment_table.csv")


STAGES = {
    "extract": stage_extract,
    "bart": stage_bart,
    "keybart": stage_keybart,
    "hybrid": stage_hybrid,
    "tune_mmr": stage_tune_mmr,
    "table": stage_table,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGES, "all"], required=True)
    parser.add_argument("--profile", choices=["dev", "full"], default="dev",
                        help="dev: train 10k/eval 2k (yaml 그대로) | full: train 530k/eval 20k 전체")
    args = parser.parse_args()

    if args.profile == "full":
        PROFILE, EVAL_N, SUFFIX = "full", 20_000, "_full"
    log(f"profile={PROFILE} eval_n={EVAL_N} suffix='{SUFFIX}'")

    set_seed(SEED)
    todo = list(STAGES) if args.stage == "all" else [args.stage]
    failed: list[str] = []
    for name in todo:
        log(f"########## STAGE: {name} ##########")
        try:
            STAGES[name]()
        except Exception:
            import traceback

            traceback.print_exc()
            failed.append(name)
            log(f"!!! stage '{name}' 실패 — 다음 stage 계속")
    if failed:
        log(f"실패한 stage: {failed}")
        sys.exit(1)
    log("모든 stage 완료")
