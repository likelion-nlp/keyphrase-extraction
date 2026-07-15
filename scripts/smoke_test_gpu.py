"""GPU 엔드투엔드 스모크 테스트.

전체 파이프라인(데이터 → 추출 → 학습 → 생성 → 융합 → MMR → 평가)을
초소형 규모로 GPU에서 1회 관통한다. 환경이 올바르면 수 분 내 종료된다.

실행: python scripts/smoke_test_gpu.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data import build_source, load_kp20k, plain_doc_text
from src.diversity import mmr_select_candidates
from src.extraction import KeyBertExtractor, TfidfExtractor
from src.generation import generate_keyphrases, load_seq2seq, make_seq2seq_dataset, train_seq2seq
from src.metrics import candidate_recall, evaluate_corpus
from src.pipeline import KeyphrasePipeline
from src.reranking import SemanticScorer, fuse_scores, merge_candidates
from src.utils import ExperimentLogger, set_seed

STEP_T0 = time.time()


def step(msg: str) -> None:
    print(f"\n[{time.time() - STEP_T0:6.1f}s] === {msg} ===", flush=True)


def main() -> None:
    set_seed(42)
    assert torch.cuda.is_available(), "GPU 필수: CUDA가 감지되지 않았습니다"
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    step("1/8 데이터 로드 (train 120 / test 24)")
    ds = load_kp20k(subset_sizes={"train": 120, "validation": 24, "test": 24})
    test_rows = list(ds["test"])
    texts = [plain_doc_text(r["title"], r["abstract"]) for r in test_rows]
    golds = [r["keyphrases"] for r in test_rows]
    prmus = [r["prmu"] for r in test_rows]
    assert all(len(g) == len(p) for g, p in zip(golds, prmus))

    step("2/8 TF-IDF 추출")
    tfidf = TfidfExtractor(max_features=50_000, min_df=1)
    tfidf.fit([plain_doc_text(r["title"], r["abstract"]) for r in ds["train"]])
    tfidf_preds = [[p for p, _ in doc] for doc in tfidf.extract_batch(texts, top_n=10)]
    assert all(tfidf_preds), "TF-IDF 예측이 비었습니다"

    step("3/8 KeyBERT 추출 (GPU 임베딩)")
    extractor = KeyBertExtractor(device="cuda")
    ext_scored = extractor.extract_batch(texts, top_n=15)
    keybert_preds = [[p for p, _ in doc] for doc in ext_scored]
    macro_kb = evaluate_corpus(keybert_preds, golds, all_prmu=prmus, all_doc_texts=texts)
    print(f"KeyBERT F1@5={macro_kb['F1@5']:.4f}, recall_U={macro_kb.get('recall_U', 0):.4f} (0이어야 정상)")

    step("4/8 BART-base 초미니 학습 (GPU, 30 step)")
    model, tokenizer = load_seq2seq("facebook/bart-base", device=device)
    train_tok = make_seq2seq_dataset(ds["train"], tokenizer, 256, 64)
    trainer = train_seq2seq(
        model, tokenizer, train_tok,
        output_dir=str(PROJECT_ROOT / "outputs" / "checkpoints" / "_smoke"),
        max_steps=30, logging_steps=10, per_device_train_batch_size=4,
        gradient_accumulation_steps=1, learning_rate=3e-4,
    )
    result = trainer.train()
    print(f"train loss: {result.training_loss:.3f}")
    assert result.training_loss < 12, "loss가 비정상적으로 큼"

    step("5/8 생성: beam5 / beam10 / top_p (GPU)")
    sources = [build_source(r["title"], r["abstract"]) for r in test_rows]
    gen5 = generate_keyphrases(model, tokenizer, sources, strategy="beam5", max_source_length=256, batch_size=8)
    gen10 = generate_keyphrases(model, tokenizer, sources, strategy="beam10", max_source_length=256, batch_size=8)
    gen_tp = generate_keyphrases(model, tokenizer, sources[:4], strategy="top_p", max_source_length=256, batch_size=4)
    print(f"beam5 phrases/doc={sum(len(d) for d in gen5) / len(gen5):.1f}, "
          f"beam10={sum(len(d) for d in gen10) / len(gen10):.1f}, "
          f"top_p(4 docs)={sum(len(d) for d in gen_tp) / max(1, len(gen_tp)):.1f}")
    gen_preds = [[g["text"] for g in doc] for doc in gen5]
    macro_gen = evaluate_corpus(gen_preds, golds, all_prmu=prmus, all_doc_texts=texts)
    print(f"BART(30step) F1@5={macro_gen['F1@5']:.4f} — 스모크에서는 0이어도 무방(파이프라인 검증이 목적)")

    step("6/8 후보 통합 + score fusion + candidate recall")
    scorer = SemanticScorer(device="cuda")
    fused_preds = []
    for i in range(len(test_rows)):
        merged = merge_candidates(ext_scored[i], gen10[i])
        ranked = fuse_scores(merged, texts[i], test_rows[i]["title"], scorer)
        fused_preds.append([c["phrase"] for c in ranked])
    cr = candidate_recall([e + [c["text"] for c in g] for e, g in zip(keybert_preds, gen10)], golds)
    print("union candidate recall:", {k: round(v, 3) for k, v in cr.items()})
    macro_hybrid = evaluate_corpus(fused_preds, golds, all_prmu=prmus, all_doc_texts=texts)
    print(f"Hybrid F1@5={macro_hybrid['F1@5']:.4f}")

    step("7/8 MMR 중복 제거")
    merged0 = merge_candidates(ext_scored[0], gen10[0])
    ranked0 = fuse_scores(merged0, texts[0], test_rows[0]["title"], scorer)
    selected = mmr_select_candidates(ranked0, scorer, top_k=5, lambda_=0.7)
    assert 0 < len(selected) <= 5
    print("MMR Top-5:", [c["phrase"] for c in selected])

    step("8/8 KeyphrasePipeline (GERD 전체) + 실험 로그")
    pipe = KeyphrasePipeline(
        generator_model=model, generator_tokenizer=tokenizer,
        extractor=extractor, scorer=scorer, max_source_length=256,
    )
    out = pipe.predict_keyphrases(test_rows[0]["title"], test_rows[0]["abstract"], top_k=5)
    assert out and all({"phrase", "score", "source", "type"} <= set(o) for o in out)
    for o in out:
        print(f"  {o['score']:.3f}  [{o['type']:>7}]  {o['phrase']}  ({'+'.join(o['source'])})")

    logger = ExperimentLogger(PROJECT_ROOT / "outputs" / "metrics" / "smoke_experiments.csv")
    logger.log(run_id="smoke_hybrid", model="smoke", **{"F1@5": round(macro_hybrid["F1@5"], 4)})
    assert logger.to_dataframe().shape[0] >= 1

    print(f"\n[{time.time() - STEP_T0:6.1f}s] ✅ 전체 파이프라인 GPU 스모크 테스트 통과")


if __name__ == "__main__":
    main()
