"""생성형 모델 모듈: One2Seq 데이터 변환, Seq2Seq fine-tuning, 후보 생성(디코딩).

- BART-base(일반 생성 베이스라인)와 KeyBART(강한 생성 베이스라인)를 동일 조건으로 다룬다.
- 학습·생성 모두 GPU를 사용한다 (bf16 지원 시 bf16, 아니면 fp16).
- 디코딩: greedy / beam / diverse beam / top-p sampling (플랜 Notebook 08).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from .data import build_example
from .preprocessing import SPECIAL_TOKENS, clean_candidates, parse_generated_sequence


def load_seq2seq(model_name: str, add_special_tokens: bool = True, device=None):
    """모델·토크나이저 로드 후 스페셜 토큰(<title>/<abstract>/<present>/<absent>/<kp_sep>) 등록."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if add_special_tokens:
        added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        if added:
            model.resize_token_embeddings(len(tokenizer))
    if device is not None:
        model = model.to(device)
    return model, tokenizer


def make_seq2seq_dataset(
    hf_dataset,
    tokenizer,
    max_source_length: int = 384,
    max_target_length: int = 96,
    input_mode: str = "title_abstract",
    target_format: str = "present_absent",
    shuffle_targets: bool = False,
    seed: int = 42,
    num_proc: int | None = None,
):
    """HF dataset → 토크나이즈된 Seq2Seq 학습 dataset."""

    def _convert(row, idx):
        shuffle_seed = (seed + idx) if shuffle_targets else None
        ex = build_example(row, input_mode, target_format, shuffle_seed)
        model_inputs = tokenizer(
            ex["source"], max_length=max_source_length, truncation=True
        )
        labels = tokenizer(text_target=ex["target"], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    cols = hf_dataset.column_names
    return hf_dataset.map(_convert, with_indices=True, remove_columns=cols, num_proc=num_proc)


def precision_dtype_kwargs() -> dict[str, bool]:
    """GPU 정밀도 자동 선택: bf16 지원 GPU(RTX 5080, A100 등)는 bf16, 아니면 fp16(T4 등)."""
    if not torch.cuda.is_available():
        return {}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True}
    return {"fp16": True}


def train_seq2seq(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    output_dir: str = "outputs/checkpoints/run",
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: float = 2.0,
    max_steps: int = -1,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    logging_steps: int = 25,
    eval_steps: int | None = None,
    seed: int = 42,
    gradient_checkpointing: bool = False,
    generation_max_length: int = 96,
    predict_with_generate: bool = False,
    disable_tqdm: bool = False,
):
    """Seq2SeqTrainer 래퍼 (플랜 13절 하이퍼파라미터 기준, GPU bf16/fp16 자동).

    predict_with_generate는 기본 False: 학습 중 eval은 loss만 계산하고,
    생성 품질 평가는 학습 후 generate_keyphrases + evaluate_corpus로 별도 수행한다
    (compute_metrics 없이 eval마다 생성하는 것은 시간 낭비).
    """
    from transformers import (
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    do_eval = eval_dataset is not None
    # eval loss 기준 best 체크포인트 자동 선택 — 마지막 에폭이 과적합이어도
    # 가장 좋았던 시점의 가중치로 복원된다 (KeyBART full 실험에서 epoch2 과적합 실측).
    use_best = do_eval and max_steps < 0
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=max(per_device_train_batch_size, 8),
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        eval_strategy=("steps" if do_eval and eval_steps else ("epoch" if do_eval else "no")),
        eval_steps=eval_steps,
        save_strategy="epoch" if max_steps < 0 else "no",
        predict_with_generate=predict_with_generate and do_eval,
        generation_max_length=generation_max_length,
        seed=seed,
        report_to=[],
        gradient_checkpointing=gradient_checkpointing,
        load_best_model_at_end=use_best,
        metric_for_best_model="eval_loss" if use_best else None,
        greater_is_better=False if use_best else None,
        save_total_limit=2 if use_best else 1,
        disable_tqdm=disable_tqdm,
        **precision_dtype_kwargs(),
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    return trainer


# ---------------------------------------------------------------------------
# 후보 생성 (디코딩)
# ---------------------------------------------------------------------------

DECODING_PRESETS: dict[str, dict[str, Any]] = {
    "greedy": {"num_beams": 1, "do_sample": False, "num_return_sequences": 1},
    "beam5": {"num_beams": 5, "do_sample": False, "num_return_sequences": 1},
    "beam10": {"num_beams": 10, "do_sample": False, "num_return_sequences": 10},
    "diverse_beam": {
        "num_beams": 10,
        "num_beam_groups": 5,
        "diversity_penalty": 1.0,
        "do_sample": False,
        "num_return_sequences": 10,
    },
    # num_beams=1 명시: BART류는 기본 generation config에 num_beams>1이 있어
    # 샘플링 시 num_return_sequences > num_beams 검증에 걸린다.
    "top_p": {
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.9,
        "num_beams": 1,
        "num_return_sequences": 8,
    },
}


@torch.no_grad()
def generate_keyphrases(
    model,
    tokenizer,
    sources: Sequence[str],
    strategy: str = "beam5",
    max_source_length: int = 384,
    max_new_tokens: int = 96,
    batch_size: int = 16,
    length_alpha: float = 0.7,
    clean: bool = True,
    generation_kwargs: dict | None = None,
) -> list[list[dict[str, Any]]]:
    """문서별 키프레이즈 후보를 생성한다 (GPU 배치 추론).

    Returns:
        문서별 리스트. 각 원소는 {"text", "gen_score", "seq_id"} dict.
        gen_score는 길이 정규화된 시퀀스 log-prob (플랜 3.1절 수식)을
        해당 시퀀스에서 나온 모든 phrase에 부여한 값이다. 같은 phrase가
        여러 시퀀스에서 나오면 최고 점수를 사용한다.

    주의: One2Seq에서 beam 1개 = '키프레이즈 목록' 1개다 (플랜 4.4 문제5).
    num_return_sequences>1이면 모든 시퀀스를 분해해 후보 pool을 만든다.
    """
    device = next(model.parameters()).device
    gen_kwargs = dict(DECODING_PRESETS.get(strategy, DECODING_PRESETS["beam5"]))
    if generation_kwargs:
        gen_kwargs.update(generation_kwargs)
    num_return = gen_kwargs.get("num_return_sequences", 1)

    model.eval()
    results: list[list[dict[str, Any]]] = []
    for start in range(0, len(sources), batch_size):
        batch = list(sources[start : start + batch_size])
        enc = tokenizer(
            batch,
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        )
        # 시퀀스 점수: beam search는 sequences_scores 제공(길이 페널티 반영된 평균 log-prob).
        # sampling 등 미제공 시 transition score로 직접 계산한다.
        if getattr(out, "sequences_scores", None) is not None:
            seq_scores = out.sequences_scores.float().cpu().numpy()
        else:
            try:
                trans = model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True
                )
                mask = (trans > -1e9) & ~torch.isinf(trans)
                lengths = mask.sum(dim=1).clamp(min=1).float()
                seq_scores = (
                    ((trans * mask).sum(dim=1) / lengths.pow(length_alpha)).float().cpu().numpy()
                )
            except Exception:
                seq_scores = np.zeros(out.sequences.shape[0], dtype=np.float32)

        texts = tokenizer.batch_decode(out.sequences, skip_special_tokens=False)
        for doc_i in range(len(batch)):
            phrase_scores: dict[str, dict[str, Any]] = {}
            for r in range(num_return):
                seq_idx = doc_i * num_return + r
                raw = texts[seq_idx]
                # pad/eos류 토큰 제거하되 커스텀 스페셜 토큰은 parse에서 사용
                for tok in [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token, "</s>", "<s>", "<pad>"]:
                    if tok:
                        raw = raw.replace(tok, " ")
                phrases = parse_generated_sequence(raw)
                if clean:
                    phrases = clean_candidates(phrases, source_text=batch[doc_i])
                score = float(seq_scores[seq_idx]) if seq_idx < len(seq_scores) else 0.0
                for ph in phrases:
                    key = ph.lower()
                    if key not in phrase_scores or score > phrase_scores[key]["gen_score"]:
                        phrase_scores[key] = {"text": ph, "gen_score": score, "seq_id": r}
            ranked = sorted(phrase_scores.values(), key=lambda d: -d["gen_score"])
            results.append(ranked)
    return results


def perplexity_proxy(gen_score: float) -> float:
    """길이 정규화 log-prob → 이해하기 쉬운 exp 스케일 값."""
    return math.exp(gen_score)
