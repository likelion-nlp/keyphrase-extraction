# 모델 카탈로그 — 코드 · 산출물 · 기법 매핑

> 이 프로젝트의 **모든 모델**을 한눈에: 각 모델이 **어떤 기법**인지, **어떤 코드**로 돌렸는지,
> **어떤 CSV**가 나오는지, **성능**이 얼마인지를 하나로 묶은 색인 문서.
> GitHub 방문자가 이 파일 하나로 프로젝트 구조를 이해할 수 있게 하는 것이 목적.
>
> 평가: KP20k test **20,000편** · 학습: train **530,809편 전체** · seed 42 · 단일 채점기(Porter stemming)

> **이 저장소에서의 파일 위치**: 아래 표의 `kp20k_test_*` 는 재현 시 스크립트가
> `outputs/predictions/`에 쓰는 원본 이름이다. 사람이 읽기 좋은 **정리본**(깨끗한 이름·번호)은
> [`results/predictions_per_document/`](results/predictions_per_document/)(문서별)와
> [`results/predictions_per_keyphrase/`](results/predictions_per_keyphrase/)(구절별)에 있다.
> 매핑은 아래 §4 표 참조.

---

## 1. 한눈에 보기 (성능순, F1@5)

> **주 비교는 베이스라인 vs P7.** 토너먼트 단일 모델(P8~P15)과 앙상블은 **채점표 및 그 재료라
> 이 순위에서 제외**한다 (재료 기록: §3.4·[`docs/ENCODER_TOURNAMENT_STATUS.md`](docs/ENCODER_TOURNAMENT_STATUS.md)).

| 순위 | 모델 (기법) | run_id | F1@5 | absent R@10 | 비고 |
|---:|---|---|---:|---:|---|
| 🥇 | **SciBERT 하이브리드 리랭커 + aux (PRMU pairwise)** | P7 | **0.2850** | 0.0504 | **최고 단일 모델 · 채점표 재현 1위** |
| 2 | Hybrid 융합 + MMR (비학습) | P2 | 0.2560 | 0.0219 | 비학습 최고 |
| 3 | BART 생성 (beam5) | B3 | 0.2445 | 0.0132 | |
| 4 | Pairwise 랭커 (무가중) | P5 | 0.2388 | 0.0460 | |
| 5 | KeyBART 생성 (beam5) | B4 | 0.2377 | 0.0097 | |
| 6 | Pairwise 랭커 (PRMU 가중) | P6 | 0.2267 | 0.0494 | |
| 7 | Hybrid 융합 (MMR 없음) | P1 | 0.2070 | 0.0323 | |
| 8 | TF-IDF | B0 | 0.1261 | 0.0113 | 무학습 베이스라인 |
| 9 | KeyBERT | B1 | 0.0502 | 0.0060 | |
| 10 | KeyBERT + MMR | B2 | 0.0285 | 0.0035 | |

**채점표 (앙상블, 정답 기준)**: 토너먼트 리랭커 4개(P8/P10/P11/P12)의 RRF. 이 4개 및 나머지
토너먼트 단일 모델(P9/P14 실패 포함)은 gold-F1 경쟁 대상이 아니라 **채점표 재료**다. 채점표로
채점하면 **P7이 문서 85%에서 베이스라인 압도** (nDCG@10 0.898 vs 0.784) —
[`docs/ENSEMBLE_REFERENCE_COMPARISON.md`](docs/ENSEMBLE_REFERENCE_COMPARISON.md).

전체 지표(채점표·재료 모델 데이터 포함): [`results/metrics/grand_comparison_all_models.csv`](results/metrics/grand_comparison_all_models.csv)

---

## 2. 파일명 읽는 법

산출물 CSV는 모두 아래 규칙을 따른다:

```
kp20k_test_{run_id}_full.csv           ← 문서별 결과 (문서 1개 = 1행, 20,000행)
kp20k_test_{run_id}_full_keyphrases.csv ← 구절별 결과 (예측 구절 1개 = 1행, 200,000행)
```

- **full** = 문서 단위. 열: `id, title, gold, gold_prmu, pred_top10([O]=정답), tp@10, F1@5, R@10`
- **keyphrases** = 구절 단위. 열: `id, title, rank, keyphrase, type, prmu, score, source, is_correct, gold_all`
- 열 상세 정의: [`docs/CSV_COLUMN_DICTIONARY.md`](docs/CSV_COLUMN_DICTIONARY.md)

**run_id 접두어**: `B*`=베이스라인, `P*`=제안 파이프라인(순서대로 발전). 숫자는 실험 진행 순서.

---

## 3. 모델별 완전 매핑

각 모델의 **기법 → 실행 코드 → 학습 체크포인트 → 산출 CSV**. `—`는 무학습(체크포인트 없음).

### 3.1 베이스라인 (무학습·생성 모델)

| run_id | 모델·기법 | 실행 코드 | 체크포인트 | 산출 CSV (`kp20k_test_*`) |
|---|---|---|---|---|
| B0 | TF-IDF 통계 추출 | `scripts/run_experiments.py` | — | `B0_tfidf_full` |
| B1 | KeyBERT (SBERT 임베딩 유사도) | `scripts/run_experiments.py` | — | `B1_keybert_full` |
| B2 | KeyBERT + MMR 다양화 | `scripts/run_experiments.py` | — | `B2_keybert_mmr_full` |
| B3 | BART seq2seq 생성 (beam5) | `scripts/run_experiments.py` | `checkpoints/bart_base_full` | `B3_bart_full` |
| B4 | KeyBART seq2seq 생성 (beam5) | `scripts/run_experiments.py` | `checkpoints/keybart_full` | `B4_keybart_full` |

### 3.2 Hybrid 융합 파이프라인 (비학습 랭킹)

KeyBERT(추출) + KeyBART(생성) 후보를 **점수 융합**으로 정렬. GERD 파이프라인의 핵심.

| run_id | 모델·기법 | 실행 코드 | 체크포인트 | 산출 CSV |
|---|---|---|---|---|
| P1 | Hybrid 융합 (MMR 없음) | `scripts/run_experiments.py` | — (생성기: `keybart_full`) | `P1_hybrid_fusion_full` |
| P2 | Hybrid 융합 + MMR(λ=0.5) | `scripts/run_experiments.py` | — | `P2_hybrid_fusion_mmr_full` |

### 3.3 Pairwise 랭킹 리랭커 (PRMU 가중 학습)

margin ranking loss로 후보를 재정렬. PRMU 타입별 가중치로 absent 개선 시도.

| run_id | 모델·기법 | 실행 코드 | 체크포인트 | 산출 CSV |
|---|---|---|---|---|
| P5 | Pairwise 랭커 (무가중, margin 1.0) | `scripts/pairwise_reranker.py` | `checkpoints/reranker_unweighted` | `P5_pairwise_full` |
| P6 | Pairwise 랭커 (PRMU 가중) | `scripts/pairwise_reranker.py` | `checkpoints/reranker_prmu_w` | `P6_pairwise_prmu_full` |

### 3.4 Hybrid 리랭커 = 인코더 + aux 3피처 (P7 방법론)

**이 프로젝트의 핵심 방법론.** 인코더(cross-encoder) + 보조피처 `[gen_score, has_gen, is_present]`
→ MLP 헤드. PRMU 가중 pairwise로 학습. 메인 모델은 **P7(SciBERT)** 하나다.

> **P8~P15는 인코더만 바꾼 통제 실험이자 채점표(앙상블) 재료**이며, 메인 모델 비교(§1)에서는
> 제외한다. 아래 표는 채점표가 어떤 리랭커로 구성됐는지의 기록 — 상세·결과는
> [`docs/ENCODER_TOURNAMENT_STATUS.md`](docs/ENCODER_TOURNAMENT_STATUS.md).

| run_id | 인코더 (기법) | 크기·풀링 | 실행 코드 | 체크포인트 | 산출 CSV |
|---|---|---|---|---|---|
| P7 | **SciBERT** (과학 어휘, 기준) | 110M·CLS | `scripts/scibert_hybrid_ranker.py` | `checkpoints/reranker_scibert` | `P7_scibert_full` |
| P8 | SPECTER2 (과학 인용) | 110M·CLS | `scripts/encoder_tournament.py` | `checkpoints/reranker_P8` | `P8_full` |
| P10 | gte-reranker-ModernBERT (리랭킹 특화) | 149M·CLS | `scripts/encoder_tournament.py` | `checkpoints/reranker_P10` | `P10_full` |
| P11 | **Qwen3-Reranker-0.6B** (LLM 디코더) | 596M·last | `scripts/encoder_tournament.py` | `checkpoints/reranker_P11` | `P11_full` |
| P12 | EmbeddingGemma-300m (Gemma 임베딩) | 303M·mean | `scripts/encoder_tournament.py` | `checkpoints/reranker_P12` | `P12_full` |
| P15 | cs_roberta (CS 도메인 RoBERTa) | 125M·CLS | `scripts/encoder_tournament.py` | `checkpoints/reranker_P15` | `P15_full` |
| P9 ✗ | DeBERTa-v3 (bf16 발산 실패) | — | `scripts/encoder_tournament.py` | `checkpoints/reranker_P9`(불완전) | 없음 |
| P14 ✗ | scideberta-cs (bf16 발산 실패) | — | `scripts/encoder_tournament.py` | 없음 | 없음 |

> P7 구조 정의는 `scripts/scibert_hybrid_ranker.py`의 `HybridRanker`, 인코더 일반화 버전은
> `scripts/encoder_tournament.py`의 `HybridRanker(encoder_name, pooling)`. 로스터는 같은 파일 `ROSTER` dict.
> 인코더별 상세 설명: [`docs/MODEL_DESCRIPTIONS_P8_P15.md`](docs/MODEL_DESCRIPTIONS_P8_P15.md)

### 3.5 앙상블 = 채점표 (정답 기준)

**앙상블은 gold-F1을 두고 겨루는 참가 모델이 아니라, 다른 모델의 랭킹을 채점하는 채점표다.**
4개 LLM 모델(P8/P10/P11/P12)을 RRF로 합쳐 합의 랭킹을 만들고, 이를 정답으로 두어 P7·베이스라인이
그 순서를 얼마나 재현하는지 채점한다.

| run_id | 기법 | 멤버 | 실행 코드 | 산출 데이터 |
|---|---|---|---|---|
| P16 | **팀 사양 채점표** (각 모델 top-10 합집합 → RRF) | P8+P10+P11+P12 | `scripts/team_spec_ensemble.py` | `P16_team_ensemble_full` |
| P13 | RRF 조합 (검증 선택, 풀 전체) — P7 포함이라 채점 기준으론 부적합(순환) | P7+P10+P11+P12 | `scripts/encoder_tournament.py` (`--stage ensemble`) | `P13_ensemble_full` |

> **채점표는 P16 구성(P8/P10/P11/P12)** — P7을 포함하지 않아 "P7이 채점표를 재현하나"를 순환 없이 잰다.
> P13은 P7을 포함하므로 채점 기준이 아니라 RRF 조합 실험 기록으로만 남긴다.
>
> **채점 결과**: 앙상블 채점표를 정답으로 두면 **P7이 문서 85%에서 KeyBART 베이스라인 압도**
> (nDCG@10 0.898 vs 0.784, RBO@10 0.452 vs 0.325, Wilcoxon p≈0). 상세:
> [`docs/ENSEMBLE_REFERENCE_COMPARISON.md`](docs/ENSEMBLE_REFERENCE_COMPARISON.md).
> P16 사양 배경: [`docs/ENCODER_TOURNAMENT_STATUS.md`](docs/ENCODER_TOURNAMENT_STATUS.md) §3b.
> → [`docs/ENSEMBLE_REFERENCE_COMPARISON.md`](docs/ENSEMBLE_REFERENCE_COMPARISON.md),
> 노트북 `notebooks/05_ensemble_reference_comparison.ipynb`.

---

## 4. 깨끗한 파일명 매핑 (results/ 에 적용됨)

`P8_full`처럼 번호만 있던 원본 이름을 아래 이름으로 정리해 `results/`에 두었다.
번호 접두어는 파이프라인 발전 순서. (원본 `outputs/predictions/`는 스크립트가 참조하므로 유지)

| 원본 (outputs/predictions/) | → results/ (문서별·구절별 공통) |
|---|---|
| `kp20k_test_B0_tfidf_full` | `00_baseline_tfidf` |
| `kp20k_test_B1_keybert_full` | `01_baseline_keybert` |
| `kp20k_test_B2_keybert_mmr_full` | `02_baseline_keybert_mmr` |
| `kp20k_test_B3_bart_full` | `03_baseline_bart_gen` |
| `kp20k_test_B4_keybart_full` | `04_baseline_keybart_gen` |
| `kp20k_test_P1_hybrid_fusion_full` | `10_hybrid_fusion` |
| `kp20k_test_P2_hybrid_fusion_mmr_full` | `11_hybrid_fusion_mmr` |
| `kp20k_test_P5_pairwise_full` | `20_pairwise_unweighted` |
| `kp20k_test_P6_pairwise_prmu_full` | `21_pairwise_prmu_weighted` |
| `kp20k_test_P7_scibert_full` | `30_reranker_scibert` |
| `kp20k_test_P8_full` | `31_reranker_specter2` |
| `kp20k_test_P10_full` | `32_reranker_gte_modernbert` |
| `kp20k_test_P11_full` | `33_reranker_qwen3_0.6b` |
| `kp20k_test_P12_full` | `34_reranker_embeddinggemma` |
| `kp20k_test_P15_full` | `35_reranker_cs_roberta` |
| `kp20k_test_P13_ensemble_full` | `40_ensemble_rrf_selected` |
| `kp20k_test_P16_team_ensemble_full` | `41_ensemble_team_spec` |

재현 후 이 정리본을 다시 만들려면: `python scripts/organize_results.py`.
(P9 DeBERTa-v3·P14 scideberta-cs는 bf16 발산으로 학습 실패 → 산출물 없음.)

---

## 5. 지표·집계 CSV (모델 아님, 참고용)

| 파일 | 내용 | 생성 코드 |
|---|---|---|
| `metrics/grand_comparison_all_models.csv` | **전 모델 통합 지표표** (B0~P16) | `encoder_tournament.py --stage grand` |
| `metrics/results_full_dataset.csv` | 베이스라인~P2 상세 지표 | `scripts/export_results_csv.py` |
| `metrics/pairwise_vs_baseline_comparison.csv` | P5/P6 vs 기존 비교 | `scripts/pairwise_reranker.py` |
| `metrics/scorecard_all_systems.csv` | reference-free 3축 채점표 | `scripts/scorecard_all_systems.py` |
| `metrics/reference_free_scorecard.csv` | 무정답 지표 (검색·특이도·환각) | `scripts/reference_free_scorecard.py` |
| `metrics/experiments.csv` | **전 실험 원장** (append 누적) | 전 스크립트 공통 (`ExperimentLogger`) |

---

## 6. 외부 데이터 예측 (정답 없는 신규 논문)

`cs_papers_20232024.csv` (2023~24 arXiv 50편, gold 없음)에 대한 예측:

| 파일 | 내용 | 생성 코드 |
|---|---|---|
| `predictions/newpapers_cs_papers_20232024_*` | fusion(P2) 예측 (summary/keyphrases) | `scripts/predict_new_papers.py --ranker fusion` |
| `predictions/newpapers_cs_papers_20232024_pairwise_*` | pairwise(P6) 예측 | `scripts/predict_new_papers.py --ranker pairwise` |
| `predictions/newpapers_allsystems_keyphrases.csv` | 전 시스템 통합 | `scripts/export_all_predictions.py` |

---

## 7. GitHub 업로드 가이드

**✅ 반드시 포함 (코드·문서·핵심 결과):**
- `scripts/` 전체, `src/` 전체, `notebooks/` 전체, `docs/` 전체, `README.md`, 이 파일
- `outputs/metrics/*.csv` (작고 핵심 — 전부 포함)
- `outputs/predictions/*_full.csv` (문서별, 각 ~7.6MB — 읽기 쉬운 최종 결과)

**⚠️ 선별 포함 (용량 큼):**
- `*_keyphrases.csv` 는 각 **45MB** — 전부 올리면 저장소 700MB+. **챔피언(P16/P13/P7)만** 올리거나
  Git LFS 사용, 또는 `.zip` 압축 권장.

**🚫 제외 (`.gitignore` — 이미 설정됨):**
- `outputs/checkpoints/` (모델 가중치, `keybart_full`만 1.5GB) → HuggingFace Hub 별도 업로드 권장
- `.secrets/` (HF 토큰), `outputs/candidates/*.jsonl` (대용량 중간 산출물), `__pycache__/`

---

## 8. 재현 명령 요약

```bash
# 베이스라인 + Hybrid 융합 (B0~B4, P1, P2)
python scripts/run_experiments.py --profile full

# Pairwise 리랭커 (P5, P6)
python scripts/pairwise_reranker.py --stage all

# SciBERT Hybrid 리랭커 (P7)
python scripts/scibert_hybrid_ranker.py

# 인코더 토너먼트 (P8~P15) + 앙상블(P13) + 통합표
python scripts/encoder_tournament.py --stage run     # 각 인코더 학습·평가
python scripts/encoder_tournament.py --stage ensemble # P13
python scripts/encoder_tournament.py --stage grand    # 통합 CSV

# 팀 사양 앙상블 (P16) — 저장된 점수 재사용, GPU 불필요
python scripts/team_spec_ensemble.py
```
