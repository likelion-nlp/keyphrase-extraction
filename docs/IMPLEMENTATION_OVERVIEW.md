# KP20k 키프레이즈 추출·생성 — GERD 파이프라인

> 학술 문서(제목+초록)에서 **저자 키프레이즈를 예측**하는 연구 코드베이스.
> 추출(KeyBERT)·생성(KeyBART)·랭킹(cross-encoder)·앙상블을 단일 평가 프로토콜로 비교한다.
> 데이터: [KP20k](https://huggingface.co/datasets/taln-ls2n/kp20k) (train 530,809 / test 20,000)

---

## 핵심 결과 (test 20,000편, F1@5)

> **주 비교는 베이스라인 vs P7이다.** 앙상블(4개 LLM 리랭커 P8/P10/P11/P12의 RRF)은 채점표(정답
> 기준)이고, 그 4개 리랭커·토너먼트 단일 모델은 **채점표 재료라 아래 모델 비교에서 제외**한다
> (기록: [docs/ENCODER_TOURNAMENT_STATUS.md](docs/ENCODER_TOURNAMENT_STATUS.md)).

| 모델 | 기법 | F1@5 | absent R@10 | nDCG@10 |
|---|---|---:|---:|---:|
| **SciBERT 하이브리드 리랭커 (P7)** | 인코더 + aux (PRMU pairwise) | **0.2850** | 0.0504 | 0.3975 |
| Hybrid 융합 + MMR (P2) | 비학습 점수 융합 | 0.2560 | 0.0219 | 0.3429 |
| BART 생성 (B3) | seq2seq (beam5) | 0.2445 | 0.0132 | 0.3088 |
| KeyBART 생성 (B4) | seq2seq (beam5) | 0.2377 | 0.0097 | 0.3111 |
| TF-IDF (B0) | 통계 추출 (무학습) | 0.1261 | 0.0113 | 0.1770 |

**채점표(앙상블)를 구성한 4개 모델** — 아래 리랭커들의 RRF 합의:

| 멤버 | 모델 (HuggingFace) | 혈통 |
|---|---|---|
| P8 | `allenai/specter2_base` | 과학 인용 (110M) |
| P10 | `Alibaba-NLP/gte-reranker-modernbert-base` | 리랭킹 전용 (149M) |
| P11 | `Qwen/Qwen3-Reranker-0.6B` | LLM 디코더 (596M) |
| P12 | `google/embeddinggemma-300m` | Gemma 임베딩 (303M) |

**채점 결과** — 이 채점표 랭킹을 정답으로 두면 **P7이 문서 85%에서 KeyBART 베이스라인을 압도**
(nDCG@10 0.898 vs 0.784). → P7이 채점표를 가장 잘 재현하는 배포용 단일 모델.
[docs/ENSEMBLE_REFERENCE_COMPARISON.md](docs/ENSEMBLE_REFERENCE_COMPARISON.md)

전체 지표(채점표·재료 모델 데이터 포함): [`results/metrics/grand_comparison_all_models.csv`](results/metrics/grand_comparison_all_models.csv)
모델별 코드·산출물 매핑: **[MODEL_CATALOG.md](MODEL_CATALOG.md)**

**주요 발견**
- **P7(SciBERT 하이브리드 리랭커)이 최고 단일 모델** — 성능의 원천은 인코더가 아니라 방법론(aux 3피처 + PRMU-가중 pairwise). (여러 인코더 교체 실험은 채점표 재료 기록 참조.)
- **앙상블 = 채점표**: 4개 LLM 리랭커의 RRF 합의를 정답 기준으로 삼아 P7·베이스라인을 채점. **P7이 문서 85%에서 우세** → P7 = 채점표를 재현하는 단일 모델.
- exact-F1 순위와 실제 검색 유용성 순위가 **역전** — "무엇에 쓸 시스템인가"가 최적 모델을 결정한다.

---

## 무엇을 하나 — GERD 파이프라인

```
문서(제목+초록)
   ├─ Generate : KeyBART seq2seq 로 키프레이즈 생성 (absent 포함)
   ├─ Extract  : KeyBERT 로 본문 구절 추출 (present)
   ├─ Rank     : 두 후보 풀 병합 → cross-encoder(+aux) 로 관련도 채점
   └─ Diversify: MMR 로 중복 제거 → top-K
```

핵심 설계 원칙은 [`README` 하단](#핵심-규칙)과 [docs/METRICS.md](docs/METRICS.md) 참조.

---

## 저장소 구조

```text
├─ src/                  # 핵심 모듈 11개 (단일 evaluator 원칙)
│  ├─ preprocessing.py       # 정규화·PRMU 분류·Porter stemming
│  ├─ metrics.py             # F1@K, PRMU recall, MAP/nDCG (전 실험 공통)
│  ├─ data.py / generation.py / extraction.py / reranking.py / diversity.py
│  └─ pipeline.py / viz.py / utils.py
├─ scripts/              # 실행 오케스트레이터 (재현 진입점)
├─ notebooks/            # 파이프라인 정합 워크스루 (src/results 로드, 재구현 아님)
├─ configs/              # baseline/bart/keybart/hybrid 하이퍼파라미터
├─ docs/                 # 지표 정의·모델 설명·실험 계획·보고서·논문
├─ data/                 # 외부 입력 (cs_papers_20232024.csv)
├─ results/              # 예측 결과 + 지표 (아래)
│  ├─ metrics/                        # 전 지표 CSV·JSON (원장 포함)
│  ├─ predictions_per_document/       # 문서별 결과 17개 (F1·정오 대조)
│  └─ predictions_per_keyphrase/      # 구절별 결과 17개 (rank/score/source) [Git LFS]
├─ tests/                # 평가기 단위 테스트
├─ MODEL_CATALOG.md      # ★ 모델 ↔ 코드 ↔ CSV ↔ 기법 완전 매핑
├─ requirements.txt
└─ outputs/              # 재현 시 스크립트가 산출물을 쓰는 위치 (비어 있음)
```

---

## 빠른 시작

```bash
pip install -r requirements.txt
# GPU 권장. PyTorch/CUDA는 환경에 맞춰 별도 설치: https://pytorch.org/get-started/locally/

pytest tests/          # 평가기 검증
```

### 재현 (전체 파이프라인)

```bash
# 베이스라인 + Hybrid 융합 (B0~B4, P1, P2)
python scripts/run_experiments.py --profile full

# Pairwise 리랭커 (P5, P6)
python scripts/pairwise_reranker.py --stage all

# SciBERT Hybrid 리랭커 (P7)
python scripts/scibert_hybrid_ranker.py

# 인코더 토너먼트 (P8~P15) + 앙상블(P13) + 통합표
python scripts/encoder_tournament.py --stage run
python scripts/encoder_tournament.py --stage ensemble
python scripts/encoder_tournament.py --stage grand

# 팀 사양 앙상블 (P16) — 저장된 점수 재사용, GPU 불필요
python scripts/team_spec_ensemble.py
```

산출물은 `outputs/`에 쌓이고, 사람이 읽기 좋은 정리본은 `python scripts/organize_results.py`로 `results/`에 생성된다.

---

## 문서 색인 (docs/)

| 문서 | 내용 |
|---|---|
| [MODEL_CATALOG.md](MODEL_CATALOG.md) | 모델 17개 전부: 기법·코드·CSV·성능 매핑 (여기부터) |
| [docs/METRICS.md](docs/METRICS.md) | 전 지표 수식 정의 |
| [docs/CSV_COLUMN_DICTIONARY.md](docs/CSV_COLUMN_DICTIONARY.md) | 산출 CSV 열 의미 |
| [docs/MODEL_DESCRIPTIONS_P8_P15.md](docs/MODEL_DESCRIPTIONS_P8_P15.md) | 토너먼트 인코더별 설명 |
| [docs/ENCODER_TOURNAMENT_STATUS.md](docs/ENCODER_TOURNAMENT_STATUS.md) | 토너먼트·앙상블 최종 결과·해석 |
| [docs/ENSEMBLE_REFERENCE_COMPARISON.md](docs/ENSEMBLE_REFERENCE_COMPARISON.md) | 앙상블 기준 P7 vs 베이스라인 랭킹 재현 비교 |
| [docs/PRMU_PAIRWISE_RERANKER_PLAN.md](docs/PRMU_PAIRWISE_RERANKER_PLAN.md) | PRMU 가중 pairwise 설계 |
| [docs/REPORT_AUTHOR_KEYPHRASE_PATTERNS.md](docs/REPORT_AUTHOR_KEYPHRASE_PATTERNS.md) | 저자 키프레이즈 선정 패턴 분석 |
| [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) | 프로젝트 마스터 플랜 |

---

## 핵심 규칙

1. **모든 모델은 `src.metrics.evaluate_corpus` 하나로 평가** — 정규화(lowercase→토큰화(하이픈 유지)→Porter stem)가 전 실험 동일해야 비교 성립.
2. **입력은 title+abstract** — present 판정 기준이 제목+초록.
3. **gold 순서 ≠ 중요도** — 랭킹은 관련도 점수로 별도 산출.
4. **하이퍼파라미터는 validation에서 선택** — test에서 고르면 선택 편향.
5. 처음부터 full train을 돌리지 않는다: 1k(overfit 확인) → 10k → full.

---

## 데이터·모델 가중치

- 학습 체크포인트(총 21GB)와 대용량 중간 산출물은 저장소에 포함하지 않는다 — 위 재현 명령으로 생성.
- `results/predictions_per_keyphrase/`는 파일당 ~45MB(총 ~708MB)로 일반 git에 포함된다(파일당 100MB 한도 이내). 저장소가 다소 커서 클론이 느릴 수 있다.

## 라이선스

TODO — 배포 전 라이선스 지정 필요 (코드/데이터 각각). KP20k 원본 데이터는 원 저작권을 따른다.
