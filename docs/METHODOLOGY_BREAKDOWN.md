# 방법론 분해 — 각 모델이 어떤 블록으로 만들어졌나

> **헷갈릴 때 여기부터.** 모든 모델은 레고 블록 조합이다:
> **`[후보 생성] → [랭킹 방법] → [다양화(선택)]`**.
> 재료(후보)는 대부분 같고, **랭킹 방법과 MMR 유무만** 다르다.

---

## 1. 한눈에 — 모델 = 블록 조합

읽는 법: `A + B + C` = "A로 후보 만들고 → B로 순위 매기고 → C로 다양화". 지표는 F1@10 (팀 리더보드 기준).

| run_id | 사람이 읽는 이름 | 후보 생성 (재료) | + 랭킹 방법 | + 다양화 | 만든 코드 | F1@10 |
|---|---|---|---|---|---|---:|
| **B3** | BART 생성 | `bart-base` seq2seq (beam5) | 생성 순서 그대로 | — | `run_experiments.py` | 0.1711 |
| **B4** | KeyBART 생성 | `KeyBART` seq2seq (beam5) | 생성 순서 그대로 | — | `run_experiments.py` | 0.1672 |
| **P1** | Hybrid 융합 | KeyBERT 추출 + KeyBART 생성 | SBERT 점수 융합 | — | `run_experiments.py` | 0.1842 |
| **P2** | Hybrid 융합 **+ MMR** | KeyBERT 추출 + KeyBART 생성 | SBERT 점수 융합 | **MMR (λ=0.5)** | `run_experiments.py` | 0.2105 |
| **P5** | Pairwise (무가중) | KeyBERT 추출 + KeyBART 생성 | 크로스인코더 pairwise **무가중** | — | `pairwise_reranker.py` | 0.2479 |
| P5+MMR | Pairwise (무가중) + MMR | 〃 | 〃 | **MMR** | `pairwise_reranker.py` | (변형) |
| **P6** | Pairwise (**PRMU 가중**) | KeyBERT 추출 + KeyBART 생성 | 크로스인코더 pairwise **PRMU 가중** | — | `pairwise_reranker.py` | 0.2443 |
| P6+MMR | Pairwise (PRMU 가중) + MMR | 〃 | 〃 | **MMR** | `pairwise_reranker.py` | (변형) |
| **P7** | SciBERT Hybrid 랭커 | KeyBERT 추출 + KeyBART 생성 | **SciBERT + aux** 하이브리드 (PRMU 가중 pairwise) | — | `scibert_hybrid_ranker.py` | **0.2621** |
| P7+MMR | SciBERT Hybrid + MMR | 〃 | 〃 | **MMR** | `scibert_hybrid_ranker.py` | 0.2532 |

> 베이스라인 B0/B1/B2는 단일 블록: **B0** = TF-IDF 통계 추출 · **B1** = KeyBERT 임베딩 추출 · **B2** = KeyBERT + MMR.

**핵심 한 줄**: P1~P7은 전부 **같은 후보(KeyBERT 추출 + KeyBART 생성)**를 쓰고, **랭킹 방법만** 융합 → pairwise → SciBERT hybrid 순으로 정교해진다. MMR은 어디에나 얹을 수 있는 옵션.

---

## 2. 블록 사전 — 각 블록이 실제로 뭘 하나

### 🧩 후보 생성 블록 (재료 만들기)

| 블록 | 쓰는 모델 | 하는 일 |
|---|---|---|
| **KeyBERT 추출** | `sentence-transformers/all-MiniLM-L6-v2` | 본문 구절을 임베딩해 문서와 가까운 구절 top-30 추출 → **present(등장) 후보** |
| **KeyBART 생성** | `bloomberg/KeyBART` (KP20k fine-tune, `keybart_full`) | seq2seq로 키프레이즈를 **생성**(beam10) → 원문에 없는 **absent 후보**까지 만듦 |
| **BART 생성** | `facebook/bart-base` (fine-tune) | KeyBART와 같은 구조지만 키프레이즈 특화 사전학습이 **없는** 순수 생성 (B3 전용) |

### 🧩 랭킹 블록 (후보를 어떤 순서로 세우나)

| 블록 | 쓰는 모델 | 하는 일 | 학습 |
|---|---|---|---|
| **SBERT 점수 융합** (P1/P2) | `all-MiniLM-L6-v2` | 후보마다 [문서 유사도 + 생성 확률 + 제목 겹침]을 가중 합산해 정렬 | ✗ 무학습 |
| **Pairwise 크로스인코더** (P5/P6) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | (문서,후보)를 함께 읽어 점수 1개 출력. "정답이 오답보다 margin↑" pairwise loss로 학습 후 정렬 | ✓ 학습 |
| **SciBERT Hybrid** (P7) | `allenai/scibert` + aux 3피처 → MLP | 인코더를 과학도메인 SciBERT로 교체 + 보조피처(생성확률·생성여부·present여부)를 붙여 MLP로 점수. PRMU 가중 pairwise 학습 | ✓ 학습 |

**P5 vs P6 차이 (딱 하나)**: 같은 크로스인코더인데 학습 가중치가 다름.
- P5 = 전부 동일 가중(무가중)
- P6 = **PRMU 타입별 가중** `{P:1.0, R:1.5, M:2.0, U:2.5}` → absent(M/U) 정답을 더 세게 끌어올림

### 🧩 다양화 블록 (선택)

| 블록 | 하는 일 |
|---|---|
| **MMR (λ=0.5)** | 이미 뽑은 것과 의미가 겹치는 후보에 페널티 → 중복 제거. **융합(P2)엔 도움**, **학습된 랭커(P7)엔 오히려 손해**(이미 다양해서 relevant 후보를 밀어냄) |

---

## 3. 발전 순서로 읽기 (왜 이렇게 쌓였나)

```
B3/B4  생성만            : KeyBART가 키프레이즈를 통째로 생성 (추출 없음, 랭킹 없음)
  │
P1     + 추출 합치기     : KeyBERT 추출을 더해 후보 풀 확장 → SBERT로 융합 정렬
  │
P2     + MMR            : 융합 후보의 중복을 MMR로 제거 (비학습 최고)
  │
P5     랭킹을 학습으로   : 융합 대신 크로스인코더를 pairwise로 학습해 정렬
  │
P6     + absent 강조     : PRMU 가중으로 absent 우선순위 학습 신호 강화
  │
P7     인코더 업그레이드 : 크로스인코더를 SciBERT+aux 하이브리드로 교체 (단일 모델 최고)
```

**요약**: "재료는 그대로 두고 랭킹을 점점 똑똑하게" 만든 과정이다. 각 단계에서 **딱 한 블록씩** 바뀐다 —
그게 이 실험 설계의 통제(controlled) 포인트다.

---

## 4. 파일 이름 ↔ 방법론 (헷갈림 방지)

`results/predictions_per_document/`의 정리된 파일명과 위 방법론 매핑:

| 파일명 | = run_id | = 방법론 |
|---|---|---|
| `03_baseline_bart_gen` | B3 | BART 생성 |
| `04_baseline_keybart_gen` | B4 | KeyBART 생성 |
| `10_hybrid_fusion` | P1 | KeyBERT+KeyBART → SBERT 융합 |
| `11_hybrid_fusion_mmr` | P2 | KeyBERT+KeyBART → SBERT 융합 → MMR |
| `20_pairwise_unweighted` | P5 | KeyBERT+KeyBART → 크로스인코더 pairwise(무가중) |
| `21_pairwise_prmu_weighted` | P6 | KeyBERT+KeyBART → 크로스인코더 pairwise(PRMU 가중) |
| `30_reranker_scibert` | P7 | KeyBERT+KeyBART → SciBERT+aux 하이브리드 |

> 코드↔산출물 전체 매핑은 [`MODEL_CATALOG.md`](../MODEL_CATALOG.md), 모델별 학습 수치·해석은 각 `docs/` 문서 참조.
