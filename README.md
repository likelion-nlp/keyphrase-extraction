# 연구계획서

## 1. Problem Definition (Needs)

### 1-1. "Vocabulary Mismatch Problem" 문제 해결

정보검색(IR) 분야에서 오래된 문제 중 하나가 어휘 불일치 문제다 — 사용자가 찾고 싶은 개념과 문서에 쓰인 실제 단어가 다를 때 검색에 실패하는 현상.

ex) "environment"가 없는데 "space"로 존재, "evolution"이 U로 분류. 이 현상이 Vocabulary Mismatch 문제의 예시이며, present-only 추출 방식은 문제를 근본적으로 해결하지 못하고 absent Keyphrase 생성 능력이 있어야 해결 가능하다.

> **실험적 근거**: B0(TF-IDF)·B1(KeyBERT)·B2(KeyBERT+MMR) 등 순수 추출 기반 baseline의 `recall_U`(unseen/absent 회수율)는 각각 0.0165 / 0.0026 / 0.0033으로 사실상 0에 가깝다. 구조적으로 absent를 생성할 수 없는 방식이라는 1-3의 주장이 정량적으로 확인된다.

### 1-2. 구체적 사용자군(Persona)별 Needs

| 사용자 | 겪는 문제 | 프로젝트가 해결하는 방식 |
| --- | --- | --- |
| 논문 검색/추천 서비스 이용자 | 논문을 다 읽지 않고 핵심 개념만 빠르게 파악 | 랭킹된 top-k 키프레이즈로 "이 논문이 뭘 다루는지"를 몇 단어로 즉시 파악 가능 |
| 일반 사용자 | 긴 요약문 대신 한눈에 들어오는 태그 형태 정보를 원함 | 요약보다 압축률이 높은 "개념 태그" 형태로 정보 제공 |
| **RAG 파이프라인 설계자** | dense 임베딩만으로는 어휘 불일치로 인한 검색 실패(recall 손실) 발생 | present keyphrase → sparse 인덱스(BM25), absent keyphrase → query/document expansion으로 hybrid retrieval 구성 |

### 1-3. 어떤 문제를 해결하려고 하는가?

입력 문서 $x = (title, abstract)$가 주어졌을 때, 다음 세 조건을 동시에 만족하는 순서가 있는 키프레이즈 집합 $Y = (y_1, y_2,...,y_k)$을 생성하는 문제:

$$
f : x \mapsto Y = (y_1, \dots, y_k), \quad y_1 \succ y_2 \succ \dots \succ y_k
$$

본 문제를 만족하려면 아래 세 가지 하위 문제가 동시에 풀려야 한다.

**1. 표현 문제** — $y_i \notin x$ (absent)인 경우도 생성 가능해야 함
→ TextRank, KeyBERT 같은 추출 방식은 $y_i \in x$인 것만 가능 (구조적으로 절반의 정답(U, M의 일부)을 애초에 낼 수 없음)

**2. 우선순위 문제** — $Y$가 정렬되어야 함 (단순 집합이 아니라 순서 있는 리스트)
→ 대부분의 키워드 추출 baseline은 정렬 없이 후보 집합만 반환 → "몇 개만 보여줘야 하는" 실제 사용 시나리오에 대응 못 함

**3. 중복/다양성 문제** — $Y$ 내에서 의미적으로 겹치는 항목 최소화, $sim(y_i,y_j)$가 낮아야 함
→ Seq2Seq beam search는 종종 표현만 다른 유사 phrase를 상위권에 중복 생성함

---

## 2. Background & Baseline

- 기존 연구들은 이 문제, 혹은 유사 문제에 대하여 어떻게 접근하였는가? (논문 리서치, 리뷰는 생략)

### 2-0. Baseline 구성

| ID | 모델 | 학습 여부 | 계열 |
| --- | --- | --- | --- |
| B0 | TF-IDF | 무학습 | Statistical extractive |
| B1 | KeyBERT | 무학습 | Embedding-based extractive |
| B2 | KeyBERT + MMR | 무학습 | Embedding-based extractive + diversity |
| B3 | BART-base (beam5, fine-tuned) | 지도학습 | Seq2Seq generative |
| B4 | **KeyBART (beam5, fine-tuned)** | 지도학습 | Seq2Seq generative (keyphrase 특화 사전학습) |

무학습 계열(B0, B1, B2)과 지도학습 seq2seq 계열(B3, B4)을 모두 baseline에 포함시켜, "학습 없이 어디까지 가능한가"와 "학습 시 얼마나 향상되는가"를 동시에 비교축으로 확보했다.

### 2-1. 정의

Keyphrase 추출 기준, 중요도 랭킹의 의미를 다음 세 관점에서 논의하고, 본 프로젝트는 **③을 채택**한다.

| 관점 | 내용 | 채택 여부 |
| --- | --- | --- |
| ① 문서를 잘 대표하는가 | 저자의 편집 의도(요약 목적)에 가까운 관점 | 참고 |
| ② 검색량을 늘리기 위한 의도가 있는가 | KP20K gold label에 "AI", "NLP" 같은 광범위 상위어가 섞여 있을 가능성 → 저자 편향 가설 (1-1 참고) | 데이터 한계로 인지, 배제 |
| **③ 검색 인덱스로서 기능하는가 (RAG 성능 향상)** | present → sparse 인덱스 재료, absent → query/document expansion 재료로서의 실용적 가치 | **채택** — 본 프로젝트의 최종 목적 |

---

## 3. Proposed Method

### 3-1. EDA
[노션 링크](https://app.notion.com/p/likelion/1-2-39944860a4f48072a9abed47f859bce4?p=39d44860a4f480bfa603fa5a8f97ec1c&pm=s)

| 항목 | 내용 |
| --- | --- |
| 원 논문 / PRMU 출처 | Meng et al. 2017 (ACL) / Boudin & Gallina 2021 (NAACL) |
| 도메인·언어 | 컴퓨터공학(CS) 논문 abstract+title, 영어 |
| 총 문서 수 | 570,809건 (약 721MB) |
| Feature | `id`, `title`, `abstract`, `keyphrases`(list), `prmu`(list) |

| Split | 문서 수 | 평균 keyphrase 수 | Present | Reordered | Mixed | Unseen |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 530,809 | 5.29 | 58.19% | 10.93% | 17.36% | 13.52% |
| Test | 20,000 | 5.28 | 58.40% | 10.84% | 17.20% | 13.56% |
| Validation | 20,000 | 5.27 | 58.20% | 10.94% | 17.26% | 13.61% |

### 3-2. 전처리 방법

- **Feature**: title, abstract
- **Target**: keyphrase, **PRMU** 라벨
  - PRMU는 gold keyphrase를 문서와의 관계에 따라 4범주로 나눈 태그다:
    - **P (Present)**: 원문에 형태소/어순 그대로 등장
    - **R (Reordered)**: 구성 단어는 원문에 있으나 어순이 다름
    - **M (Mixed)**: 원문 단어 일부 + 원문에 없는 단어가 혼합
    - **U (Unseen/absent)**: 원문에 전혀 등장하지 않는 표현
  - P/R은 추출(extractive) 계열이 담당 가능, M/U는 생성(generative) 계열이 필요 — hybrid 구조 설계의 직접적 근거가 되는 라벨

### 3-3. 표본추출 및 전처리

- KP20K 전체 학습셋(530,809건) 규모를 감안, 학습 단계에서는 전량 사용하되 평가는 20,000건 sample(`eval_docs=20000`)로 고정하여 baseline/proposed 간 공정 비교가 가능하도록 통제
- 정규화: 소문자화, Porter stemming (평가 시 매칭 기준과 통일)
- 중복 제거(dedup): 각 keyphrase를 임베딩 벡터로 변환 후 일정 표준편차 이내를 동일 의미로 간주해 병합 (라이브러리 기반)

### 3-4. 왜 효과가 있을 것으로 예상하는가? (검증 대상 가설)

- **H1.** present를 추출하고 absent를 생성하는 hybrid 모델이 성능 면에서 우수할 것이다.
- **H2.** 랭킹화는 pairwise 전략을 도입한다.
- **H3.** 도메인 특화 encoder(SciBERT)를 hybrid 구조의 present 추출기로 사용하면, 일반 encoder 기반 추출기보다 present 추출 정확도와 최종 F1/MAP/nDCG가 모두 향상될 것이다.
  - 근거: KP20K가 CS 논문 abstract 도메인에 특화되어 있어, 도메인 특화 사전학습(SciBERT)이 일반 encoder 대비 어휘·개념 표현에서 우위를 가질 것으로 예상

---

## 4. Experiment Design

### 4-1. 수행할 실험

| 실험 | 내용 | 대응 run_id |
| --- | --- | --- |
| E1. Baseline 재현 및 성능 측정 | 무학습(TF-IDF/KeyBERT/KeyBERT+MMR) + 지도학습(BART/KeyBART) 5종 재현 | B0, B1, B2, B3, B4 |
| E2. Proposed 학습 및 측정, candidate 생성(P/R, M/U) | Hybrid fusion 구조로 present(P/R)·absent(M/U) 동시 생성 | P1, P2 |
| E3. 랭킹화 및 중복 제거 기법 비교 | pairwise 랭킹 손실 도입, PRMU 조건부 실험, encoder 교체(SciBERT) | P5, P6, P7, P7+MMR |
| E4. 정성 분석 | 상위 예측 keyphrase 정성 비교, LLM(Qwen) 채점표 활용 | (전체 run 대상 사후 분석) |

### 4-2. 비교 분석

- **하이브리드 모델(present 추출 + absent 생성) vs. one-stop 모델(인코더-디코더 기반)**
  → P1(hybrid fusion, F1@10=0.1842)이 B3(BART one-stop, F1@10=0.1711)를 상회했고, absent_R@10은 P1(0.0323)이 B3(0.0132) 대비 약 2.4배 — hybrid 구조가 H1을 지지
- **랭킹화 전략: pairwise vs MMR**
  → pairwise 계열(P5, P6)이 F1@10에서 MMR 계열(P2)과 유사하거나 근소 우위(P5 F1@10=0.2479 vs P2 0.2105)를 보였고, 특히 absent_R@10에서 pairwise가 뚜렷이 높음(P5=0.0460, P6=0.0494 vs P2=0.0219) — 순위 학습이 absent 후보의 우선순위 조정에 특히 효과적임을 시사
- **Encoder 교체(SciBERT) 효과**
  → P7(SciBERT hybrid)이 전체 run 중 F1@10(0.2621), MAP@10(0.2659), nDCG@10(0.3975) 최고치 기록 — H3 지지

### 4-3. Evaluation Metrics 설계 및 검증

| 구분 | 지표 | 비고 |
| --- | --- | --- |
| Extractive (present) | Precision / Recall / F1 (@5, @10, @M) | 표준 stemmed exact-match |
| Abstractive (absent) | MAP@10, nDCG@10, absent_R@5/@10 | 순위 품질 + absent 회수율 별도 측정 |
| 다양성 | dup_ratio | 중복 제거 효과 정량화 |
| 생성량 | avg_preds_per_doc | 모델별 평균 예측 개수 — precision/recall 트레이드오프 해석 시 참조 |

---

## 5. Plan (일자별)

| **일자** | **수행 내용** | **산출물** |
| --- | --- | --- |
| 7/9 | 팀 구성 및 주제 후보 선택 | Keyphrase Extraction 선정 |
| 7/10 | EDA, 모델 전략, 후보 모델 비교 실험 | Baseline 노트북 |
| 7/13 | 랭킹화 전략, 평가지표 선정 | 1차 모델(P1, P2), 예측값 csv |
| 7/14 | pairwise 랭킹 손실 실험(P5, P6), PRMU 조건부 실험, SciBERT encoder 교체(P7), KeyBART 앙상블 시도 | 2차 모델(P5~P7), `final_all_systems` 평가지표 리스트 |
| 7/15 | 전체 run 비교 분석, MVP(P7) 결정, 멘토링 피드백 반영 | 리더보드 확정, MVP 선정 근거 문서 |
| 7/16 | 결과 정리, 발표자료 작성 | 시연 영상, PPT |
| 7/20 | 발표 진행 | 최종 발표 |

---

## 6. 실험 결과 (중간)

### 6-1. 전체 리더보드 (F1@10 기준 정렬)

| rank | run_id | 계열 | F1@5 | F1@10 | F1@M | present_F1@5 | absent_R@10 | MAP@10 | nDCG@10 | dup_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **P7_scibert_hybrid_full** | Hybrid+SciBERT | 0.2850 | **0.2621** | 0.2890 | 0.3619 | 0.0504 | **0.2659** | **0.3975** | 0.0 |
| 2 | P7_scibert_hybrid_mmr_full | Hybrid+SciBERT+MMR | 0.2695 | 0.2532 | 0.2762 | 0.3418 | 0.0481 | 0.2569 | 0.3844 | 0.0 |
| 3 | P5_pairwise_full | Hybrid+Pairwise | 0.2388 | 0.2479 | 0.2360 | 0.3109 | 0.0460 | 0.2133 | 0.3396 | 0.0 |
| 4 | P6_pairwise_prmu_full | Hybrid+Pairwise+PRMU | 0.2267 | 0.2443 | 0.2228 | 0.2980 | **0.0494** | 0.1994 | 0.3246 | 0.0 |
| 5 | P2_hybrid_fusion_mmr_full | Hybrid+MMR | 0.2560 | 0.2105 | 0.2662 | 0.3420 | 0.0219 | 0.2312 | 0.3429 | 0.0 |
| 6 | P1_hybrid_fusion_full | Hybrid(raw fusion) | 0.2070 | 0.1842 | 0.2201 | 0.3137 | 0.0323 | 0.1876 | 0.3012 | 0.0 |
| 7 | B3_bart_beam5_full | Baseline(BART) | 0.2445 | 0.1711 | 0.2653 | 0.3039 | 0.0132 | 0.2144 | 0.3088 | 0.0 |
| 8 | B4_keybart_beam5_full | Baseline(KeyBART) | 0.2377 | 0.1672 | 0.2632 | 0.2965 | 0.0097 | 0.2186 | 0.3111 | 0.0 |
| — | B0_tfidf_full | Baseline(무학습) | 0.1261* | 0.1142 | 0.1275 | 0.1734 | — | 0.0994 | 0.1770 | 0.0231 |
| — | B1_keybert_full | Baseline(무학습) | 0.0502* | 0.0601 | 0.0484 | 0.1369 | — | 0.0340 | 0.0767 | 0.0240 |
| — | B2_keybert_mmr_full | Baseline(무학습) | 0.0285* | 0.0270 | 0.0297 | 0.0590 | — | 0.0215 | 0.0421 | 0.0002 |

\* B0~B2의 F1@5는 `results_full_dataset` 산출값 (F1 컬럼, avg_preds_per_doc≈20 조건에서 계산)

### 6-2. 핵심 수치 요약 (최고 baseline 대비 최고 proposed 모델, B4 vs P7)

| 지표 | B4 (Baseline 최고) | P7 (Proposed 최고) | 개선폭 |
| --- | --- | --- | --- |
| F1@10 | 0.1672 | 0.2621 | **+56.8%** |
| MAP@10 | 0.2186 | 0.2659 | **+21.6%** |
| nDCG@10 | 0.3111 | 0.3975 | **+27.8%** |
| absent_R@10 | 0.0097 | 0.0504 | **약 5.2배** |
| dup_ratio | 0.0 | 0.0 | 유지 |

### 6-3. 해석

1. **H1 (hybrid 우위) 지지**: P1~P7 전 계열이 F1@10 기준으로 B3/B4를 상회 (P1 최저 0.1842도 B3 0.1711보다 높음)
2. **H2 (pairwise 랭킹) 지지**: absent_R@10에서 pairwise 계열(P5·P6)이 MMR 계열(P2) 대비 2배 이상 — 순위 학습이 특히 absent 후보 우선순위에 기여
3. **H3 (SciBERT encoder) 지지**: P7이 전 지표에서 최고치 — 도메인 특화 encoder 가설 확인
4. **주의**: dup_ratio가 대부분 run에서 0.0으로 수렴 — 후처리 dedup이 효과적으로 작동했으나, 동시에 이 지표만으로는 모델 간 다양성 차이를 변별하기 어려움 → 정성 분석(E4)으로 보완 필요
5. **MMR 적용 효과가 혼재**: P7 대비 P7+MMR은 모든 지표에서 소폭 하락 (F1@10 0.2621→0.2532) — dedup이 이미 충분히 이루어진 상태에서 MMR의 추가 다양성 페널티가 오히려 relevant한 후보를 밀어내는 것으로 추정, 후속 정성 분석 필요

### 6-4. 남은 검증 (Open Items)

- [ ] KeyBART ↔ P7(최고 proposed) 앙상블 결과의 정량 비교 (7/13 오후 시도 중, 본 리더보드에는 미반영)
- [ ] LLM(Qwen) 앙상블 채점표 기반 정성 평가 — 5개 run 상위 10개 keyphrase 비교
- [ ] Gold label 결측치(present인데 gold에서 누락된 케이스) 보정 후 재평가
- [ ] indifference margin 기준 P5/P6/P7 간 통계적 동등성 검정 (paired bootstrap)

---

## 7. 타당성 확보 (Validity)

### 가설: KP20K gold label 자체의 편향 가능성

| 찬성 논리 | 반박 논리 |
| --- | --- |
| 논문 저자 본인이 라벨링 → 논문을 가장 잘 아는 사람의 판단 | 인간 편향 존재 가능 — 검색량을 늘리려는 의도로 "AI", "NLP" 같은 과도하게 큰 범주어를 포함했을 가능성 |

**정량적 검증 방안**: LLM 앙상블 채점표
**정성적 검증 방안**: LLM 출력 10개 vs 본 모델(P7) 출력 10개 직접 비교

6-1의 실험 결과 자체도 간접적 타당성 근거가 된다 — gold label만을 정답으로 삼는 present_F1@5 지표에서조차 P7(0.3619)이 B3/B4(각 0.3039, 0.2965)를 상회한다는 것은, 단순히 "gold를 더 잘 외운" 결과가 아니라 구조적 개선(hybrid + domain encoder)의 효과로 해석할 수 있다.
