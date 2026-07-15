# 평가 지표 수식 레퍼런스

> `outputs/metrics/results_full_dataset.csv` · `final_experiment_table.csv`의 모든 컬럼 정의.
> 구현: [`src/metrics.py`](../src/metrics.py) · 검증: [`tests/test_metrics.py`](../tests/test_metrics.py) (38 tests)
> 실행 가능한 워크스루: [`01_data_and_evaluation.ipynb`](../notebooks/01_data_and_evaluation.ipynb).

---

## 0. 공통 전처리 — 모든 지표의 기반

문자열 비교 전에 예측·정답 모두 동일한 정규화를 거친다:

```text
lowercase → 공백 정리 → 토큰화(하이픈 단어는 한 토큰 유지: graph-based) → Porter stemming → 공백 join

"Neural Networks"  →  "neural network"(stem)
"neural network"   →  "neural network"(stem)   ← 같은 정규형 = 매칭 성공
```

- 정규형이 같은 예측은 **중복으로 간주해 하나만** 남긴다 (순서 유지).
- 이후 모든 매칭은 **정규형의 완전 일치(exact match)**.
- 이 규칙이 실험 전체에서 단 하나여야 모델 간 비교가 성립한다 (마스터 플랜 8.1절).

## 1. Precision@K · Recall@K · F1@K

컬럼: `P@5, P@10, R@5, R@10, F1@5, F1@10`

문서 하나에서 정규화·중복제거한 예측 상위 $K$개와 gold 집합 $Y$를 비교, $TP@K = |\hat{Y}_{1..K} \cap Y|$일 때:

$$P@K = \frac{TP@K}{K} \qquad R@K = \frac{TP@K}{|Y|} \qquad F1@K = \frac{2 \cdot P@K \cdot R@K}{P@K + R@K}$$

**핵심 규칙 (플랜 8.2)** — 예측이 $K$개보다 적어도 분모는 $K$를 유지한다:

| 상황 | 계산 | 이유 |
|---|---|---|
| 예측 3개, 전부 정답, K=5 | $P@5 = 3/5 = 0.6$ | 적게 출력해 정밀도만 올리는 꼼수 차단 |
| 예측 0개 | $P = R = F1 = 0$ | |
| gold가 빈 문서 | $R = 0$ 처리 (예외 없이) | |

## 2. F1@M

컬럼: `F1@M`

$K$를 고정하지 않고 **그 문서의 gold 개수** $M = |Y|$로 두는 F1@K:

$$F1@M = F1@K \big|_{K=|Y|}$$

KP20k는 문서당 정답 수가 평균 5.3개지만 1~20개까지 편차가 크다.
F1@5는 정답이 많은 문서에서 recall 상한이 깎이는데, F1@M은 이를 보정한다.

## 3. MAP@10 (Mean Average Precision)

컬럼: `MAP@10`

정답이 **앞 순위**에 있을수록 높다. 상위 10개를 순회하며 $i$번째 예측이 정답일 때마다 그 시점의 precision을 누적:

$$AP@10 = \frac{1}{\min(|Y|,\,10)} \sum_{\substack{i=1 \\ \hat{y}_i \in Y}}^{10} \frac{\mathrm{hits}_{\le i}}{i}$$

$$MAP@10 = \frac{1}{N}\sum_{d=1}^{N} AP@10^{(d)}$$

예: 정답이 1·3위에 있으면 $AP = \frac{1}{2}(\frac{1}{1} + \frac{2}{3}) = 0.833$, 2·4위면 $0.5$ — 같은 개수를 맞혀도 순위가 늦으면 점수가 낮다.

## 4. nDCG@10 (binary)

컬럼: `nDCG@10`

로그 할인 기반 순위 지표. 모든 gold의 관련도를 1로 두는 binary 버전:

$$DCG@10 = \sum_{\substack{i=1 \\ \hat{y}_i \in Y}}^{10} \frac{1}{\log_2(i+1)} \qquad nDCG@10 = \frac{DCG@10}{IDCG@10}$$

$IDCG@10$은 정답 $\min(|Y|, 10)$개를 전부 맨 앞에 놓았을 때의 이상값.

> ⚠️ **해석 주의 (플랜 8.5)**: KP20k gold에는 중요도 등급이 없다.
> 이 지표는 "정답이 상위에 배치되는가"까지만 말하며, "더 중요한 것이 더 위에 있는가"는 말할 수 없다.

## 5. Present / Absent 분리 지표

컬럼: `present_F1@5, absent_R@5, absent_R@10`

**gold 분리** — 공식 `prmu` 라벨 사용 (엄격 기준):

$$Y_{present} = \{y : prmu(y) = P\} \qquad Y_{absent} = \{y : prmu(y) \in \{R, M, U\}\}$$

**예측 분리** — 예측 phrase의 stem 시퀀스가 제목+초록의 stem 시퀀스에 **같은 순서로 연속 등장**하면 present:

$$\hat{y} \in \hat{Y}_{present} \iff \mathrm{stems}(\hat{y}) \sqsubseteq_{\text{연속}} \mathrm{stems}(\text{title} + \text{abstract})$$

그다음 $\hat{Y}_{present}$ vs $Y_{present}$, $\hat{Y}_{absent}$ vs $Y_{absent}$를 각각 1번 수식으로 계산한다.
Absent는 맞히는 수가 적어 F1보다 **Recall을 주 지표**로 본다 (플랜 8.3).

## 6. PRMU 유형별 Recall

컬럼: `recall_P, recall_R, recall_M, recall_U`

유형 $T \in \{P, R, M, U\}$의 gold 중 정규형이 예측 집합에 포함된 비율:

$$recall_T = \frac{\left|\{y \in Y_T : \mathrm{norm}(y) \in \mathrm{norm}(\hat{Y})\}\right|}{|Y_T|}$$

| 유형 | 정의 (Boudin & Gallina 2021) |
|---|---|
| **P** resent | stem이 원문에 같은 순서로 연속 등장 |
| **R** eordered | stem이 모두 원문에 있으나 연속·동일 순서 아님 |
| **M** ixed | 일부 stem만 원문에 존재 |
| **U** nseen | 어떤 stem도 원문에 없음 |

> ⚠️ **비교 주의**: 이 지표는 $K$ 잘림 없이 **모델이 반환한 리스트 전체**를 쓴다.
> 리스트 길이가 모델마다 다르다 — TF-IDF/KeyBERT 20개, BART/KeyBART ≈5개, **P1(fusion)은 후보 전체 ≈45개**, P2는 10개.
> 따라서 P1의 `recall_P=0.76`은 "후보 pool 커버리지"로 읽어야 하며,
> 모델 간 공정한 절대 비교에는 $K$가 통일된 `R@5, R@10`을 사용한다.

## 7. 다양성 지표

### dup_ratio (stem 중복률) — 컬럼: `dup_ratio`

$$dup = 1 - \frac{|\mathrm{unique}(\mathrm{norm}(\hat{Y}))|}{|\hat{Y}|}$$

`neural network` / `neural networks`처럼 stem이 같은 반복 출력을 포착. 0이면 중복 없음.

### semantic_redundancy — `results_mmr_sweep_full.csv`

예측들의 SBERT 임베딩 쌍별 cosine 평균 (대각 제외):

$$Red = \frac{1}{N(N-1)} \sum_{i \neq j} \cos(e_i, e_j)$$

stem이 달라도 의미가 겹치는 중복(`neural network` vs `deep neural model`)까지 포착한다.
낮을수록 다양하지만, **지나치게 낮으면 서로 무관한 출력**일 수 있으니 F1과 함께 해석 (플랜 8.6).

## 8. Candidate Recall — `results_candidate_recall_full.csv`

후보 pool 상위 $K$개가 gold를 얼마나 덮는지:

$$candR@K = \frac{|\mathrm{norm}(C_{1..K}) \cap \mathrm{norm}(Y)|}{|Y|}$$

**의미 (플랜 11.2)**: 어떤 reranker도 후보에 없는 정답을 올릴 수 없으므로, 이 값은 **선택 단계 성능의 상한선**이다.
Union(추출∪생성, 인터리브)이 개별 소스보다 높다는 것이 하이브리드 설계의 정량 근거.

| 소스 | @20 | @50 | 해석 |
|---|---|---|---|
| KeyBERT | 0.171 | 0.222 | present 표면 표현 담당 |
| Generator | 0.412 | 0.420 | present+absent 모두 |
| **Union** | **0.421** | **0.464** | 상호 보완 확인 (H4) |

## 9. 집계: Macro 평균

CSV의 모든 값은 **macro 평균** — 문서별로 지표를 계산한 뒤 20,000 문서에 단순 평균:

$$\mathrm{Macro} = \frac{1}{N} \sum_{d=1}^{N} \mathrm{metric}^{(d)}$$

micro(전체 TP/FP를 합산)는 키프레이즈가 많은 문서에 가중치가 쏠리므로 보조로만 쓴다 (플랜 8.8).

---

## 부록: 컬럼 → 수식 빠른 매핑

| CSV 컬럼 | 수식 절 | 잘림 K | 비고 |
|---|---|---|---|
| `P@5, P@10, R@5, R@10` | §1 | 5 / 10 | 분모 K 고정 |
| `F1@5, F1@10` | §1 | 5 / 10 | |
| `F1@M` | §2 | 문서별 \|Y\| | |
| `MAP@10, nDCG@10` | §3, §4 | 10 | binary relevance |
| `present_F1@5` | §5 | 5 | gold=prmu P, 예측=연속등장 판정 |
| `absent_R@5, absent_R@10` | §5 | 5 / 10 | gold=R∪M∪U |
| `recall_P/R/M/U` | §6 | **없음(리스트 전체)** | 모델 간 비교 주의 |
| `dup_ratio` | §7 | 없음 | stem 기준 |
| `avg_preds_per_doc` | — | 없음 | 문서당 출력 개수 평균 |
