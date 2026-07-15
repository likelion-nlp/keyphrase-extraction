# 연구계획서

## 1.  Problem Definition (Needs)

기존의 문제 및 어떤 문제를 해결하고 왜 중요한가? 

### 1-1. “Vocabulary Mismatch Problem” 문제 해결

- 정보검색(IR) 분야에서 오래된 문제 중 하나가 어휘 불일치 문제 - 사용자가 찾고 싶은 개념과 문서에 쓰인 실제 단어가 다를 때 검색에 실패하는 현상

ex) “environment”가 없는데 “space”로 존재, “evolution”이 U로 분류 이 현상이 Vocabulary 문제의 예시(present-only 추출 방식은 문제를 근본적으로 해결하지 못하고, absent Keyphrase 생성 능력이 있어야 해결 가능)

### 1-2. 구체적 사용자군(Persona)별 Needs

| 사용자 | 겪는 문제 | 프로젝트가 해결하는 방식 |
| --- | --- | --- |
| 논문 검색/추천 서비스 이용자 | 논문을 다 읽지 않고 핵심 개념만 빠르게 파악 | 랭킹된 top-k 키프레이즈로 “이 논문이 뭘 다루는지”를 몇 단어로 즉시 파악 가능  |
| 일반 사용자 | 긴 요약문 대신 한눈에 들어오는 태그 형태 정보를 원함 | 요약보다 압축률이 높은 “개념 태그” 형태로 정보 제공  |

### 1-3. 어떤 문제를 해결하려고 하는가?

입력 문서 $x = (title, abstract)$가 주어졌을 때, 다음 세 조건을 동시에 만족하는 순서가 있는 키 프레이즈 집합 $Y = (y_1, y_2,...,y_k)$을 생성하는 문제

$$
f : x \mapsto Y = (y_1, \dots, y_k), \quad y_1 \succ y_2 \succ \dots \succ y_k
$$

본 문제를 만족하려면 아래 세 가지 하위 문제가 동시에 풀려야함

1. 표현 문제

$y_i \notin x$ (absent)인 경우도 생성 가능해야함

→ TextRank, KeyBERT 같은 추출 방식은 $y_i \in x$ 인 것만 가능 (구조적으로 절반의 정답(U, M의 일부)을 애초에 낼 수 없음

1. 우선 순위 문제

$Y$가 정렬되어야함( 단순 집합이 아니라 순서 있는 리스트 )

→ 대부분의 키워드 추출 Baseline은 정렬 없이 후보 집합만 반환 → “몇 개만 보여줘야 하는” 실제 사용 시나리오에 대응 못 함

1. 중복/다양성 문제

$Y$ 내에서 의미적으로 겹치는 항목 최소화 $sim(y_i,y_j)$가 낮아야 함

→ Seq2Seq beam search 종종 표현만 다른 유사 phrase를 상위권에 중복 생성함

---

## 2. Background & Baseline

- 기존에 연구들은 이 문제, 혹은 유사 문제에 대하여 어떻게 접근하였는가?
(논문 리서치. 단, 논문 리뷰는 생략합니다)
- 이번 프로젝트에서 사용할 baseline 모델
    - BART-base
    - KeyBART

### 2-1. 정의 (보완 필요)

- keyphrase 추출 기준, 중요도 랭킹의 의미
    - 문서를 잘 대표하는가?
    - 검색 인덱스로서 기능하는가? (rag
    - 검색량을 늘리기 위한 의도가 있는가?
- 

## 3. Proposed Method

### 3-1. EDA

kp20k EDA 

#### 3-2. 전처리 방법

- feature: title, abstract
- target: keyphrase, PRMU

### 3-4. 왜 효과가 있을 것으로 예상하는가?

- **H1.**
- **H2.**
- **H3.**

## 4. Experiment Design

### 4-1. 수행할 실험

- E1. Baseline 재현 및 성능 측정
- E2. Proposed 학습 및 측정 + 랭킹 도입
    - candidate 생성(P/R, M/U) -> 랭킹화 -> 중복 제거
- E3.
- E4. 정성 분석

### 4-2. 비교 분석

- 하이브리드 모델(present 추출, absent 생성) vs. one-stop 모델(인코더-디코더 기반)
- 랭킹화 전략: pairwise, MMR

### 4-3. Evaluation Metrics 설계 및 검증

- Extractive (present) - Precision/Recall/F1
- Abstractive (absent) - MAP, nDCG

## 5. Plan (일자별)

| **일자** | **수행 내용** | **산출물** |
| --- | --- | --- |
| 7/9 | 팀 구성 및 주제 후보 선택 | Keyphrase Extraction 선정 |
| 7/10 | EDA, 모델 전략, 후보 모델 비교 실험 | Baseline 노트북 |
| 7/13 | 랭킹화 전략, 평가지표 선정 | 1차 모델, 예측값 csv |
| 7/14 |  | 2차 모델 |
| 7/15 | 멘토링 |  |
| 7/16 | 결과 정리, 발표자료 작성 | 시연 영상, PPT |
| 7/20 | 발표 진행 | 최종 발표 |
