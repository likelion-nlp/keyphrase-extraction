# KP20K 기반 Keyphrase / Keyword Extraction·Generation 프로젝트 마스터 플랜

> 작성일: 2026-07-11  
> 목적: **이해도가 낮은 팀원도 “무엇을, 왜, 어떤 순서로 구현하는지” 알 수 있도록** 문제 정의부터 주피터 노트북 구현 순서, 베이스라인, 제안 방법론, 평가 및 실험 설계까지 하나의 문서로 정리한다.

---

## 0. 먼저 결론부터

이 프로젝트는 단순히 문서에 있는 단어를 몇 개 골라내는 문제가 아니다.

입력으로 논문의 **제목(title)과 초록(abstract)** 을 받고, 출력으로 다음 조건을 만족하는 키프레이즈 목록을 만드는 문제다.

1. **원문에 실제로 등장하는 핵심 표현을 찾아야 한다.**  
   예: 원문에 `neural network`가 있으면 이를 찾아낸다.
2. **원문에 그대로 등장하지 않더라도 문서의 핵심 개념을 생성할 수 있어야 한다.**  
   예: 원문에 `machine learning`이라는 표현이 없더라도 내용상 적절하면 생성한다.
3. **가장 중요한 후보가 앞에 오도록 정렬해야 한다.**
4. **비슷하거나 중복된 표현을 여러 번 내놓지 않아야 한다.**  
   예: `neural network`, `neural networks`, `deep neural network`만 반복해서 출력하지 않는다.

따라서 최종적으로는 다음 네 기능을 연결해야 한다.

```text
문서 이해
   ↓
후보 추출·생성
   ↓
관련도 기반 재정렬
   ↓
중복 제거 및 Top-K 선택
```

이 문서에서 권장하는 최종 구조는 다음과 같다.

```text
                   ┌─ Extractive 후보: TF-IDF / KeyBERT / BERT-KPE
Title + Abstract ──┤
                   └─ Generative 후보: BART / T5 / KeyBART
                                  ↓
                        후보 통합·정규화
                                  ↓
                    Relevance Reranker 또는 점수 결합
                                  ↓
                          MMR 중복 제거
                                  ↓
                  Ranked Present + Absent Keyphrases
```

프로젝트 진행 순서는 다음과 같다.

```text
데이터 검증
→ 평가 함수 고정
→ 가장 단순한 추출형 베이스라인
→ 생성형 베이스라인
→ 강한 생성형 베이스라인
→ 후보 통합
→ 랭킹
→ 중복 제거
→ 비교 실험
→ 오류 분석
```

---

# 1. 이 Task를 아주 쉽게 이해하기

## 1.1 Keyphrase는 무엇인가?

문서 전체를 대표하는 짧은 개념 표현이다.

예를 들어 다음과 같은 초록이 있다고 하자.

```text
We propose a convolutional neural network for classifying medical images.
The model improves diagnostic accuracy using transfer learning.
```

가능한 키프레이즈는 다음과 같다.

```text
convolutional neural network
medical image classification
transfer learning
diagnostic accuracy
computer-aided diagnosis
```

여기서 앞의 네 개는 원문에 있는 단어를 주로 활용하지만, `computer-aided diagnosis`는 원문에 정확히 존재하지 않아도 문서의 의미를 요약하는 적절한 개념일 수 있다.

즉, 키프레이즈 시스템은 다음 두 능력이 모두 필요하다.

- **Extraction:** 원문에서 중요한 표현을 골라내는 능력
- **Generation:** 원문에 없지만 의미적으로 필요한 표현을 만들어내는 능력

---

## 1.2 Keyword와 Keyphrase의 차이

- **Keyword:** 보통 한 단어 중심  
  예: `classification`, `network`
- **Keyphrase:** 한 단어 또는 여러 단어로 이루어진 핵심 구  
  예: `image classification`, `neural network`

KP20k에는 여러 단어로 구성된 표현이 많으므로, 이 프로젝트에서는 **Keyphrase Prediction**이라는 표현이 더 정확하다.

---

## 1.3 이 문제를 수식으로 표현하면

입력 문서를 다음처럼 정의한다.

$$
x = (\text{title}, \text{abstract})
$$

모델은 키프레이즈 목록을 출력한다.

$$
f(x) = Y = (y_1, y_2, \dots, y_k)
$$

여기서 프로젝트가 원하는 조건은 다음과 같다.

$$
\operatorname{rel}(y_1, x) \ge \operatorname{rel}(y_2, x) \ge \dots \ge \operatorname{rel}(y_k, x)
$$

즉 앞의 키프레이즈일수록 문서와 관련성이 높아야 한다.

그리고 서로 지나치게 비슷한 출력은 줄여야 한다.

$$
\operatorname{sim}(y_i, y_j) \text{가 지나치게 높지 않도록 한다.}
$$

중요한 점은 **KP20k의 정답 키프레이즈가 “중요도 점수”를 가진 것은 아니라는 사실**이다. 따라서 실제 학습 목표는 다음처럼 나누어 생각해야 한다.

1. 정답 키프레이즈인지 판단하는 **관련성 문제**
2. 후보를 앞에서부터 보여주는 **랭킹 문제**
3. 서로 겹치는 후보를 줄이는 **다양성 문제**

---

# 2. KP20k 데이터셋 이해

## 2.1 공식 데이터 정보

Hugging Face 데이터셋:

- https://huggingface.co/datasets/taln-ls2n/kp20k

공식 데이터 카드 기준 전체 문서는 570,809개이며 다음과 같이 나뉜다.

| Split | 문서 수 | 문서당 평균 키프레이즈 수 | P | R | M | U |
|---|---:|---:|---:|---:|---:|---:|
| Train | 530,809 | 5.29 | 58.19% | 10.93% | 17.36% | 13.52% |
| Validation | 20,000 | 5.27 | 58.20% | 10.94% | 17.26% | 13.61% |
| Test | 20,000 | 5.28 | 58.40% | 10.84% | 17.20% | 13.56% |

필드는 다음과 같다.

| 필드 | 설명 |
|---|---|
| `id` | 문서 식별자 |
| `title` | 논문 제목 |
| `abstract` | 논문 초록 |
| `keyphrases` | 정답 키프레이즈 리스트 |
| `prmu` | 각 키프레이즈의 P/R/M/U 라벨 |

---

## 2.2 PRMU를 쉽게 이해하기

입력 문서에 다음 단어들이 있다고 가정하자.

```text
deep neural networks are used for image classification
```

| 분류 | 의미 | 예시 |
|---|---|---|
| **P: Present** | 키프레이즈가 원문에 같은 순서로 연속 등장 | `image classification` |
| **R: Reordered** | 구성 단어는 모두 있지만 순서가 달라짐 | `classification image` |
| **M: Mixed** | 일부 단어는 원문에 있고 일부는 없음 | `medical image classification` |
| **U: Unseen** | 구성 단어가 원문에 없음 | `computer vision` |

실험에서는 두 가지 관점으로 나눌 수 있다.

### 엄격 기준

```text
Present = P
Absent = R + M + U
```

### 느슨한 기준

```text
표면 단어 활용 = P + R + M
완전히 새로운 개념 확장 = U
```

주 평가에서는 기존 연구와 비교하기 쉽게 **P 대 non-P**를 사용하고, 보조 분석에서는 P/R/M/U를 각각 보고하는 것을 권장한다.

---

## 2.3 팀 연구계획서에서 반드시 수정해야 할 부분

### 수정 1. 입력은 `abstract`만이 아니라 `title + abstract`여야 한다

팀 계획에는 다음과 같은 변환이 있었다.

```text
input: abstract
output: kp1; kp2; kp3
```

그러나 KP20k의 P 판정은 제목과 초록을 함께 기준으로 한다. 제목에만 있는 핵심어를 놓치지 않으려면 입력은 다음처럼 구성하는 편이 맞다.

```text
title: {title} abstract: {abstract}
```

또는 다음과 같이 특별 토큰을 사용한다.

```text
<title> {title} <abstract> {abstract}
```

---

### 수정 2. `prmu`가 이미 제공되므로 처음부터 다시 분류하지 않는다

데이터셋에 `prmu` 필드가 있으므로 기본 실험에서는 이를 그대로 사용한다.

직접 PRMU를 다시 계산하면 토큰화, 하이픈, 어간 추출 규칙에 따라 공식 라벨과 다른 결과가 생길 수 있다.

직접 재분류는 다음 경우에만 한다.

- 공식 라벨 생성 방식을 재현하는 검증 실험
- 모델이 생성한 예측 키프레이즈를 P/R/M/U로 분류할 때
- 전처리 방식에 따른 라벨 민감도를 분석할 때

---

### 수정 3. 팀에서 계산한 비율과 공식 비율이 다르다

팀 문서에는 P가 약 60.78%로 기록되어 있지만 공식 데이터 카드의 전체적인 P 비율은 약 58.2%다.

가능한 원인은 다음과 같다.

- 특정 split만 계산함
- 중복 키프레이즈를 제거함
- 빈 값 또는 길이 제한 샘플을 제외함
- 직접 만든 PRMU 판정 함수가 공식 방식과 다름
- title을 포함하거나 제외하는 방식이 다름
- 토큰화와 stemming 방식이 다름

따라서 EDA 노트북에서는 다음을 함께 출력해야 한다.

```text
1. 공식 prmu 필드로 계산한 통계
2. 팀 자체 함수로 계산한 통계
3. 두 라벨이 다른 사례 50개
```

---

### 수정 4. 하이픈 처리 설명을 조심해야 한다

공식 데이터 카드는 spaCy 토큰화 시 `graph-based`처럼 하이픈을 포함한 단어를 쪼개지 않도록 특별 규칙을 사용했다고 설명한다.

따라서 `virtual-reality`가 하이픈 때문에 무조건 U가 된다는 설명은 공식 전처리와 맞지 않을 수 있다.

정확한 표현은 다음과 같다.

> 커스텀 토크나이저가 하이픈을 분리하면 PRMU 판정이 공식 라벨과 달라질 수 있으므로, 공식 데이터 생성 규칙을 재현하거나 제공된 `prmu`를 사용해야 한다.

---

### 수정 5. 키프레이즈 순서는 중요도 순위가 아니다

가장 중요한 수정 사항이다.

공식 데이터 카드에서는 P 키프레이즈가 **제목+초록에 등장하는 순서**로 정렬되어 있다고 설명한다.

따라서 다음 해석은 성립하지 않는다.

```text
첫 번째 정답 키프레이즈가 가장 중요하다.
두 번째 정답 키프레이즈가 그다음으로 중요하다.
```

올바른 해석은 다음과 같다.

```text
정답 keyphrases는 관련 키프레이즈의 집합이다.
일부 저장 순서는 학습 편의를 위한 규칙일 뿐, 중요도 정답이 아니다.
```

이 데이터만으로는 “A가 B보다 더 중요하다”라는 진짜 랭킹 라벨을 직접 학습할 수 없다.

따라서 본 프로젝트의 랭킹은 다음 중 하나로 정의해야 한다.

1. **모델 관련도 확률 순위**
2. **문서-키프레이즈 의미 유사도 순위**
3. **생성 확률과 추출 점수를 합친 순위**
4. 별도의 사람 평가로 만든 중요도 순위
5. 검색 성능을 가장 많이 높이는 순위

이번 프로젝트에서는 1~3을 구현하고, 4~5는 확장 실험으로 둔다.

---

## 2.4 데이터셋의 한계

KP20k는 대규모라는 장점이 있지만 다음 한계가 있다.

- 주로 컴퓨터과학 논문으로 구성되어 다른 분야에 바로 일반화되지 않을 수 있다.
- 저자가 선택한 키프레이즈는 가능한 정답의 일부일 뿐이다.
- 모델이 적절한 동의어를 생성해도 정답 목록에 없으면 exact match에서 오답이 된다.
- 키프레이즈마다 중요도 등급이 없다.
- 원문 전체가 아니라 제목과 초록만 제공된다.
- absent 키프레이즈는 표현 확장에 유용할 수 있지만 환각 위험도 있다.

이 때문에 exact-match F1만으로 성능을 판단하지 않고 의미 기반 평가와 다양성 평가를 함께 사용해야 한다.

---

# 3. 팀이 제시한 기술 스택은 어떻게 연결되는가?

팀이 제시한 기술은 서로 경쟁하는 하나의 모델 목록이 아니라, 파이프라인의 서로 다른 역할을 담당한다.

| 기술 | 역할 | Present | Absent | 학습 필요 | 이 프로젝트에서 위치 |
|---|---|---:|---:|---:|---|
| TF-IDF | 빈도 기반 후보 점수 | 가능 | 불가능 | 불필요 | 가장 낮은 하한선 |
| TextRank | 그래프 기반 중요 단어/구 추출 | 가능 | 불가능 | 불필요 | 비학습 추출 베이스라인 |
| KeyBERT | 문서와 후보 임베딩 유사도 | 가능 | 불가능 | 보통 불필요 | 강한 간편 추출 베이스라인 |
| BERT-JointKPE | 후보 구간 추출+랭킹 | 가능 | 불가능 | 필요 | 지도학습 추출 베이스라인 |
| T5/BART | Seq2Seq 생성 | 가능 | 가능 | 필요 | 기본 생성 베이스라인 |
| KeyBART | 키프레이즈 생성에 맞춰 사전학습된 BART | 가능 | 가능 | fine-tuning 권장 | 강한 실용 생성 베이스라인 |
| Beam Search | 모델이 아니라 디코딩 방법 | 가능 | 가능 | 해당 없음 | 후보 생성량 조절 |
| Sentence-BERT | 의미 유사도 계산 | 해당 없음 | 해당 없음 | 보통 불필요 | 재정렬·중복 판단 |
| MMR | 관련성과 다양성의 균형 | 해당 없음 | 해당 없음 | 불필요 | 최종 중복 제거 |
| Cross-Encoder | 문서-후보 관련도 재평가 | 모두 | 모두 | 선택적 | 최종 랭커 |
| Optimizer | 손실을 줄이도록 파라미터 갱신 | 간접 | 간접 | 학습 시 사용 | 중복 제거 기능 자체는 아님 |

---

## 3.1 잘못 이해하기 쉬운 부분

### “Optimizer로 중복을 줄인다”는 표현은 정확하지 않다

Optimizer는 주어진 loss를 줄이는 도구일 뿐이다.

```text
loss에 중복 패널티가 없음
→ optimizer도 중복을 줄일 이유가 없음
```

중복을 줄이려면 다음 중 하나가 명시적으로 필요하다.

- MMR 후처리
- n-gram blocking
- repetition penalty
- unlikelihood loss
- coverage 또는 exclusion loss
- One2Set처럼 집합 단위로 생성하는 구조
- 중복 후보에 낮은 점수를 주는 reranker

---

### `T5 + MMR`는 하나의 모델명이 아니다

정확한 구조는 다음과 같다.

```text
T5가 후보를 여러 개 생성
→ 후보를 phrase 단위로 분리
→ SBERT로 관련도와 후보 간 유사도 계산
→ MMR로 일부 후보 선택
```

MMR은 T5 내부 학습이 아니라 보통 **추론 후 후처리 단계**다.

---

### Beam Search가 자동으로 좋은 랭킹을 만들어 주는 것은 아니다

Beam search의 score는 생성 시퀀스 확률이다.

문제는 다음과 같다.

- 긴 문장은 토큰 확률을 여러 번 곱하므로 불리해질 수 있다.
- 전체 `kp1; kp2; kp3` 시퀀스 점수와 각 phrase의 중요도는 다르다.
- 모델이 학습한 정답 순서가 중요도 순서가 아니면 앞에 생성됐다고 더 중요한 것이 아니다.

따라서 phrase 점수는 최소한 길이 정규화를 해야 한다.

$$
s_{gen}(y)=\frac{1}{|y|^\alpha}\sum_{t=1}^{|y|}\log P(y_t\mid y_{<t},x)
$$

하지만 이것도 “생성하기 쉬운 표현”의 점수에 가깝고 “사용자에게 가장 중요한 표현”과 완전히 같지는 않다.

---

# 4. 팀이 만든 학습 로직을 정확히 이해하기

## 4.1 학습 시점: 토큰 단위 예측

정답 키프레이즈를 하나의 문자열로 만든다고 하자.

```text
machine learning ; neural network ; classification
```

토크나이저를 통과하면 다음처럼 토큰 시퀀스가 된다.

$$
y=(y_1,y_2,\dots,y_T)
$$

모델은 매 시점에서 다음 토큰을 맞힌다.

$$
P(y_t \mid y_{<t},x)
$$

Loss는 정답 토큰의 확률을 높이는 negative log-likelihood다.

$$
\mathcal{L}_{NLL}=-\sum_{t=1}^{T}\log P(y_t\mid y_{<t},x)
$$

쉽게 말하면 다음 과정을 반복한다.

```text
입력 문서 + 지금까지의 정답 토큰
→ 다음 정답 토큰 맞히기
```

학습 시에는 보통 **teacher forcing**을 사용한다.

```text
실제 이전 정답 토큰을 모델에 넣어 다음 토큰을 예측
```

추론 시에는 정답이 없으므로 모델이 방금 생성한 토큰을 다시 입력으로 사용한다.

```text
모델의 이전 예측을 넣어 다음 토큰을 예측
```

이 차이 때문에 학습 loss는 낮지만 실제 생성은 흔들리는 **exposure bias**가 생길 수 있다.

---

## 4.2 왜 학습은 토큰인데 평가는 phrase인가?

모델은 문장을 토큰 단위로 생성해야 하므로 학습은 토큰 단위다.

하지만 사용자가 원하는 것은 토큰 정확도가 아니라 완성된 키프레이즈다.

예를 들어 정답이 `neural network`일 때 다음 예측은 토큰 일부는 맞았지만 키프레이즈 exact match에서는 오답이다.

```text
정답: neural network
예측: neural model
```

따라서 추론 후에는 separator로 나누어 phrase 집합으로 변환한다.

```python
raw_output = "machine learning <sep> neural network <sep> classification"
predicted_phrases = raw_output.split("<sep>")
```

그리고 정답 집합과 비교한다.

$$
Y=\{y_1,y_2,\dots,y_m\}
$$

$$
\hat{Y}=\{\hat{y}_1,\hat{y}_2,\dots,\hat{y}_n\}
$$

---

## 4.3 팀 로직의 좋은 점

- T5/BART류 모델에 바로 적용할 수 있다.
- present와 absent를 한 모델이 함께 생성할 수 있다.
- Hugging Face `Seq2SeqTrainer`로 구현하기 쉽다.
- 평균적으로 5개 안팎인 KP20k 출력에 적합하다.
- KeyBART가 사용하는 concatenated keyphrase sequence와 방향이 유사하다.

---

## 4.4 팀 로직의 문제점

### 문제 1. 키프레이즈는 집합인데 학습은 순서에 민감하다

다음 두 출력은 집합으로는 동일하다.

```text
A; B; C
C; A; B
```

하지만 Seq2Seq loss에서는 완전히 다른 시퀀스다.

즉 잘못된 순서 편향이 들어간다.

해결 방법:

- 초기 베이스라인에서는 하나의 canonical order를 고정
- 학습 중 일부 샘플에서 phrase 순서를 shuffle하는 augmentation 실험
- 연구 확장으로 One2Set 적용

---

### 문제 2. Present → Absent 순서를 “중요도 순서”로 해석하면 안 된다

Present를 먼저 생성하면 모델이 extraction을 먼저 하고 generation을 나중에 하도록 구조화할 수는 있다.

그러나 이것은 중요도 랭킹이 아니다.

권장 target 형식:

```text
<present> kp_p1 <sep> kp_p2 <absent> kp_a1 <sep> kp_a2
```

이렇게 하면 모델에게 타입 경계를 알려줄 수 있다.

하지만 최종 화면의 순위는 별도의 점수로 다시 정렬해야 한다.

---

### 문제 3. 구분자 `;`는 일반 문자다

세미콜론은 원문이나 phrase 내부에도 드물게 나타날 수 있다.

특별 토큰 사용을 권장한다.

```text
<kp_sep>
<present>
<absent>
```

토크나이저에 special token으로 추가한다.

---

### 문제 4. 토큰 loss는 긴 phrase에 더 많은 비중을 줄 수 있다

`deep learning`은 2개 토큰, `convolutional neural network architecture`는 더 많은 토큰을 가진다.

기본 loss는 긴 phrase가 더 많은 loss 항을 만든다.

초기 베이스라인에서는 그대로 사용하되, 확장 실험으로 다음을 고려한다.

- phrase 단위 loss 정규화
- present/absent loss 가중치
- absent phrase oversampling
- focal loss 또는 unlikelihood loss

---

### 문제 5. One2Seq에서 beam search 결과를 잘못 해석하기 쉽다

두 가지 생성 방식이 있다.

#### One2One

```text
한 beam = 한 키프레이즈
```

여러 beam을 만들면 각 beam을 후보 phrase로 사용할 수 있다.

#### One2Seq

```text
한 beam = kp1; kp2; kp3가 들어 있는 전체 목록
```

이때 beam 10개는 phrase 10개가 아니라 **키프레이즈 목록 후보 10개**다.

팀 계획의 학습 형식은 One2Seq이므로, 기본 추론은 다음처럼 하는 것이 자연스럽다.

```text
가장 좋은 전체 시퀀스 1개 생성
→ separator로 분리
→ phrase별 후처리·재정렬
```

후보 pool을 크게 만들고 싶다면 다음 방법을 사용한다.

- 여러 beam의 전체 시퀀스를 모두 분리해 후보 통합
- diverse beam search
- top-p sampling 여러 회
- One2One 별도 모델
- One2Set
- DeSel과 같은 decode-then-select 방식

---

# 5. 프로젝트의 연구 질문

## 5.1 최종 연구 질문

> KP20k의 제목과 초록을 입력으로 받아, 원문에 존재하는 표현과 원문에 없는 추상적 표현을 모두 예측하고, 관련성이 높은 순서로 정렬하면서 의미 중복을 최소화하는 end-to-end keyphrase prediction system을 어떻게 구축할 것인가?

---

## 5.2 하위 연구 질문

1. 추출형 모델은 어느 정도까지 present keyphrase를 찾을 수 있는가?
2. 생성형 모델은 absent keyphrase recall을 얼마나 개선하는가?
3. 일반 BART/T5보다 키프레이즈 사전학습 모델인 KeyBART가 유리한가?
4. 생성 확률만 사용한 순위보다 문서-phrase reranker가 더 좋은가?
5. MMR이 exact-match 성능을 크게 잃지 않고 중복을 줄이는가?
6. title을 포함했을 때 성능이 얼마나 달라지는가?
7. exact-match F1과 semantic evaluation의 모델 순위가 같은가?
8. P/R/M/U 중 어떤 유형이 가장 어려운가?

---

## 5.3 검증할 가설

### H1. 추출형 모델의 한계

KeyBERT와 TextRank는 present 성능은 확보하지만 구조적으로 U를 생성할 수 없으므로 전체 recall에 한계가 있을 것이다.

### H2. 생성형 모델의 이점

BART/T5는 absent keyphrase를 생성하여 전체 recall과 U recall을 개선할 것이다.

### H3. KeyBART의 이점

일반 BART보다 키프레이즈 생성 목적에 맞게 사전학습된 KeyBART가 적은 학습량으로 더 좋은 성능을 낼 것이다.

### H4. Hybrid의 이점

추출 후보와 생성 후보를 통합하면 present precision과 absent recall의 균형이 좋아질 것이다.

### H5. Reranking의 이점

생성 순서나 beam score만 사용하는 것보다 문서-후보 관련도 모델로 재정렬하면 Precision@5와 nDCG@5가 개선될 것이다.

### H6. MMR의 이점

MMR은 약간의 recall 손실 가능성이 있지만 semantic duplicate rate를 낮추고 사용자 관점의 출력 품질을 개선할 것이다.

---

# 6. 구현할 모델 계층

## 6.1 Baseline 0: Random / Frequency Oracle Check

모델을 만들기 전에 평가 코드가 정상인지 확인하는 용도다.

- 무작위 후보
- 전체 학습 데이터에서 자주 나오는 키프레이즈 Top-K
- gold를 그대로 넣었을 때 F1=1인지 확인하는 oracle test
- prediction을 비웠을 때 F1=0인지 확인

이 단계는 성능 경쟁용이 아니라 **평가 함수 단위 테스트**다.

---

## 6.2 Baseline 1: TF-IDF n-gram

### 목적

가장 단순한 통계 기반 하한선이다.

### 방법

1. title+abstract에서 1~3gram 후보 생성
2. corpus 전체 document frequency 계산
3. 각 문서에서 TF-IDF가 높은 후보 선택
4. 명사구 필터를 선택적으로 적용

### 장점

- 매우 빠름
- 결과를 설명하기 쉬움
- 랭킹 점수가 기본 제공됨

### 단점

- absent 생성 불가능
- 단순 빈도와 희귀도에 의존
- 문맥 의미를 이해하지 못함

---

## 6.3 Baseline 2: TextRank 또는 YAKE

### 목적

비지도 그래프/통계 기반 추출 성능을 본다.

### 권장

시간이 부족하면 TF-IDF와 TextRank를 모두 하지 말고 **TF-IDF + KeyBERT**만 구현해도 된다.

---

## 6.4 Baseline 3: KeyBERT + MMR

### 목적

간단하지만 의미 임베딩을 사용하는 강한 추출 베이스라인이다.

### 구조

```text
문서
→ CountVectorizer로 1~3gram 후보 생성
→ 문서 embedding과 후보 embedding 계산
→ cosine similarity로 랭킹
→ MMR로 중복 제거
```

### 주의

KeyBERT는 후보가 원문에 있어야 하므로 U를 생성하지 못한다.

따라서 다음처럼 보고한다.

```text
전체 F1
P-only F1
non-P recall은 구조적 하한으로 해석
```

---

## 6.5 Baseline 4: BART-base 또는 T5-base One2Seq

### 목적

일반 사전학습 Seq2Seq 모델이 이 task를 얼마나 학습할 수 있는지 확인한다.

### 권장 선택

- 빠른 실험: `google-t5/t5-small`
- 본 실험: `facebook/bart-base` 또는 `google-t5/t5-base`

둘을 모두 학습할 필요는 없다. 프로젝트 시간이 짧으면 BART-base 하나를 일반 생성 베이스라인으로 둔다.

---

## 6.6 Baseline 5: KeyBART Fine-tuning

Hugging Face 모델:

- https://huggingface.co/bloomberg/KeyBART

KeyBART는 BART 구조를 키프레이즈 시퀀스를 복원하도록 사전학습한 모델이다.

### 권장 이유

- 이 task와 목적이 직접적으로 맞는다.
- Hugging Face에서 바로 불러올 수 있다.
- 기존 팀 계획의 One2Seq 방식과 잘 맞는다.
- 일반 BART 대비 강한 실용 baseline이 된다.

### 사용 위치

```text
일반 생성 baseline: BART-base
강한 생성 baseline: KeyBART
```

---

## 6.7 선택 Baseline: BERT-JointKPE

논문 및 코드:

- https://github.com/thunlp/BERT-KPE
- https://arxiv.org/abs/2004.13639

이 모델은 chunking과 ranking을 함께 학습한다.

장점은 지도학습 extractive 모델의 강한 기준점이라는 것이다.

단점은 기존 코드가 현재 Hugging Face Trainer 기반 코드보다 복잡하고, absent 생성이 불가능하다는 것이다.

시간이 촉박하면 구현 우선순위를 낮춘다.

---

# 7. 최종 제안 방법론: Generate–Extract–Rerank–Diversify

이 프로젝트에서 가장 현실적이고 설명하기 좋은 제안 방법이다.

이하에서는 편의상 **GERD 파이프라인**이라고 부른다.

```text
G: Generate
E: Extract
R: Rerank
D: Diversify
```

---

## 7.1 Step 1. 입력 인코딩

```text
<title> 논문 제목 <abstract> 논문 초록
```

긴 문서는 max source length에 맞게 truncate한다.

EDA에서 다음을 먼저 확인한다.

- 256 토큰 안에 들어오는 비율
- 384 토큰 안에 들어오는 비율
- 512 토큰 안에 들어오는 비율
- truncation된 문서의 정답 keyphrase가 잘리는 구간에 있는지

기본값 후보:

```text
max_source_length = 384 또는 512
max_target_length = 64 또는 96
```

---

## 7.2 Step 2. Extractive candidate generation

KeyBERT 또는 supervised extractor로 후보를 만든다.

```text
C_ext = {c1, c2, ..., cN}
```

후보 수 권장:

```text
N_ext = 20~50
```

특징:

- present 후보 정밀도 확보
- title에 등장하는 전문 용어 확보
- 생성 모델이 놓치는 표면 표현 보완

---

## 7.3 Step 3. Generative candidate generation

KeyBART를 이용해 후보를 만든다.

```text
C_gen = {g1, g2, ..., gM}
```

후보 확장 방식:

- greedy 1개 시퀀스
- beam search 여러 시퀀스
- top-p sampling 여러 회
- diverse beam search

각 시퀀스를 `<kp_sep>`로 나누고 모두 합친다.

```text
C = C_ext ∪ C_gen
```

후보 수가 너무 많으면 30~100개 안에서 제한한다.

---

## 7.4 Step 4. 후보 정규화

다음 순서로 처리한다.

1. 앞뒤 공백 제거
2. 소문자 변환
3. 특수 토큰 제거
4. 빈 문자열 제거
5. 지나치게 긴 phrase 제거
6. 문자열 중복 제거
7. stem 중복 제거
8. 입력 전체를 그대로 복사한 이상 출력 제거
9. `<unk>` 포함 출력 제거

권장 길이 필터:

```text
1~5 words를 기본으로 사용
6 words 이상은 삭제가 아니라 별도 실험
```

긴 phrase가 적다고 무조건 제거하면 정답 recall이 떨어질 수 있으므로 validation에서 결정한다.

---

## 7.5 Step 5. 후보 관련도 점수

초기에는 학습 없는 점수 결합을 사용한다.

$$
S(c)=w_g S_{gen}(c)+w_e S_{ext}(c)+w_s S_{sem}(c)+w_t S_{title}(c)
$$

각 항은 다음과 같다.

- $S_{gen}$: 길이 정규화된 생성 log-probability
- $S_{ext}$: KeyBERT 또는 추출 모델 점수
- $S_{sem}$: 문서와 candidate의 SBERT cosine similarity
- $S_{title}$: title에 등장하거나 title과 유사할 때 주는 보조 점수

모든 점수는 0~1 범위로 min-max 또는 rank normalization한 후 합친다.

초기 가중치 예시:

```text
w_gen = 0.35
w_ext = 0.20
w_sem = 0.35
w_title = 0.10
```

이 값은 정답이 아니며 validation에서 탐색한다.

---

## 7.6 Step 6. Learned Reranker

점수 결합 다음 단계로 Cross-Encoder reranker를 추가한다.

입력:

```text
[문서 title+abstract] [SEP] [candidate keyphrase]
```

출력:

```text
candidate가 해당 문서의 정답 keyphrase일 확률
```

### 라벨 구성

- Positive: 정답 keyphrase
- Hard negative: extractor/generator가 만들었지만 정답과 exact match하지 않는 후보
- Random negative: 다른 문서의 keyphrase

### 중요한 해석

이 reranker도 중요도 등급을 배우는 것이 아니라 **관련/비관련**을 배운다.

모델 출력 확률을 최종 순위 점수로 사용하는 것이다.

### 데이터 누수 방지

generator가 train 문서를 그대로 외운 후보로 reranker를 학습시키지 않도록 다음 방법을 권장한다.

```text
Train split의 일부를 generator 학습용과 reranker 후보 생성용으로 나눔
```

예:

```text
train_generator: 95%
train_reranker: 5%
```

시간이 부족하면 learned reranker는 확장 실험으로 두고 점수 결합만 구현한다.

---

## 7.7 Step 7. MMR 중복 제거

관련도만 높은 순서로 뽑으면 비슷한 표현이 반복될 수 있다.

MMR은 다음 기준으로 다음 후보를 선택한다.

$$
\operatorname{MMR}(c)=\lambda \operatorname{Rel}(c,x)
-(1-\lambda)\max_{s\in S}\operatorname{Sim}(c,s)
$$

- 첫 번째 항: 문서와 관련성이 높은가?
- 두 번째 항: 이미 선택된 keyphrase와 너무 비슷한가?

권장 탐색 범위:

```text
lambda ∈ {0.5, 0.6, 0.7, 0.8, 0.9}
```

- lambda가 높음: 관련도 우선
- lambda가 낮음: 다양성 우선

기본 시작값:

```text
lambda = 0.7
```

---

## 7.8 Step 8. 출력 개수 결정

평가용과 데모용을 분리한다.

### 평가용

```text
Top-5
Top-10
Top-M: M은 해당 문서의 gold keyphrase 수
```

### 실제 데모용

다음 중 하나를 사용한다.

- 고정 5개
- 관련도 threshold 이상만 출력
- 최대 10개, 최소 3개
- 모델의 `<eos>` 또는 `<none>` 기반 variable number

발표 데모에서는 평균 정답 수가 약 5개이므로 기본 5개가 이해하기 쉽다.

---

# 8. 평가 설계

## 8.1 평가 전 정규화

정답과 예측을 그대로 문자열 비교하면 다음을 모두 다르게 본다.

```text
Neural Networks
neural network
neural-network
```

기본 exact-match 평가는 다음을 적용한다.

1. lowercase
2. 공백 정리
3. punctuation 정규화
4. tokenizer 적용
5. Porter stemming
6. 중복 제거

주의: 하이픈 처리 규칙을 실험 전체에서 고정한다.

---

## 8.2 F1@K

Top-K 예측을 정답 집합과 비교한다.

$$
P@K = \frac{TP@K}{K}
$$

$$
R@K = \frac{TP@K}{|Y|}
$$

$$
F1@K = \frac{2P@K R@K}{P@K + R@K}
$$

예측이 K개보다 적으면 부족한 개수는 오답으로 간주해 $K$를 분모로 유지한다. 이렇게 해야 적게 출력해서 precision만 높아지는 현상을 줄일 수 있다.

주요 지표:

- F1@5
- F1@10
- F1@M

여기서 M은 문서별 gold keyphrase 개수다.

---

## 8.3 Present / Absent 분리 평가

### Gold 분리

```text
Present gold = prmu == P
Absent gold = prmu in {R, M, U}
```

### 예측 분리

예측 phrase가 title+abstract에 연속적으로 나타나는지 동일한 stemming 규칙으로 판정한다.

보고할 항목:

| 구분 | 추천 지표 |
|---|---|
| Present | P@5, R@5, F1@5, F1@M |
| Absent | R@5, R@10, R@50, F1@5, F1@M |
| 전체 | F1@5, F1@10, F1@M |

Absent는 맞히는 개수가 매우 적을 수 있으므로 Recall도 반드시 함께 본다.

---

## 8.4 PRMU 세부 평가

정답을 P/R/M/U로 나눠 각 유형의 recall을 계산한다.

```text
Recall_P
Recall_R
Recall_M
Recall_U
```

이 분석으로 다음을 알 수 있다.

- Extractor는 P에 강하고 U에 약함
- Generator가 M/U를 얼마나 개선하는지
- 모델이 단순 재배열 R은 잘하는지
- 완전한 개념 생성 U는 얼마나 어려운지

---

## 8.5 Ranking 평가

KP20k에는 graded importance label이 없으므로 ranking 평가를 과장해서 해석하면 안 된다.

사용 가능한 지표:

### Precision@K / Recall@K

정답 phrase가 앞쪽에 많이 위치하는지 본다.

### MAP@K

정답이 등장하는 순위를 누적해 평가한다.

### nDCG@K

모든 gold phrase를 relevance=1로 두는 binary nDCG는 사용할 수 있다.

다만 모든 gold가 같은 중요도를 갖는다고 가정한 지표다.

### 실제 중요도 랭킹을 평가하려면

- 사람에게 1~5점 중요도 부여
- pairwise preference annotation
- 검색 성능 개선량을 relevance grade로 사용

이번 프로젝트의 기본 ranking 주장은 다음 범위로 제한한다.

> 정답 keyphrase가 상위 K개 안에 더 자주 배치되는가?

---

## 8.6 Diversity 평가

### Exact duplicate ratio

$$
1-\frac{|\operatorname{unique}(\hat{Y})|}{|\hat{Y}|}
$$

### Stem duplicate ratio

stemming 후 같은 표현이 반복되는 비율이다.

### Semantic redundancy

모든 예측 쌍의 cosine similarity 평균 또는 최대값을 계산한다.

$$
\operatorname{Redundancy}=\frac{1}{N(N-1)}\sum_{i\neq j}\operatorname{sim}(y_i,y_j)
$$

낮을수록 다양하다.

단, 지나치게 낮으면 서로 관련 없는 표현을 뽑았을 수 있으므로 relevance와 함께 본다.

---

## 8.7 의미 기반 평가

Exact match는 동의어를 오답 처리한다.

예:

```text
gold: neural network
prediction: artificial neural networks
```

이를 보완하기 위해 KPEval의 관점을 적용한다.

1. **Reference agreement:** 정답과 의미가 일치하는가?
2. **Faithfulness:** 문서 내용과 모순되거나 뜬금없는 환각이 아닌가?
3. **Diversity:** 출력이 중복되지 않는가?
4. **Utility:** 검색이나 분류에 실제 도움이 되는가?

프로젝트 최소 구현:

- exact F1
- embedding-based semantic F1
- duplicate ratio
- document-keyphrase semantic similarity

확장 구현:

- KPEval toolkit 적용
- keyphrase를 query로 사용한 retrieval Recall@K/MRR

KPEval:

- https://aclanthology.org/2024.findings-acl.117/
- https://github.com/uclanlp/KPEval

---

## 8.8 Macro와 Micro

기본 보고는 **문서별 지표를 평균내는 macro average**를 권장한다.

```text
각 논문마다 F1 계산
→ 전체 논문의 평균
```

Micro는 전체 정답과 예측을 한데 모으므로 키프레이즈가 많은 문서가 더 큰 영향을 준다.

최종 표에는 다음처럼 기록한다.

```text
Macro F1@5
Macro F1@M
Micro F1@5 (보조)
```

---

# 9. 평가 코드 골격

아래 코드는 최종 구현 전에 구조를 이해하기 위한 뼈대다.

```python
import re
from collections import OrderedDict
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def normalize_phrase(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # 실제 최종본에서는 공식 tokenization/hyphen 규칙과 맞출 것
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    stems = [stemmer.stem(tok) for tok in tokens]
    return " ".join(stems)


def unique_normalized(phrases):
    seen = OrderedDict()
    for phrase in phrases:
        norm = normalize_phrase(phrase)
        if norm and norm not in seen:
            seen[norm] = phrase.strip()
    return list(seen.keys())


def f1_at_k(pred_phrases, gold_phrases, k: int):
    pred = unique_normalized(pred_phrases)[:k]
    gold = set(unique_normalized(gold_phrases))

    tp = len(set(pred) & gold)

    # K개보다 적게 출력하면 부족한 수는 오답으로 취급
    precision = tp / k if k > 0 else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp}
```

반드시 작성할 단위 테스트:

```python
def test_perfect_match():
    result = f1_at_k(["neural network"], ["neural networks"], 1)
    assert result["f1"] == 1.0


def test_empty_prediction():
    result = f1_at_k([], ["neural network"], 5)
    assert result["f1"] == 0.0


def test_duplicate_prediction():
    result = f1_at_k(
        ["neural network", "neural networks", "classification"],
        ["neural network", "classification"],
        2,
    )
    assert result["tp"] == 2
```

---

# 10. 주피터 노트북 전체 설계

한 개의 거대한 노트북보다 역할별로 분리하는 것을 권장한다.

```text
keyphrase_project/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ baseline.yaml
│  ├─ bart.yaml
│  └─ keybart.yaml
├─ notebooks/
│  ├─ 00_project_overview.ipynb
│  ├─ 01_data_loading_and_audit.ipynb
│  ├─ 02_eda_prmu.ipynb
│  ├─ 03_evaluation_metrics.ipynb
│  ├─ 04_tfidf_baseline.ipynb
│  ├─ 05_keybert_baseline.ipynb
│  ├─ 06_bart_t5_one2seq.ipynb
│  ├─ 07_keybart_finetuning.ipynb
│  ├─ 08_decoding_candidate_pool.ipynb
│  ├─ 09_reranking.ipynb
│  ├─ 10_mmr_diversity.ipynb
│  ├─ 11_hybrid_pipeline.ipynb
│  ├─ 12_experiment_table.ipynb
│  ├─ 13_error_analysis.ipynb
│  └─ 14_demo.ipynb
├─ src/
│  ├─ data.py
│  ├─ preprocessing.py
│  ├─ metrics.py
│  ├─ generation.py
│  ├─ reranking.py
│  ├─ diversity.py
│  └─ utils.py
├─ outputs/
│  ├─ predictions/
│  ├─ metrics/
│  ├─ figures/
│  └─ checkpoints/
└─ tests/
   └─ test_metrics.py
```

---

## Notebook 00. Project Overview

### 목적

팀원 누구나 노트북 하나만 열어도 프로젝트 전체를 이해한다.

### Markdown 셀

- 문제 정의
- 입력/출력 예시
- P/R/M/U 설명
- 전체 파이프라인 그림
- 실험 모델 목록
- 평가 지표 목록

### Code 셀

- random seed
- device 확인
- library version 출력
- config 로드

### 산출물

```text
프로젝트 지도 역할을 하는 노트북
```

---

## Notebook 01. Data Loading and Audit

### 목표

데이터를 안정적으로 불러오고 필드가 예상과 같은지 확인한다.

```python
from datasets import load_dataset

dataset = load_dataset("taln-ls2n/kp20k")
print(dataset)
print(dataset["train"][0])
```

확인 항목:

- split 크기
- null title/abstract
- 빈 keyphrases
- `len(keyphrases) == len(prmu)` 여부
- 중복 id
- 동일 title 중복
- exact duplicate document
- keyphrase 내부 빈 문자열
- 비영어 문자 비율

### 산출물

```text
outputs/metrics/data_audit.json
```

---

## Notebook 02. EDA and PRMU

### 필수 그래프

1. 문서당 keyphrase 개수 분포
2. keyphrase word length 분포
3. title/abstract token length 분포
4. P/R/M/U 비율
5. 문서별 P 비율 분포
6. title에만 등장하는 P 비율
7. 1-word / 2-word / 3-word 이상 비율
8. 중복 keyphrase 비율
9. 분야별 단어 빈도 또는 상위 키프레이즈
10. target token length 분포

### 반드시 비교할 통계

```text
공식 prmu 통계
vs
팀 custom PRMU 함수 통계
```

### EDA에서 결정할 하이퍼파라미터

- max source length
- max target length
- keyphrase 최대 단어 수
- 평가 K값
- baseline n-gram range

---

## Notebook 03. Evaluation Metrics

### 목표

모델 학습 전에 평가 함수를 완성한다.

### 구현 함수

```text
normalize_phrase
stem_phrase
deduplicate_phrases
split_present_absent
classify_prediction_prmu
precision_recall_f1_at_k
f1_at_m
map_at_k
ndcg_at_k
semantic_f1
duplicate_ratio
semantic_redundancy
```

### 단위 테스트

- 완전 정답
- 완전 오답
- 대소문자 차이
- 복수형 차이
- 중복 예측
- 예측 개수 부족
- gold가 비어 있는 예외
- present/absent 분리
- 하이픈 포함 표현

### 가장 중요한 원칙

모든 모델이 동일한 evaluator를 사용해야 한다.

---

## Notebook 04. TF-IDF Baseline

### 단계

1. title+abstract 결합
2. `TfidfVectorizer(ngram_range=(1, 3))`
3. 불용어 제거
4. 각 문서 Top-20 후보 생성
5. Top-5, Top-10 평가
6. P/non-P 분리 평가

### 코드 골격

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

train_texts = [
    f"{row['title']} {row['abstract']}"
    for row in dataset["train"]
]

X_train = vectorizer.fit_transform(train_texts)
```

전체 53만 문서에 3gram TF-IDF를 한 번에 적용하면 메모리가 많이 들 수 있으므로 다음 중 하나를 사용한다.

- 일부 train subset으로 vocabulary 구축
- `max_features` 제한
- HashingVectorizer
- 문서별 candidate frequency + corpus DF 사전 계산

---

## Notebook 05. KeyBERT Baseline

### 단계

1. 소규모 test subset 100~1,000건으로 작동 확인
2. 후보 n-gram 1~3 설정
3. embedding model 선택
4. MMR off/on 비교
5. 전체 test 또는 샘플 test 평가

### 코드 골격

```python
from keybert import KeyBERT

kw_model = KeyBERT(model="sentence-transformers/all-MiniLM-L6-v2")

text = f"{sample['title']}. {sample['abstract']}"
keywords = kw_model.extract_keywords(
    text,
    keyphrase_ngram_range=(1, 3),
    stop_words="english",
    top_n=10,
    use_mmr=True,
    diversity=0.5,
)
```

### 주의

20,000개 test 전체에 KeyBERT를 실행하면 시간이 길 수 있으므로 embedding batch 처리와 캐시를 사용한다.

---

## Notebook 06. BART/T5 One2Seq Baseline

### 데이터 변환

```python
KP_SEP = "<kp_sep>"
P_TOKEN = "<present>"
A_TOKEN = "<absent>"


def build_example(row):
    present = [
        kp for kp, tag in zip(row["keyphrases"], row["prmu"])
        if tag == "P"
    ]
    absent = [
        kp for kp, tag in zip(row["keyphrases"], row["prmu"])
        if tag != "P"
    ]

    source = f"<title> {row['title']} <abstract> {row['abstract']}"
    target = (
        f"{P_TOKEN} "
        + f" {KP_SEP} ".join(present)
        + f" {A_TOKEN} "
        + f" {KP_SEP} ".join(absent)
    )
    return {"source": source, "target": target}
```

### special token 등록

```python
special_tokens = {
    "additional_special_tokens": [
        "<title>", "<abstract>", "<present>", "<absent>", "<kp_sep>"
    ]
}

tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
```

### 먼저 작은 데이터로 검증

```text
1,000 train / 200 validation
→ overfit 가능한지 확인
→ 10,000 train
→ full train
```

바로 53만 개 전체를 학습하면 전처리나 평가 버그를 늦게 발견한다.

### 학습 모니터링

- train loss
- validation loss
- generated phrase count
- duplicate ratio
- F1@5
- P F1@5
- absent Recall@10

loss만 보고 best checkpoint를 고르지 않는다.

---

## Notebook 07. KeyBART Fine-tuning

### 모델 로드

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "bloomberg/KeyBART"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

### 비교를 공정하게 하기 위한 조건

BART와 KeyBART에 다음을 동일하게 적용한다.

- 같은 train subset
- 같은 max length
- 같은 target format
- 같은 epoch 또는 update step
- 같은 batch size와 gradient accumulation
- 같은 decoding 설정
- 같은 evaluator

---

## Notebook 08. Decoding and Candidate Pool

### 비교할 decoding

| 설정 | 설명 | 기대 효과 |
|---|---|---|
| Greedy | 매번 최고 확률 토큰 | 높은 precision, 낮은 다양성 가능 |
| Beam 5 | 상위 시퀀스 탐색 | 안정적 baseline |
| Beam 10 | 후보 증가 | recall 개선 가능, 중복 증가 가능 |
| Diverse Beam | beam 간 차이 유도 | 다양성 개선 |
| Top-p Sampling | 확률적 여러 출력 | 후보 recall 개선, 노이즈 증가 |
| DeSel형 | 많이 생성 후 likelihood로 선택 | precision-recall 균형 |

### 저장 형식

```json
{
  "id": "...",
  "raw_sequences": ["...", "..."],
  "phrases": [
    {"text": "neural network", "gen_score": -1.23, "beam_id": 0}
  ]
}
```

후보와 점수를 파일로 저장하면 모델을 다시 돌리지 않고 reranking 실험을 반복할 수 있다.

---

## Notebook 09. Reranking

### 1차: 학습 없는 점수 결합

- gen score
- KeyBERT score
- SBERT document similarity
- title overlap

validation에서 grid search한다.

### 2차: Cross-Encoder

모델 후보:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
allenai/scibert_scivocab_uncased 기반 분류기
```

과학 논문 도메인이므로 SciBERT 기반 reranker를 실험할 가치가 있다.

### 평가

- rerank 전후 F1@5
- MAP@10
- binary nDCG@10
- P/absent 각각의 변화

---

## Notebook 10. MMR and Diversity

### 실험

```text
MMR 없음
lambda=0.5
lambda=0.6
lambda=0.7
lambda=0.8
lambda=0.9
```

### 측정

- F1@5
- semantic F1@5
- duplicate ratio
- semantic redundancy
- 평균 출력 개수

### 해석

MMR의 성공은 단순히 F1 상승만이 아니다.

```text
F1이 거의 유지되면서 redundancy가 크게 감소
```

하면 실제 사용자 관점에서 개선으로 볼 수 있다.

---

## Notebook 11. Full Hybrid Pipeline

최종 함수 형태:

```python
def predict_keyphrases(
    title: str,
    abstract: str,
    top_k: int = 5,
    n_extract: int = 30,
    n_generate: int = 30,
    mmr_lambda: float = 0.7,
):
    text = build_source(title, abstract)

    ext_candidates = extract_candidates(text, top_n=n_extract)
    gen_candidates = generate_candidates(text, top_n=n_generate)

    candidates = merge_and_normalize(ext_candidates, gen_candidates)
    scored = score_or_rerank(text, candidates)
    selected = mmr_select(scored, top_k=top_k, lambda_=mmr_lambda)

    return selected
```

출력 예시:

```python
[
    {
        "phrase": "neural network",
        "score": 0.91,
        "source": ["extractive", "generative"],
        "type": "present",
    },
    {
        "phrase": "medical image classification",
        "score": 0.84,
        "source": ["generative"],
        "type": "absent",
    },
]
```

이 형태는 발표 시 설명 가능성이 높다.

---

## Notebook 12. Experiment Table

모든 실험을 하나의 DataFrame으로 관리한다.

| run_id | model | input | decoder | reranker | MMR | F1@5 | F1@M | P-F1@5 | A-R@10 | SemF1 | DupRatio |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| B0 | TF-IDF | T+A | - | - | No | | | | | | |
| B1 | KeyBERT | T+A | - | - | No | | | | | | |
| B2 | KeyBERT | T+A | - | - | Yes | | | | | | |
| B3 | BART | T+A | Beam5 | - | No | | | | | | |
| B4 | KeyBART | T+A | Beam5 | - | No | | | | | | |
| P1 | Hybrid | T+A | Beam10 | ScoreFusion | No | | | | | | |
| P2 | Hybrid | T+A | Beam10 | ScoreFusion | Yes | | | | | | |
| P3 | Hybrid | T+A | Sampling | CrossEncoder | Yes | | | | | | |

---

## Notebook 13. Error Analysis

최소 50~100개 실패 사례를 다음 유형으로 분류한다.

| 오류 유형 | 설명 |
|---|---|
| Missing present | 원문에 있는데 못 찾음 |
| Missing absent | 추상 개념을 못 만듦 |
| Hallucination | 문서와 관련 없는 표현 생성 |
| Over-general | `machine learning`처럼 너무 일반적 |
| Over-specific | 지나치게 세부적인 표현 |
| Partial match | 정답 일부만 맞음 |
| Synonym mismatch | 의미는 맞으나 exact match 실패 |
| Morphological mismatch | 단수/복수·파생형 차이 |
| Duplicate | 의미가 같은 표현 반복 |
| Wrong ranking | 맞는 표현이 있으나 뒤쪽에 위치 |
| Truncation | 입력 잘림 때문에 정보 누락 |
| Domain term | 희귀 전문용어 처리 실패 |

### 반드시 볼 비교 사례

```text
KeyBERT는 맞히고 KeyBART는 틀린 사례
KeyBART는 맞히고 KeyBERT는 틀린 사례
Hybrid가 둘 모두보다 좋아진 사례
MMR 때문에 정답이 제거된 사례
Exact F1은 실패하지만 semantic F1은 성공한 사례
```

---

## Notebook 14. Demo

입력창:

```text
Title
Abstract
Top-K
```

출력:

| Rank | Keyphrase | Present/Absent | Score | Source |
|---:|---|---|---:|---|
| 1 | deep learning | Present | 0.94 | Extract+Generate |
| 2 | image classification | Present | 0.90 | Extract |
| 3 | computer-aided diagnosis | Absent | 0.82 | Generate |

추가 시각화:

- 원문에서 present phrase 하이라이트
- absent phrase는 별도 색 또는 아이콘
- 중복 제거 전후 비교
- 모델별 결과 비교

---

# 11. 실험 설계

## 11.1 핵심 비교 실험

### E1. Extractive baseline

```text
TF-IDF vs KeyBERT vs KeyBERT+MMR
```

목적:

- 추출만으로 가능한 최대 범위 확인
- MMR의 다양성 효과 확인

---

### E2. Generative baseline

```text
BART vs KeyBART
```

목적:

- 일반 PLM과 task-specific PLM 비교

---

### E3. Hybrid

```text
KeyBERT
KeyBART
KeyBERT + KeyBART 후보 통합
Hybrid + Reranker
Hybrid + Reranker + MMR
```

목적:

- 각 구성 요소의 기여 분리

---

### E4. Input ablation

```text
Abstract only
Title only
Title + Abstract
```

예상:

- title+abstract가 전체적으로 가장 좋음
- title only는 precision은 높을 수 있으나 recall 낮음
- abstract only는 title에만 있는 P를 놓칠 수 있음

---

### E5. Target format ablation

```text
kp1 <sep> kp2 <sep> kp3
<present> ... <absent> ...
P/R/M/U 별 special token
phrase order fixed
phrase order shuffled
```

목적:

- present/absent 경계 토큰의 효과
- 순서 편향의 크기

---

### E6. Decoding ablation

```text
Greedy
Beam5
Beam10
Diverse Beam
Top-p sampling
```

평가:

- precision
- recall
- unique phrase count
- duplicate ratio
- inference time

---

### E7. MMR ablation

```text
No MMR
MMR lambda grid
```

---

### E8. 데이터 크기 실험

```text
1%
10%
100%
```

목적:

- KeyBART의 data efficiency 확인
- 일반 BART와 차이가 저자원 조건에서 더 큰지 확인

---

## 11.2 Candidate Recall 실험

Hybrid에서 매우 중요한 분석이다.

Reranker가 아무리 좋아도 후보 pool에 정답이 없으면 선택할 수 없다.

따라서 selector 성능 전에 다음을 계산한다.

```text
Candidate Recall@20
Candidate Recall@50
Candidate Recall@100
```

모델별로 계산한다.

```text
KeyBERT candidate recall
KeyBART candidate recall
Union candidate recall
```

이 결과는 다음을 구분해 준다.

- 정답 후보를 애초에 만들지 못한 generator 문제
- 후보는 있지만 앞에 올리지 못한 ranker 문제

---

## 11.3 3개 Seed

최종 핵심 실험은 seed 3개 평균과 표준편차를 보고한다.

```text
seed = 42, 43, 44
```

모든 실험을 3회 수행하기 어렵다면 다음만 3회 한다.

- BART baseline
- KeyBART baseline
- 최종 proposed model

---

# 12. 추천 우선순위

## 반드시 구현

1. 데이터 audit + EDA
2. exact-match evaluator
3. TF-IDF 또는 KeyBERT
4. BART 또는 T5 One2Seq
5. KeyBART fine-tuning
6. 생성 결과 parsing/정규화
7. present/absent 분리 평가
8. MMR
9. 비교표
10. 오류 분석

## 시간 있으면 구현

1. Cross-Encoder reranker
2. decoding 비교
3. semantic F1 또는 KPEval
4. candidate recall 분석
5. phrase order shuffle
6. title ablation

## 연구 확장

1. One2Set / SetTrans
2. ExHiRD exclusion mechanism
3. DeSel decoding
4. unlikelihood training
5. retrieval utility evaluation
6. human importance ranking annotation
7. One2Set + selector 구조

---

# 13. GPU와 학습 전략

16GB GPU 기준 현실적인 시작값이다.

## BART-base / KeyBART

```text
max_source_length: 384
max_target_length: 96
per_device_train_batch_size: 4~8
gradient_accumulation_steps: 4~8
effective batch size: 32 정도
learning_rate: 3e-5 ~ 6e-5
weight_decay: 0.01
epochs: 2~5
fp16 또는 bf16
predict_with_generate: True
gradient_checkpointing: 필요 시 사용
```

## 데이터 단계적 확장

```text
Stage A: 1,000개로 overfit test
Stage B: 10,000개로 pipeline 검증
Stage C: 50,000개로 모델 비교
Stage D: full 530,809개 최종 학습
```

처음부터 full train을 돌리지 않는다.

---

# 14. 실험 추적

최소한 CSV 또는 JSONL로 기록한다.

```json
{
  "run_id": "keybart_beam5_seed42",
  "model": "bloomberg/KeyBART",
  "seed": 42,
  "max_source_length": 384,
  "max_target_length": 96,
  "beam_size": 5,
  "f1_at_5": 0.0,
  "present_f1_at_5": 0.0,
  "absent_recall_at_10": 0.0,
  "duplicate_ratio": 0.0
}
```

가능하면 Weights & Biases 또는 MLflow를 사용한다.

하지만 프로젝트 시간이 짧다면 pandas DataFrame + CSV로 충분하다.

---

# 15. 일주일 실행 계획

## Day 1: 문제와 데이터 고정

- 데이터 로드
- 공식 split/field 검증
- EDA
- PRMU 통계 재현
- 팀 통계와 차이 분석
- max input/target length 결정

산출물:

```text
01_data_loading_and_audit.ipynb
02_eda_prmu.ipynb
```

---

## Day 2: 평가 함수와 추출 baseline

- normalize/stemming
- F1@5/F1@10/F1@M
- P/absent 분리
- 단위 테스트
- TF-IDF 또는 KeyBERT

산출물:

```text
03_evaluation_metrics.ipynb
04_tfidf_baseline.ipynb
05_keybert_baseline.ipynb
```

---

## Day 3: 생성 baseline

- target formatter
- BART/T5 small subset overfit
- generation parsing
- checkpoint 및 prediction 저장

산출물:

```text
06_bart_t5_one2seq.ipynb
```

---

## Day 4: KeyBART

- KeyBART fine-tuning
- BART와 동일 조건 비교
- decoding 기본값 확정

산출물:

```text
07_keybart_finetuning.ipynb
08_decoding_candidate_pool.ipynb
```

---

## Day 5: Hybrid + MMR

- KeyBERT와 KeyBART 후보 통합
- 점수 정규화
- score fusion
- MMR lambda 탐색

산출물:

```text
09_reranking.ipynb
10_mmr_diversity.ipynb
11_hybrid_pipeline.ipynb
```

---

## Day 6: 비교 실험과 오류 분석

- 실험표 작성
- ablation 2~4개
- candidate recall
- 실패 사례 분류
- 정성 예시 선정

산출물:

```text
12_experiment_table.ipynb
13_error_analysis.ipynb
```

---

## Day 7: 보고서와 데모

- 최종 결과표
- 파이프라인 그림
- 성공/실패 사례
- limitations
- demo 정리

산출물:

```text
14_demo.ipynb
최종 연구계획서
발표용 결과표와 그림
```

---

# 16. 최종 보고서 구성

## 1. Problem Definition

- 핵심 개념 압축 문제
- vocabulary mismatch
- present와 absent의 동시 처리
- 랭킹과 중복 제거 필요성

## 2. Dataset

- KP20k 구성
- P/R/M/U
- EDA
- domain limitation

## 3. Baselines

- TF-IDF/KeyBERT
- BART/T5
- KeyBART

## 4. Proposed Method

- extractive + generative candidate pool
- reranking
- MMR

## 5. Evaluation

- exact F1
- present/absent
- PRMU
- semantic evaluation
- diversity
- ranking caveat

## 6. Experiments

- baseline comparison
- ablation
- decoding
- candidate recall

## 7. Error Analysis

- 실패 유형
- 모델별 장단점

## 8. Conclusion

- 무엇이 실제로 개선됐는지
- absent 생성과 hallucination trade-off
- ranking label 부재
- 향후 연구

---

# 17. 발표에서 반드시 설명할 핵심 문장

> 이 프로젝트는 단순 추출이 아니라 추출과 생성을 동시에 수행하는 keyphrase prediction 문제다.

> KP20k의 약 58%는 원문에 연속 등장하는 P이고, 나머지는 재배열·혼합·미등장 표현이므로 추출형 모델만으로 전체 문제를 해결할 수 없다.

> 다만 KP20k의 정답 순서는 중요도 순위가 아니므로, 생성 시퀀스 순서를 그대로 중요도라고 해석하지 않고 별도의 관련도 기반 reranking을 수행한다.

> 최종 모델은 KeyBERT의 present precision과 KeyBART의 absent recall을 후보 통합으로 결합하고, reranker와 MMR을 이용해 관련성과 다양성의 균형을 맞춘다.

> 평가는 exact-match F1뿐 아니라 present/absent, PRMU, semantic agreement, faithfulness, diversity를 함께 본다.

---

# 18. 주요 논문과 읽는 순서

## 1순위: Task 원형

### Deep Keyphrase Generation — Meng et al., ACL 2017

- https://aclanthology.org/P17-1054/

읽을 포인트:

- CopyRNN
- present/absent 정의
- Seq2Seq 기반 keyphrase generation
- KP20k의 기원

---

## 2순위: 출력 개수와 One2Seq

### One Size Does Not Fit All: Generating and Evaluating Variable Number of Keyphrases — Yuan et al., ACL 2020

- https://aclanthology.org/2020.acl-main.710/

읽을 포인트:

- delimiter-separated generation
- variable number generation
- F1@5의 한계
- 다양성 제어

---

## 3순위: 중복 억제

### Exclusive Hierarchical Decoding for Deep Keyphrase Generation — Chen et al., ACL 2020

- https://aclanthology.org/2020.acl-main.103/

읽을 포인트:

- hierarchical decoding
- soft/hard exclusion
- 모델 내부에서 중복을 줄이는 방법

---

## 4순위: PRMU와 retrieval 관점

### Redefining Absent Keyphrases and their Effect on Retrieval Effectiveness — Boudin & Gallina, NAACL 2021

- https://aclanthology.org/2021.naacl-main.330/

읽을 포인트:

- P/R/M/U
- absent를 하나의 범주로만 볼 때의 문제
- 검색 성능과 unseen word의 관계

---

## 5순위: 순서 편향 해결

### One2Set: Generating Diverse Keyphrases as a Set — Ye et al., ACL 2021

- https://aclanthology.org/2021.acl-long.354/

읽을 포인트:

- keyphrase는 sequence가 아니라 set이라는 문제
- bipartite matching
- parallel keyphrase generation
- duplication 감소

---

## 6순위: 실용 모델

### Learning Rich Representation of Keyphrases from Text / KeyBART — Kulkarni et al., NAACL Findings 2022

- https://arxiv.org/abs/2112.08547
- https://huggingface.co/bloomberg/KeyBART

읽을 포인트:

- KeyBART 사전학습 목적
- CatSeq 형식
- 일반 BART 대비 이점

---

## 7순위: 모델과 decoding 선택

### Rethinking Model Selection and Decoding for Keyphrase Generation with Pre-trained Seq2Seq Models — Wu et al., EMNLP 2023

- https://aclanthology.org/2023.emnlp-main.410/

읽을 포인트:

- greedy, beam, sampling 차이
- precision-recall trade-off
- DeSel decode-select
- 큰 모델이 항상 좋은 것은 아님

---

## 8순위: 평가 개선

### KPEval: Towards Fine-Grained Semantic-Based Keyphrase Evaluation — Wu et al., ACL Findings 2024

- https://aclanthology.org/2024.findings-acl.117/
- https://github.com/uclanlp/KPEval

읽을 포인트:

- exact match의 한계
- reference agreement
- faithfulness
- diversity
- utility

---

## 9순위: Generate-then-Select 최신 방향

### One2Set + Large Language Model: Best Partners for Keyphrase Generation — Shao et al., EMNLP 2024

- https://aclanthology.org/2024.emnlp-main.624/

읽을 포인트:

- 한 모델이 recall과 precision 모두 최고가 되기 어렵다는 관찰
- generator + selector 분해
- 이번 프로젝트의 candidate generation + reranking 구조와 연결

---

## 10순위: 최근 전체 분석

### An Analysis of Datasets, Metrics and Models in Keyphrase Generation — Boudin & Aizawa, 2025

- https://arxiv.org/abs/2506.10346

읽을 포인트:

- 데이터셋과 평가 관행의 문제
- 모델 성능 비교 시 주의점
- 향후 연구 방향

---

# 19. 최종 의사결정

## 프로젝트의 최소 성공 버전

```text
KeyBERT
vs
BART 또는 KeyBART
vs
KeyBART + MMR
```

이 구성만으로도 다음을 설명할 수 있다.

- extraction과 generation의 차이
- absent keyphrase의 필요성
- duplicate 제거의 필요성

## 프로젝트의 권장 완성 버전

```text
TF-IDF 또는 KeyBERT baseline
BART baseline
KeyBART strong baseline
KeyBERT + KeyBART candidate union
Score fusion 또는 Cross-Encoder reranker
MMR diversification
Exact + Semantic + PRMU evaluation
```

## 프로젝트의 연구형 확장 버전

```text
One2Set/SetTrans generator
+ learned selector
+ semantic/diversity objective
+ retrieval utility evaluation
```

---

# 20. 최종 체크리스트

## Task 이해

- [ ] extraction과 generation을 구분해 설명할 수 있다.
- [ ] P/R/M/U를 예시로 설명할 수 있다.
- [ ] 키프레이즈 순서가 중요도 gold rank가 아니라는 것을 이해했다.
- [ ] optimizer가 자동으로 중복을 제거하는 것이 아님을 이해했다.

## 데이터

- [ ] title+abstract를 사용한다.
- [ ] 공식 split 크기를 검증했다.
- [ ] `keyphrases`와 `prmu` 길이를 확인했다.
- [ ] 공식 PRMU 통계와 팀 통계의 차이를 설명했다.
- [ ] truncation 비율을 확인했다.

## 평가

- [ ] stemming exact-match evaluator가 있다.
- [ ] F1@5, F1@10, F1@M이 있다.
- [ ] present와 absent를 분리한다.
- [ ] P/R/M/U recall을 계산한다.
- [ ] duplicate ratio를 계산한다.
- [ ] semantic metric을 최소 하나 포함한다.
- [ ] evaluator 단위 테스트를 작성했다.

## 모델

- [ ] extractive baseline이 있다.
- [ ] 일반 Seq2Seq baseline이 있다.
- [ ] KeyBART 실험이 있다.
- [ ] 생성 결과 parsing을 검증했다.
- [ ] decoding 설정을 기록했다.

## Proposed Method

- [ ] 후보 통합을 구현했다.
- [ ] 점수 normalization을 했다.
- [ ] reranking 또는 score fusion을 했다.
- [ ] MMR을 적용했다.
- [ ] candidate recall을 측정했다.

## 분석

- [ ] 최소 50개 오류 사례를 분류했다.
- [ ] exact와 semantic 평가가 다른 사례를 찾았다.
- [ ] MMR이 정답을 제거한 실패 사례를 확인했다.
- [ ] absent hallucination 사례를 확인했다.
- [ ] 모델별 장단점을 정리했다.

---

# 21. 한 문장 최종 요약

> KP20k 프로젝트의 올바른 접근은 “키워드 하나를 뽑는 모델”을 만드는 것이 아니라, **추출형 후보와 생성형 후보를 함께 만들고, 관련도 기반으로 다시 정렬한 뒤, 중복을 제거하여 사용자에게 유용한 Top-K 핵심 개념을 제공하는 전체 파이프라인을 설계하는 것**이다.
