# RAG MVP — 진척 정리 & 개선 로드맵 (확정)

> 작성일: 2026-07-16 · 대상: `rag_mvp/` (P7 키프레이즈 + 벡터DB RAG 풀스택)
> 설계 배경: [`../../kp20k-keyphrase-extraction/docs/RAG_MVP_PLAN.md`](../) · 상태: **핵심 논지 검증 완료, 개선 단계 진입**

---

## 1. 지금까지 만든 것 (진척)

### 1-1. 풀스택 구현 (전부 작동 확인)

| 계층 | 구현 | 상태 |
|---|---|---|
| 프론트 | React SPA (CDN, 검색·카드·키프레이즈칩·LLM답변·코퍼스/모드 토글) | ✅ |
| 백엔드 | FastAPI `/api/{health,search,answer,ingest}` + 정적 서빙 | ✅ |
| 벡터DB | Chroma (arXiv `papers` 33,000 벡터 + KP20k `kp20k` 220,000 벡터) | ✅ |
| 키프레이즈 | **P7**(SciBERT+aux, KP20k 학습) + KeyBART + KeyBERT | ✅ |
| 임베더 | bge-small-en-v1.5 (검색), 문서+키프레이즈 혼합 랭킹 | ✅ |
| LLM 답변 | Qwen2.5-1.5B, 인용·근거·환각억제 | ✅ |

### 1-2. 코퍼스

- **arXiv cs.CL 3,000편** (최신 2026-06~07, 메타데이터 전처리) — 사용자용 "최신 논문 검색"
- **KP20k test 20,000편** (gold+PRMU 보유) — 논지 정량 평가용
- 수집: `fetch_arxiv.py` · 색인: `ingest.py --source {arxiv,kp20k}`

### 1-3. 검증된 동작

- **검색**: "on-device mobile agent" → PalmClaw 논문 최상위, **absent 키프레이즈 "device agent[A]"로 매칭** (어휘 불일치 해결 실증)
- **LLM 답변**: 4편 인용 종합 답변, 근거 없으면 "모른다" (환각 억제 확인)
- **검색 품질 수정**: 일반어 키프레이즈 지배 → 본문·키프레이즈 **혼합 랭킹(0.6:0.4)** 으로 해결

---

## 2. 핵심 결과 — 논지 정량 검증 ✅

**"absent 키프레이즈가 어휘 불일치를 해결해 원 논문을 찾아간다"** 를 self-retrieval로 증명.
(각 문서의 absent gold(U)를 질의로 → plain 색인 vs kp 강화 색인의 검색 성공률 비교)

| 지표 | 2,000편 개선 | **20,000편 개선** |
|---|---:|---:|
| MRR | +3.7% | **+14.7%** |
| Hit@1 | +2.3% | **+31.5%** |
| Hit@10 | +7.7% | +6.2% |

**결론**: 코퍼스가 10배 커지자 **키프레이즈 강화의 상대적 이득이 급증**(Hit@1 +31.5%).
문서가 많아질수록 본문만으로는 못 찾고 **absent 키프레이즈(document expansion)가 결정적**이 됨 → 논지 성립.
(절대 수치는 문서 급증으로 하락 — 2만 편에서 정답 1편 찾기는 본질적으로 어려움.)

재현: `python rag_mvp/eval_absent_retrieval.py --limit 20000 --absent U --topk 50 --qchunk 100`

---

## 3. 알려진 한계 · 이슈

| # | 이슈 | 원인 | 영향 |
|---|---|---|---|
| I1 | **검색/평가 느림** | Chroma `where={"kind"}` 메타데이터 필터 쿼리가 22만 벡터에서 질의당 ~70ms | 2만 질의 평가 ~17분 |
| I2 | 절대 검색 성능 낮음 | absent 질의가 본질적으로 어려움 + 일반어 키프레이즈 노이즈 | Hit@1 0.017 |
| I3 | LLM 작음 | Qwen 1.5B (GPU·속도 절충) | 답변 품질 준수하나 상한 있음 |
| I4 | 프론트 빌드 없음 | Node 미설치 → CDN React | 프로덕션엔 Vite 권장 |
| I5 | 전체 KP20k 불가 | 530,809편 P7 색인 ~49시간 | test 20k로 대체 |
| I6 | 배포 안 됨 | localhost 전용 | 외부 공유 불가 |
| I7 | Chroma 단일 프로세스 | 색인·서버 동시 DB 접근 시 충돌 | 색인 중 서버 정지 필요 |

---

## 4. 개선 로드맵 (확정 — 우선순위순)

### 🥇 P0. 벡터 컬렉션 분리 (doc / keyphrase 별도) — **최우선**
- **문제 해결**: I1(느림), I7(동시성 일부)
- 문서 벡터와 키프레이즈 벡터를 **별도 컬렉션**(`*_doc`, `*_kp`)에 저장 → `where` 필터 제거 →
  순수 ANN 쿼리로 **검색·평가 대폭 가속** (질의당 70ms→수 ms 기대)
- 작업: `index.py` 2-컬렉션 구조로 리팩터 + 재색인(arXiv/KP20k). 검색 로직(혼합)은 그대로.
- 효과: 평가 17분 → 1~2분, 실시간 검색도 빨라짐.

### 🥈 P1. 검색 품질 — absent 특화 + 일반어 억제
- **문제 해결**: I2
- (a) **absent 질의용 kp-가중 모드**: 본문이 못 맞추는 질의는 키프레이즈 가중을 높여 랭킹
- (b) **일반어 키프레이즈 IDF 다운웨이트**: "language models"처럼 흔한 키프레이즈 색인 가중 ↓
- (c) 근거성 임계값 필터(환각 색인 방지)
- 효과: Hit@1/MRR 추가 상승, absent 검색 실전 품질 개선.

### 🥉 P2. LLM 업그레이드 + 스트리밍
- **문제 해결**: I3
- Qwen 1.5B → **7B**(GPU 여유 확인 후) 또는 API 옵션
- 답변 **SSE 스트리밍**(체감 속도), 근거 하이라이트

### P3. 프론트 정식화 + 배포
- **문제 해결**: I4, I6
- Node 설치 시 **Vite+React 이관**, Tailwind 빌드
- **docker-compose**(backend+frontend), 모델 캐시 볼륨, env

### P4. 코퍼스 확장
- **문제 해결**: I5(부분)
- arXiv 더 수집(1만+), KP20k train 일부 추가(선택)
- 새 문서 업로드→P7 색인 UI

### P5. 평가·시연 강화
- absent 범위 M/U 확장, topk 스윕
- 프론트 **"kp 강화 vs 본문만" 비교 토글**로 논지 실시간 시연(이미 있음 → 데모 다듬기)

---

## 5. 확정 실행 순서

```
[1] P0 벡터 컬렉션 분리 + 재색인   → 속도 병목 제거 (가장 큰 체감 개선)
[2] P1 검색 품질(absent 가중·IDF)  → 논지 수치 추가 상승
[3] P5 비교 데모 다듬기 + 평가 재측정 → "개선 전/후" 정량 대비
[4] P2 LLM 업그레이드/스트리밍     → 답변 품질·체감
[5] P3 배포(docker) / Vite 이관    → 외부 공유 가능
[6] P4 코퍼스 확장                 → 규모 키우기 (선택)
```

**MVP 완성 기준**: P0+P1까지 = "빠르고, absent로 원 논문을 확실히 더 잘 찾는" 시스템.
P2~P4는 서비스 고도화 단계.

---

## 6. 파일 맵 (`rag_mvp/`)

```
fetch_arxiv.py              arxiv 수집·전처리
ingest.py                   색인 (--source arxiv|kp20k)
eval_absent_retrieval.py    논지 평가 (배치)
search_cli.py               검색 CLI
core/keyphrases.py          P7 키프레이즈 생성
core/index.py               bge + Chroma (검색·혼합 랭킹)
core/answer.py              Qwen 인용 답변
backend/app.py              FastAPI
frontend/index.html         React SPA
data/arxiv_corpus.{csv,jsonl}   arXiv 3,000편
~/.rag_mvp/chroma/          벡터DB (OneDrive 밖)
```
