# 아키텍처 — P7 키프레이즈 RAG MVP

> 문서(제목+초록) → **P7이 키프레이즈(특히 absent) 생성** → 벡터DB 색인 → 어휘가 달라도 검색 → LLM 인용 답변.
> 3계층(React · FastAPI · 벡터DB+모델) 풀스택, GPU 도커로 배포.

---

## 1. 시스템 아키텍처 (3계층)

```
┌──────────────────────────────────────────────────────────┐
│  브라우저 · React SPA            (frontend/index.html)      │
│  검색창 · 결과 카드(키프레이즈 칩) · LLM 답변 · 코퍼스/모드 토글 │
└────────────────────────────┬─────────────────────────────┘
                             │  REST / JSON  (fetch)
┌────────────────────────────▼─────────────────────────────┐
│  FastAPI 백엔드                    (backend/app.py)         │
│  POST /api/search   POST /api/answer                       │
│  POST /api/ingest   GET  /api/health                       │
│  (모델은 起動 시 1회 로드, 요청 간 재사용)                     │
└──────┬────────────────────┬───────────────────────┬───────┘
       ▼                    ▼                       ▼
┌─────────────┐     ┌──────────────────┐    ┌──────────────┐
│  Chroma DB  │     │  모델 (GPU)        │    │  Qwen LLM     │
│  벡터 색인   │     │  bge · P7 · KeyBART │    │  인용 답변     │
│  (코사인)   │     │  · KeyBERT         │    │  (지연 로드)   │
└─────────────┘     └──────────────────┘    └──────────────┘
        │ 영속: ~/.rag_mvp/chroma
        └ arXiv 33k · KP20k 220k · mixed 242k 벡터
```

---

## 2. 색인 파이프라인 (문서 넣을 때)

```
문서 (title + abstract)
   │
   ├─ KeyBERT 추출  (all-MiniLM) ───┐  present 후보
   ├─ KeyBART 생성  (beam5) ────────┤  absent 포함 후보
   │                               ├─→ 후보 병합
   └─ P7 재랭크 (SciBERT + aux) ─────┘  → 키프레이즈 top-10  [present · absent]
                                              │
   본문(title+abstract) ───────────────────────┤
   키프레이즈 N개 ─────────────────────────────┤─→ bge-small 임베딩 (384차원)
                                              │
                                              ▼
                        Chroma 색인:  문서 벡터 1  +  키프레이즈 벡터 N
                        (absent 키프레이즈 = document expansion)
```

핵심: 문서 하나가 **본문 1 + 키프레이즈 N 벡터**로 여러 각도 색인 → 본문에 없는 개념(absent)까지 검색 대상.

---

## 3. 검색 · 답변 흐름 (질의 들어올 때)

```
질의 (query)
   │→ bge 임베딩  (질의 접두어 "Represent this sentence for searching…")
   ▼
Chroma 코사인 검색
   ├─ 본문 벡터 유사도      doc_sim
   └─ 키프레이즈 벡터 유사도  kp_sim  (문서별 최댓값)
   ▼
혼합 랭킹     score = 0.6 · doc_sim  +  0.4 · kp_sim     →  문서 top-k
   │                                                        │
   │                                        (LLM 답변 ON일 때)▼
   │                                   상위 문서 초록 → Qwen → 인용 답변 [1][2]
   ▼                                   (근거 없으면 "모른다" — 환각 억제)
결과 카드: 제목 · arXiv 링크 · 매칭 키프레이즈(absent 강조) · 본문/키프레이즈 매칭 배지
```

- `mode=kp` : 본문+키프레이즈 혼합 (기본) · `mode=plain` : 본문만 (비교용)
- 임베딩은 **bge(bi-encoder)** 담당, **P7은 색인 재료(키프레이즈) 생성** — 검색 벡터를 만들지 않음.

---

## 4. 모델 구성 (역할 분담)

| 모델 | 체크포인트/ID | 방식 | 역할 |
|---|---|---|---|
| **bge-small-en-v1.5** | HF | bi-encoder | **검색 임베딩** (색인·질의 벡터) |
| **P7** | `reranker_scibert` (SciBERT+aux, KP20k 학습) | **cross-encoder** | 키프레이즈 **재랭크** |
| **KeyBART** | `keybart_full` (KP20k 전체 fine-tune) | seq2seq | **absent 포함 후보 생성** |
| **KeyBERT** | all-MiniLM-L6-v2 | bi-encoder | present 후보 **추출** |
| **Qwen2.5-1.5B-Instruct** | HF | LLM | **인용 답변** (지연 로드) |

> P7/KeyBART가 KP20k로 학습됐지만, **학습 안 한 arXiv 논문에도 키프레이즈를 생성·색인** → 재학습 없이 새 논문 검색 가능(일반화).

---

## 5. 데이터 (코퍼스)

| 코퍼스 | 문서 | 벡터 | 용도 |
|---|---|---|---|
| arXiv cs.CL | 3,000 (2026 최신) | 33,000 | 사용자용 "최신 논문 검색" |
| KP20k test | 20,000 (gold 보유) | 220,000 | 논지 정량 평가 |
| mixed | 22,000 (KP20k+arXiv) | 242,000 | 혼합 강건성 |

색인 위치: `~/.rag_mvp/chroma` (OneDrive 밖 로컬, 영속)

---

## 6. 배포 (Docker + 터널)

```
[ 내 PC · RTX 5080 (GPU) ]
    │
    ├─ Docker 컨테이너  p7-rag-mvp        (Dockerfile · docker-compose.yml)
    │    · uvicorn  0.0.0.0:8000
    │    · GPU 예약(nvidia)  · 볼륨: 벡터DB · HF캐시 · 체크포인트
    │    · 켜기 docker compose up -d  /  끄기 docker compose stop
    │
    └─ cloudflared 터널  (--url http://localhost:8000)
         │
         ▼
   https://<랜덤>.trycloudflare.com   ←  공개 URL (PC 켜져 있는 동안)
```

- **로컬**: `http://localhost:8000` · **같은 네트워크**: `http://<내IP>:8000` · **인터넷**: 터널 URL
- GPU는 내 PC 것을 사용 → PC/컨테이너/터널이 떠 있는 동안만 접속 가능

---

## 7. 파일 맵

```
rag_mvp/
├─ frontend/index.html      React SPA (CDN, 빌드 불필요)
├─ backend/app.py           FastAPI (검색·답변·색인·서빙)
├─ core/keyphrases.py       KeyBERT+KeyBART → P7 재랭크
├─ core/index.py            bge 임베더 + Chroma (혼합 랭킹)
├─ core/answer.py           Qwen 인용 답변
├─ ingest.py                색인 (--source arxiv|kp20k)
├─ eval_absent_fast.py      논지 평가 (GPU 행렬곱, 빠름)
├─ Dockerfile · docker-compose.yml   GPU 배포
└─ DEPLOY.md                배포 가이드
```

> 흐름 요약: **[색인] 문서→P7 키프레이즈→bge 임베딩→Chroma  ·  [검색] 질의→bge→코사인→혼합랭킹→(Qwen 답변)**
