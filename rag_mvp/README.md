# RAG MVP — P7 키프레이즈 + 벡터DB + FastAPI + React

P7이 뽑은 키프레이즈(특히 **absent**)로 벡터 색인을 강화해 **어휘 불일치**에도 검색되게 하고,
검색 결과로 LLM이 인용 답변하는 풀스택 RAG. 설계 배경: [`../docs/RAG_MVP_PLAN.md`](../docs/RAG_MVP_PLAN.md).

## 구성

```
rag_mvp/
├─ fetch_arxiv.py      # arxiv 최신 논문 수집·전처리 → data/arxiv_corpus.{csv,jsonl}
├─ ingest.py           # 코퍼스 → P7 키프레이즈 → bge 임베딩 → Chroma 색인
├─ search_cli.py       # 검색 CLI (디버그)
├─ core/
│  ├─ keyphrases.py    # KeyBERT+KeyBART → P7(SciBERT+aux) 재랭크
│  ├─ index.py         # bge 임베더 + Chroma (색인/검색)  ※ 벡터DB는 ~/.rag_mvp/chroma
│  └─ answer.py        # LLM(Qwen2.5-1.5B) 인용 답변
├─ backend/app.py      # FastAPI: /api/{health,search,answer,ingest} + 프론트 서빙
└─ frontend/index.html # React SPA (빌드 불필요 — CDN React, 백엔드가 서빙)
```

## 준비물

- 이미 학습된 체크포인트: `outputs/checkpoints/{keybart_full, reranker_scibert}`
- 패키지: `chromadb`, `fastapi`, `uvicorn`, `sentence-transformers`, `transformers`, `torch`
- 최초 실행 시 `BAAI/bge-small-en-v1.5`(임베더)·`Qwen/Qwen2.5-1.5B-Instruct`(LLM) 자동 다운로드

## 실행 (3단계)

```bash
# 1) 코퍼스 수집 (최신 arxiv cs.CL)         — 이미 있으면 생략
python rag_mvp/fetch_arxiv.py --category cs.CL --target 3000

# 2) 색인 (P7 키프레이즈 + 임베딩 → 벡터DB)  — 한 번, ~15-20분(3000편, GPU)
python rag_mvp/ingest.py --reset

# 3) 서버 실행 → 브라우저 http://127.0.0.1:8000
python rag_mvp/backend/app.py
```

## 사용

- **검색**: 질의 입력 → 관련 논문 카드 (제목·arxiv 링크·매칭 키프레이즈·본문/키프레이즈 매칭 배지)
- **LLM 답변**: "LLM 답변 생성" 켜면 검색 근거로 인용([n]) 답변
- **비교 토글**: 답변 OFF에서 `키프레이즈 강화` vs `본문만` 색인 결과 비교 → absent 색인 효과 확인

## API

| 엔드포인트 | 설명 |
|---|---|
| `GET /api/health` | 색인 벡터 수·모델 상태 |
| `POST /api/search` | `{query, top_k, mode:"kp"\|"plain"}` → 문서 top-k |
| `POST /api/answer` | `{query, top_k}` → 인용 답변 + 근거 문서 |
| `POST /api/ingest` | `{docs:[{title,abstract}]}` → 새 문서 색인 |

## 평가 (논지 검증, 무채점)

문서의 absent gold를 질의로 self-retrieval → `kp` 색인이 `plain`보다 어휘 불일치 질의에서
Hit@k/MRR이 높은지 측정 (계획서 §7). *(평가 스크립트는 다음 단계.)*
