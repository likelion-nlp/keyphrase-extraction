"""RAG MVP 백엔드 — FastAPI. 검색/답변/색인 API + React 프론트 정적 서빙.

    uvicorn rag_mvp.backend.app:app --host 127.0.0.1 --port 8000
    (또는  python rag_mvp/backend/app.py)
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

RAG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG.parent))

import torch  # noqa: F401  chromadb보다 먼저 (DLL 순서)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_mvp.core.answer import Answerer
from rag_mvp.core.index import VectorIndex

STATE: dict = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


COLLECTIONS = {"arxiv": "papers", "kp20k": "kp20k", "mixed": "mixed"}   # 코퍼스 → Chroma 컬렉션


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["indices"] = {name: VectorIndex(device=DEVICE, collection=col)  # 임베더는 싱글턴 공유
                        for name, col in COLLECTIONS.items()}
    STATE["answerer"] = Answerer(device=DEVICE)     # LLM 지연 로드
    STATE["kpgen"] = None                           # 키프레이즈 생성기 지연 로드
    counts = {k: v.count() for k, v in STATE["indices"].items()}
    print(f"[startup] 코퍼스 {counts} · device={DEVICE}")
    yield


def pick(corpus: str) -> VectorIndex:
    return STATE["indices"].get(corpus, STATE["indices"]["arxiv"])


app = FastAPI(title="P7 Keyphrase RAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SearchReq(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "kp"          # 'kp'(키프레이즈 강화) | 'plain'(본문만)
    corpus: str = "arxiv"     # 'arxiv'(최신) | 'kp20k'


class AnswerReq(BaseModel):
    query: str
    top_k: int = 4
    corpus: str = "arxiv"


class IngestReq(BaseModel):
    docs: list                # [{title, abstract, doc_id?}]
    corpus: str = "arxiv"


@app.get("/api/health")
def health():
    return {"status": "ok", "corpora": {k: v.count() for k, v in STATE["indices"].items()},
            "device": DEVICE, "llm": STATE["answerer"].model_name}


@app.post("/api/search")
def search(req: SearchReq):
    results = pick(req.corpus).search(req.query, top_k=req.top_k, mode=req.mode)
    return {"query": req.query, "mode": req.mode, "corpus": req.corpus, "results": results}


@app.post("/api/answer")
def answer(req: AnswerReq):
    results = pick(req.corpus).search(req.query, top_k=req.top_k, mode="kp")
    contexts = [{"doc_id": r["doc_id"], "title": r["title"], "abstract": r["abstract"]}
                for r in results]
    if not contexts:
        return {"query": req.query, "answer": "관련 논문을 찾지 못했습니다.", "cited": [], "results": []}
    out = STATE["answerer"].answer(req.query, contexts)
    return {"query": req.query, "answer": out["answer"], "cited": out["cited"], "results": results}


@app.post("/api/ingest")
def ingest(req: IngestReq):
    if STATE["kpgen"] is None:
        from rag_mvp.core.keyphrases import KeyphraseGenerator
        STATE["kpgen"] = KeyphraseGenerator(device=DEVICE)
    docs = req.docs
    kps = STATE["kpgen"].generate([{"title": d["title"], "abstract": d["abstract"]} for d in docs])
    payload = []
    for i, (d, kp) in enumerate(zip(docs, kps)):
        payload.append({"doc_id": d.get("doc_id", f"user-{i}"), "title": d["title"],
                        "abstract": d["abstract"], "keyphrases": kp, "meta": {}})
    n = pick(req.corpus).add_documents(payload)
    return {"indexed_docs": len(docs), "added_vectors": n,
            "keyphrases": [[k["phrase"] for k in kp] for kp in kps]}


# ---- 프론트엔드 정적 서빙 ----
FRONT = RAG / "frontend"
if FRONT.exists():
    app.mount("/static", StaticFiles(directory=str(FRONT)), name="static")


@app.get("/")
def root():
    # no-cache: 프론트 갱신이 브라우저 캐시에 안 막히게
    return FileResponse(str(FRONT / "index.html"), headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
