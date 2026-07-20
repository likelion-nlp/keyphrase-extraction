"""벡터 색인 — bi-encoder(bge) 임베딩 + Chroma 저장/검색.

문서마다 [본문 벡터 1개 + 키프레이즈 벡터 N개]를 색인한다 (키프레이즈가 document expansion).
검색은 질의 임베딩 → Chroma 유사도 → doc_id로 히트 집계.
"""
from __future__ import annotations

from pathlib import Path

import torch  # noqa: F401  ← 반드시 chromadb보다 먼저 (Windows DLL 로드 순서 충돌 방지: 세그폴트)
import chromadb
from sentence_transformers import SentenceTransformer

RAG = Path(__file__).resolve().parent.parent
# 벡터DB는 OneDrive 밖 로컬에 (동기화 잠금/오버헤드 회피)
CHROMA_DIR = Path.home() / ".rag_mvp" / "chroma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge-v1.5: 질의에 검색 지시문을 붙이면 성능↑ (문서 쪽은 그대로)
Q_PREFIX = "Represent this sentence for searching relevant passages: "


_EMBEDDER = None


def get_embedder(device: str = "cuda", model: str = EMBED_MODEL) -> SentenceTransformer:
    """bge 임베더 모듈 싱글턴 (여러 컬렉션이 공유)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model, device=device)
    return _EMBEDDER


class VectorIndex:
    def __init__(self, device: str = "cuda", embed_model: str = EMBED_MODEL,
                 collection: str = "papers"):
        self.embedder = get_embedder(device, embed_model)
        self.collection = collection
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"})

    def _embed(self, texts: list[str], is_query: bool = False):
        if is_query:
            texts = [Q_PREFIX + t for t in texts]
        return self.embedder.encode(texts, normalize_embeddings=True,
                                    batch_size=128, show_progress_bar=False).tolist()

    def count(self) -> int:
        return self.col.count()

    def reset(self) -> None:
        """색인 비우기 — 파일 삭제 대신 컬렉션 재생성 (파일 잠금 회피)."""
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self.col = self.client.get_or_create_collection(
            self.collection, metadata={"hnsw:space": "cosine"})

    def add_documents(self, docs: list[dict]) -> int:
        """docs: [{doc_id, title, abstract, keyphrases:[{phrase,present,prmu}], meta:{...}}]
        각 문서 → 본문 벡터 1 + 키프레이즈 벡터 N."""
        ids, texts, metas = [], [], []
        for d in docs:
            base = {k: v for k, v in d.get("meta", {}).items()}
            base.update({"doc_id": d["doc_id"], "title": d["title"]})
            ids.append(f"{d['doc_id']}::doc")
            texts.append(f"{d['title']}. {d['abstract']}")
            metas.append({**base, "kind": "doc", "text_snip": d["abstract"][:400]})
            for j, kp in enumerate(d.get("keyphrases", [])):
                ids.append(f"{d['doc_id']}::kp{j}")
                texts.append(kp["phrase"])
                metas.append({**base, "kind": "keyphrase", "phrase": kp["phrase"],
                              "prmu": kp["prmu"], "present": bool(kp["present"])})
        if not ids:
            return 0
        embs = self._embed(texts)
        # Chroma는 대량 add를 배치로 (5,000개 단위)
        for s in range(0, len(ids), 5000):
            e = s + 5000
            self.col.add(ids=ids[s:e], embeddings=embs[s:e],
                         metadatas=metas[s:e], documents=texts[s:e])
        return len(ids)

    # 본문·키프레이즈 혼합 가중 (일반어 키프레이즈 단독 지배 방지: 본문도 맞아야 상위)
    W_DOC, W_KP, KP_ONLY_PENALTY = 0.6, 0.4, 0.8

    def _dist_to_sim(self, res):
        return [(m, 1.0 - float(d)) for m, d in zip(res["metadatas"][0], res["distances"][0])]

    def search(self, query: str, top_k: int = 5, mode: str = "kp", pool: int = 150) -> list[dict]:
        """본문 벡터 유사도 + 키프레이즈 벡터 유사도를 혼합해 문서 순위 결정.
        mode='plain'이면 본문만 사용(비교용)."""
        qv = self._embed([query], is_query=True)

        # ① 본문 벡터 검색
        doc_sim, meta = {}, {}
        for m, sim in self._dist_to_sim(self.col.query(query_embeddings=qv, n_results=pool,
                                                       where={"kind": "doc"})):
            doc_sim[m["doc_id"]] = sim
            meta[m["doc_id"]] = m

        # ② 키프레이즈 벡터 검색 (mode='kp'에서만)
        kp_best: dict[str, list] = {}
        if mode == "kp":
            for m, sim in self._dist_to_sim(self.col.query(query_embeddings=qv, n_results=pool,
                                                           where={"kind": "keyphrase"})):
                did = m["doc_id"]
                e = kp_best.setdefault(did, [0.0, []])
                e[0] = max(e[0], sim)
                tag = "[A]" if not m.get("present") else "[P]"
                e[1].append((sim, f"{m['phrase']}{tag}"))
                meta.setdefault(did, m)

        # ③ 혼합 점수
        scored = []
        for did in set(doc_sim) | set(kp_best):
            ds = doc_sim.get(did)
            ks = kp_best.get(did, [0.0, []])[0]
            if ds is not None and did in kp_best:
                score, via = self.W_DOC * ds + self.W_KP * ks, ("keyphrase" if ks > ds else "doc")
            elif ds is not None:
                score, via = ds, "doc"
            else:
                score, via = self.KP_ONLY_PENALTY * ks, "keyphrase"   # 본문 히트 없음 → 페널티
            scored.append((did, score, via))
        scored.sort(key=lambda x: -x[1])

        out = []
        for did, score, via in scored[:top_k]:
            m = meta[did]
            kps = [p for _, p in sorted(kp_best.get(did, [0, []])[1], reverse=True)][:5]
            out.append({"doc_id": did, "title": m["title"], "score": round(score, 4),
                        "hit_via": via, "matched_keyphrases": kps,
                        "abstract": m.get("text_snip", ""),
                        "authors": m.get("authors", ""), "published": m.get("published", ""),
                        "category": m.get("primary_category", ""), "abs_url": m.get("abs_url", "")})
        return out
