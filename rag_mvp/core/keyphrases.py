"""P7 키프레이즈 생성기 — KeyBERT 추출 + KeyBART 생성 → P7(SciBERT+aux) 재랭크.

문서(title+abstract) → 상위 K개 키프레이즈 [{phrase, present, prmu, score}].
RAG 색인의 재료(특히 absent)를 만든다. (기존 src/·scripts/ 파이프라인 재사용)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent      # KP20K 프로젝트 루트
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data import build_source, plain_doc_text
from src.extraction import KeyBertExtractor
from src.generation import generate_keyphrases
from src.preprocessing import classify_prmu, stem_tokens
from src.reranking import merge_candidates
from scibert_hybrid_ranker import HybridRanker, MAX_LEN, make_aux

CKPT = PROJ / "outputs" / "checkpoints"


class KeyphraseGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        kb = CKPT / "keybart_full"
        self.gen_tok = AutoTokenizer.from_pretrained(str(kb))
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(str(kb)).to(device).eval()
        self.extractor = KeyBertExtractor(device=device)
        p7 = CKPT / "reranker_scibert"
        self.p7_tok = AutoTokenizer.from_pretrained(str(p7))
        self.p7 = HybridRanker().to(device)
        self.p7.load_state_dict(torch.load(p7 / "model.pt", map_location=device))
        self.p7.eval()

    @torch.no_grad()
    def _score(self, text: str, recs: list[dict], batch: int = 64) -> list[float]:
        out: list[float] = []
        for s in range(0, len(recs), batch):
            ch = recs[s : s + batch]
            enc = self.p7_tok([text] * len(ch), [r["phrase"] for r in ch],
                              truncation="only_first", max_length=MAX_LEN,
                              padding=True, return_tensors="pt").to(self.device)
            aux = torch.tensor([make_aux(r["gen_score"], r["source"] in ("gen", "both"),
                                         r["is_present"]) for r in ch],
                               dtype=torch.float).to(self.device)
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sc = self.p7(enc["input_ids"], enc["attention_mask"], aux)
            else:
                sc = self.p7(enc["input_ids"], enc["attention_mask"], aux)
            out.extend(sc.float().cpu().tolist())
        return out

    @torch.no_grad()
    def generate(self, docs: list[dict], top_k: int = 10, batch_gen: int = 4) -> list[list[dict]]:
        texts = [plain_doc_text(d["title"], d["abstract"]) for d in docs]
        sources = [build_source(d["title"], d["abstract"]) for d in docs]
        gen_pool = generate_keyphrases(self.gen_model, self.gen_tok, sources,
                                       strategy="beam5", max_source_length=384, batch_size=batch_gen)
        ext = self.extractor.extract_batch(texts, top_n=30)

        results = []
        for i, text in enumerate(texts):
            merged = merge_candidates(ext[i], gen_pool[i])
            ds = stem_tokens(text)
            recs = [{"phrase": c["phrase"], "gen_score": c["gen_score"],
                     "source": "both" if len(c["sources"]) == 2 else c["sources"][0][:3],
                     "is_present": classify_prmu(c["phrase"], "", ds) == "P"} for c in merged]
            if not recs:
                results.append([]); continue
            scores = self._score(text, recs)
            order = sorted(range(len(recs)), key=lambda j: -scores[j])[:top_k]
            results.append([{"phrase": recs[j]["phrase"], "present": recs[j]["is_present"],
                             "prmu": "P" if recs[j]["is_present"] else "A",
                             "score": round(float(scores[j]), 4)} for j in order])
        return results
