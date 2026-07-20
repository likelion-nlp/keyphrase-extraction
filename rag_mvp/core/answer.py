"""LLM 답변 — 검색된 논문을 근거로 인용([1][2]) 답변 생성. Qwen 지연 로드.

검색 결과에 없으면 "모른다"고 답하도록 프롬프트로 강제 (환각 억제).
"""
from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM = ("You are a research assistant for arXiv papers. Answer the question using ONLY the "
          "provided papers. Cite each claim inline with [n] matching the paper number. If the "
          "papers do not contain the answer, say you don't know. Be concise (3-5 sentences).")


class Answerer:
    def __init__(self, device: str = "cuda", model: str = LLM_MODEL):
        self.device, self.model_name = device, model
        self.tok = self.model = None

    def _lazy(self):
        if self.model is None:
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16).to(self.device).eval()

    @torch.no_grad()
    def answer(self, query: str, contexts: list[dict], max_new_tokens: int = 350) -> dict:
        self._lazy()
        ctx = "\n\n".join(f"[{i+1}] {c['title']}\n{(c.get('abstract') or '')[:600]}"
                          for i, c in enumerate(contexts))
        user = f"Papers:\n{ctx}\n\nQuestion: {query}\n\nAnswer (cite with [n]):"
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        ans = self.tok.decode(gen, skip_special_tokens=True).strip()
        cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", ans)
                        if 1 <= int(m) <= len(contexts)})
        return {"answer": ans, "cited": [contexts[i - 1]["doc_id"] for i in cited]}
