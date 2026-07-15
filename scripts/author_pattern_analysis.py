"""저자 키프레이즈 선정 패턴 분석 — 전체 KP20k train 530,809편 (CPU 전용).

Qwen LLM 정성 분석(Phase B)에 앞서, 정량으로 잡히는 패턴을 전수 조사한다:
  1. 개수·길이 분포        — 몇 개를, 몇 단어짜리로 다는가
  2. 위치 패턴             — 제목/초록 첫 문장에서 뽑는가
  3. 관행어 집중도          — 소수의 단골 키워드가 얼마나 재사용되는가 (Zipf)
  4. 문서별 구성 공식       — "일반어 몇 개 + 전문어 몇 개" 조합 패턴
  5. 축약어 사용            — 두문자어 비율
  6. PRMU·순서             — present/absent 배치 관행

실행: python scripts/author_pattern_analysis.py
산출: outputs/metrics/author_patterns.json + 콘솔 리포트
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data import load_kp20k
from src.preprocessing import normalize_phrase
from src.utils import save_json

T0 = time.time()
ACRO_RE = re.compile(r"^[a-z0-9]{2,6}$")   # 소문자화된 짧은 단일 토큰 (svm, lstm, qos...)


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


log("KP20k train 전체 로드")
train = load_kp20k(["train"], seed=42)["train"]
N = len(train)

kp_counts, kp_word_lens = [], []
in_title_flags, first_sent_flags = [], []
prmu_all: Counter = Counter()
absent_positions = []          # absent(R/M/U)가 목록에서 어디에 오는가 (상대 위치)
phrase_freq: Counter = Counter()
acronym_cnt = 0
total_kp = 0

log("전수 스캔 (530,809편)")
for i, r in enumerate(train):
    kps, prmu = r["keyphrases"], r["prmu"]
    title_norm = normalize_phrase(r["title"] or "")
    first_sent = (r["abstract"] or "").split(". ")[0]
    first_norm = normalize_phrase(first_sent)
    kp_counts.append(len(kps))
    for j, (kp, t) in enumerate(zip(kps, prmu)):
        total_kp += 1
        norm = normalize_phrase(kp)
        words = kp.split()
        kp_word_lens.append(len(words))
        phrase_freq[norm] += 1
        prmu_all[t] += 1
        if norm and norm in title_norm:
            in_title_flags.append(1)
        else:
            in_title_flags.append(0)
        if norm and norm in first_norm:
            first_sent_flags.append(1)
        else:
            first_sent_flags.append(0)
        if len(words) == 1 and ACRO_RE.match(kp.lower()) and not kp.lower().isdigit():
            acronym_cnt += 1
        if t != "P" and len(kps) > 1:
            absent_positions.append(j / (len(kps) - 1))
    if (i + 1) % 100_000 == 0:
        log(f"  {i + 1:,}/{N:,}")

kp_counts = np.array(kp_counts)
kp_word_lens = np.array(kp_word_lens)

# ---- 관행어 집중도 (Zipf) ----
top = phrase_freq.most_common(100)
top100_share = sum(c for _, c in top) / total_kp
top1000_share = sum(c for _, c in phrase_freq.most_common(1000)) / total_kp
distinct = len(phrase_freq)

# ---- 문서별 구성 공식: 단골(df≥1000) vs 희귀(df≤5) 조합 ----
log("문서별 일반어/전문어 구성 분석")
generic_set = {p for p, c in phrase_freq.items() if c >= 1000}
rare_thresh = 5
mix_counter: Counter = Counter()
generic_per_doc, rare_per_doc = [], []
for r in train.select(range(0, N, 5)):   # 20% 표본이면 구성 통계는 충분
    kps = [normalize_phrase(k) for k in r["keyphrases"]]
    g = sum(1 for k in kps if k in generic_set)
    rr = sum(1 for k in kps if phrase_freq.get(k, 0) <= rare_thresh)
    generic_per_doc.append(g)
    rare_per_doc.append(rr)
    mix_counter[(min(g, 3), min(rr, 4))] += 1

result = {
    "corpus": {"docs": N, "total_keyphrases": total_kp,
               "distinct_normalized": distinct},
    "1_개수": {
        "평균": round(float(kp_counts.mean()), 2),
        "중앙값": int(np.median(kp_counts)),
        "3~6개 비율": round(float(((kp_counts >= 3) & (kp_counts <= 6)).mean()), 4),
    },
    "2_길이": {
        "평균 단어수": round(float(kp_word_lens.mean()), 2),
        "1단어": round(float((kp_word_lens == 1).mean()), 4),
        "2단어": round(float((kp_word_lens == 2).mean()), 4),
        "3단어 이상": round(float((kp_word_lens >= 3).mean()), 4),
    },
    "3_위치": {
        "제목에 등장": round(float(np.mean(in_title_flags)), 4),
        "초록 첫문장에 등장": round(float(np.mean(first_sent_flags)), 4),
    },
    "4_관행어_집중도": {
        "top100 구절이 차지하는 비율": round(top100_share, 4),
        "top1000 구절이 차지하는 비율": round(top1000_share, 4),
        "최다 20개": [(p, c) for p, c in top[:20]],
    },
    "5_문서별_구성": {
        "문서당 단골어(전체서 1000회+) 평균": round(float(np.mean(generic_per_doc)), 2),
        "문서당 희귀어(5회 이하) 평균": round(float(np.mean(rare_per_doc)), 2),
        "단골0+희귀2이상(전문어만) 문서 비율": round(
            float(np.mean([(g == 0 and rr >= 2) for g, rr in zip(generic_per_doc, rare_per_doc)])), 4),
        "단골1개 이상 포함 문서 비율": round(float(np.mean([g >= 1 for g in generic_per_doc])), 4),
    },
    "6_축약어": {"단일토큰 축약어 추정 비율": round(acronym_cnt / total_kp, 4)},
    "7_PRMU_순서": {
        "PRMU 비율": {t: round(prmu_all[t] / total_kp, 4) for t in "PRMU"},
        "absent의 목록 내 평균 상대위치(0=처음,1=끝)": round(float(np.mean(absent_positions)), 4),
    },
}
save_json(result, PROJECT_ROOT / "outputs" / "metrics" / "author_patterns.json")

print("\n" + "=" * 70)
for section, vals in result.items():
    print(f"\n[{section}]")
    if isinstance(vals, dict):
        for k, v in vals.items():
            if k == "최다 20개":
                print(f"  {k}:")
                for p, c in v:
                    print(f"     {c:>6,}회  {p}")
            else:
                print(f"  {k}: {v}")
log("저장: outputs/metrics/author_patterns.json")
