"""채점이 끝난 시트를 읽어 시스템별 정밀도·재현율·F1을 계산한다.

전제: outputs/annotation/sheet_A_적절성채점.csv 의 '적절함(O/X)' 열이 채워져 있어야 함.
      (선택) sheet_B_누락정답.csv 의 '빠진_키프레이즈'가 채워지면 recall/F1도 계산.

실행: python scripts/score_annotations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

ANN = PROJECT_ROOT / "outputs" / "annotation"

sheet = pd.read_csv(ANN / "sheet_A_적절성채점.csv", encoding="utf-8-sig")
key = pd.read_csv(ANN / "_KEY_비공개.csv", encoding="utf-8-sig")
df = sheet.merge(key, on=["논문ID", "번호"], suffixes=("", "_k"))

col = "적절함(O/X)"
df[col] = df[col].astype(str).str.strip().str.upper()
graded = df[df[col].isin(["O", "X"])]
if graded.empty:
    print("채점된 행이 없습니다 — sheet_A의 '적절함(O/X)' 열을 채운 뒤 다시 실행하세요.")
    sys.exit(0)
print(f"채점 완료: {len(graded):,} / {len(df):,} 구절 ({100*len(graded)/len(df):.0f}%)"
      f" | 논문 {graded.논문ID.nunique()}편")
df["correct"] = (df[col] == "O").astype(int)

# ---- 시스템별 정밀도 (해당 시스템이 top-K에 넣은 구절만 대상) ----
rows = []
for name, rank_col in [("Fusion (P2)", "fusion_rank"), ("Pairwise (P6)", "pairwise_rank")]:
    for K in (5, 10):
        sub = df[df[rank_col].notna() & (df[rank_col] <= K) & df[col].isin(["O", "X"])]
        if sub.empty:
            continue
        # 문서별 정밀도의 평균 (macro)
        per_doc = sub.groupby("논문ID").correct.agg(["sum", "count"])
        macro_p = (per_doc["sum"] / K).mean()          # 분모 K 고정 (F1@K 규칙과 동일)
        hit_rate = sub.correct.mean()                   # 채점된 구절 중 O 비율
        rows.append({"시스템": name, "K": K, f"정밀도@K(macro)": round(macro_p, 4),
                     "O비율": round(hit_rate, 4), "채점된구절수": len(sub)})
prec = pd.DataFrame(rows)
print("\n=== 사람 채점 기준 정밀도 ===")
print(prec.to_string(index=False))

# ---- 두 시스템 합의 여부별 정확도 ----
both = df[df.fusion_rank.notna() & df.pairwise_rank.notna() & df[col].isin(["O", "X"])]
only_f = df[df.fusion_rank.notna() & df.pairwise_rank.isna() & df[col].isin(["O", "X"])]
only_p = df[df.pairwise_rank.isna().eq(False) & df.fusion_rank.isna() & df[col].isin(["O", "X"])]
print("\n=== 두 시스템 합의 vs 단독 제안의 적중률 ===")
for label, sub in [("둘 다 제안(합의)", both), ("Fusion만 제안", only_f), ("Pairwise만 제안", only_p)]:
    if len(sub):
        print(f"  {label:<16}: {sub.correct.mean():.1%}  (n={len(sub)})")

# ---- 재현율 (시트 B가 채워진 경우) ----
b_path = ANN / "sheet_B_누락정답.csv"
if b_path.exists():
    b = pd.read_csv(b_path, encoding="utf-8-sig")
    b = b[b["빠진_키프레이즈"].notna() & (b["빠진_키프레이즈"].astype(str).str.strip() != "")]
    if len(b):
        print(f"\n=== 재현율 (사람이 추가한 정답 {len(b)}개 반영) ===")
        missed = b.groupby("논문ID").size()
        for name, rank_col in [("Fusion (P2)", "fusion_rank"), ("Pairwise (P6)", "pairwise_rank")]:
            recalls = []
            for aid, g in df[df[rank_col].notna() & (df[rank_col] <= 10)].groupby("논문ID"):
                hits = int(g.correct.sum())
                total = hits + int(missed.get(aid, 0))
                if total:
                    recalls.append(hits / total)
            if recalls:
                r = float(np.mean(recalls))
                p_at10 = prec[(prec.시스템 == name) & (prec.K == 10)]["정밀도@K(macro)"].iloc[0]
                f1 = 2 * p_at10 * r / (p_at10 + r) if p_at10 + r else 0
                print(f"  {name:<14}: Recall@10={r:.4f}  F1@10={f1:.4f}")
    else:
        print("\n(시트 B가 비어 있어 재현율은 생략 — 정밀도만 계산됨)")

out = PROJECT_ROOT / "outputs" / "metrics" / "human_evaluation.csv"
prec.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n저장: {out}")
