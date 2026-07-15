"""신규 논문 예측에 대한 블라인드 채점표를 만든다 (사람 평가용).

두 시스템(fusion/pairwise)의 top-10을 합쳐(pooling) 섞고 시스템명을 숨긴 채 채점하게 한다.
→ 채점자가 특정 시스템을 편들 수 없으므로 공정한 precision 비교가 가능하다 (IR pooling 관행).

산출물:
    outputs/annotation/sheet_A_적절성채점.csv   ← 예측 구절을 O/X로 채점 (precision용)
    outputs/annotation/sheet_B_누락정답.csv     ← 빠진 키프레이즈를 직접 적기 (recall용)
    outputs/annotation/_KEY_비공개.csv          ← 정답표(어느 구절이 어느 시스템인지). 채점 중엔 열지 말 것!

실행: python scripts/make_annotation_sheet.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

PRED = PROJECT_ROOT / "outputs" / "predictions"
ANN = PROJECT_ROOT / "outputs" / "annotation"
ANN.mkdir(parents=True, exist_ok=True)

fu = pd.read_csv(PRED / "newpapers_cs_papers_20232024_keyphrases.csv", encoding="utf-8-sig")
pw = pd.read_csv(PRED / "newpapers_cs_papers_20232024_pairwise_keyphrases.csv", encoding="utf-8-sig")
papers = pd.read_csv(PROJECT_ROOT / "cs_papers_20232024.csv")
abstracts = dict(zip(papers.arxiv_id, papers.abstract))

rng = random.Random(42)
sheet_rows, key_rows = [], []

for aid, grp in fu.groupby("arxiv_id", sort=False):
    title = grp.title.iloc[0]
    fu_map = dict(zip(grp.keyphrase, grp["rank"]))
    pw_grp = pw[pw.arxiv_id == aid]
    pw_map = dict(zip(pw_grp.keyphrase, pw_grp["rank"]))

    # 두 시스템의 top-10을 합집합(pool)으로 — 소문자 기준 중복 병합
    pool: dict[str, dict] = {}
    for phrase, rank in fu_map.items():
        pool.setdefault(phrase.lower(), {"phrase": phrase, "fusion_rank": None, "pairwise_rank": None})
        pool[phrase.lower()]["fusion_rank"] = int(rank)
    for phrase, rank in pw_map.items():
        pool.setdefault(phrase.lower(), {"phrase": phrase, "fusion_rank": None, "pairwise_rank": None})
        pool[phrase.lower()]["pairwise_rank"] = int(rank)

    items = list(pool.values())
    rng.shuffle(items)   # 시스템 순서 편향 제거

    for n, it in enumerate(items, 1):
        sheet_rows.append({
            "논문ID": aid,
            "제목": title,
            "번호": n,
            "키프레이즈": it["phrase"],
            "적절함(O/X)": "",          # ← 채점자가 입력
            "비고": "",                  # ← 선택 입력 (예: 너무 일반적, 오타 등)
        })
        key_rows.append({
            "논문ID": aid, "번호": n, "키프레이즈": it["phrase"],
            "fusion_rank": it["fusion_rank"], "pairwise_rank": it["pairwise_rank"],
        })

sheet = pd.DataFrame(sheet_rows)
sheet.to_csv(ANN / "sheet_A_적절성채점.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(key_rows).to_csv(ANN / "_KEY_비공개.csv", index=False, encoding="utf-8-sig")

# 시트 B: 모델이 놓친 정답을 직접 적는 칸 (recall 측정용)
b_rows = []
for aid, grp in fu.groupby("arxiv_id", sort=False):
    for slot in range(1, 6):   # 논문당 최대 5개까지 추가 입력
        b_rows.append({
            "논문ID": aid,
            "제목": grp.title.iloc[0] if slot == 1 else "",
            "초록": (str(abstracts.get(aid, ""))[:400] + "...") if slot == 1 else "",
            "빠진_키프레이즈": "",     # ← 채점자가 입력
        })
pd.DataFrame(b_rows).to_csv(ANN / "sheet_B_누락정답.csv", index=False, encoding="utf-8-sig")

n_papers = fu.arxiv_id.nunique()
print(f"논문 {n_papers}편 | 채점 대상 구절 {len(sheet):,}개 (논문당 평균 {len(sheet)/n_papers:.1f}개)")
print(f"  두 시스템 공통 제안: {sum(1 for r in key_rows if r['fusion_rank'] and r['pairwise_rank']):,}개")
print()
print("생성:")
print(f"  {ANN / 'sheet_A_적절성채점.csv'}   ← 여기에 O/X 입력")
print(f"  {ANN / 'sheet_B_누락정답.csv'}     ← 빠진 키프레이즈 입력 (선택)")
print(f"  {ANN / '_KEY_비공개.csv'}          ← 채점 끝날 때까지 열지 마세요")
print()
print("채점 요령:")
print("  O = 이 논문의 키프레이즈로 적절하다 (논문 검색·색인에 쓸 만하다)")
print("  X = 부적절하다 (너무 일반적/무관/문법 파괴/중복)")
print("  애매하면 '이 구절로 논문을 검색했을 때 이 논문이 나와야 하는가'로 판단")
print()
print("채점 후: python scripts/score_annotations.py 를 실행하면 시스템별 정밀도가 계산됩니다.")
