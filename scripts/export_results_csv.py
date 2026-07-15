"""전체 데이터셋 실험 결과를 종합 CSV로 내보낸다 (재현율 포함 전 지표)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"c:\Users\wodlf\OneDrive\Desktop\KP20K_Keyphrase_Extraction_")

import numpy as np
import pandas as pd

from src.metrics import precision_recall_f1_at_k
from src.utils import load_jsonl

OUT = Path(r"c:\Users\wodlf\OneDrive\Desktop\KP20K_Keyphrase_Extraction_\outputs")

FILES = {
    "B0_tfidf_full": ("TF-IDF", "B0_tfidf_full.jsonl"),
    "B1_keybert_full": ("KeyBERT", "B1_keybert_full.jsonl"),
    "B2_keybert_mmr_full": ("KeyBERT+MMR", "B2_keybert_mmr_full.jsonl"),
    "B3_bart_beam5_full": ("BART", "B3_bart_beam5_full.jsonl"),
    "B4_keybart_beam5_full": ("KeyBART", "B4_keybart_beam5_full.jsonl"),
    "P1_hybrid_fusion_full": ("Hybrid (fusion)", "P1_hybrid_fusion_full.jsonl"),
    "P2_hybrid_fusion_mmr_full": ("Hybrid+MMR (final)", "P2_hybrid_mmr_full.jsonl"),
}

# 예측 파일에서 P/R@K 재계산 (기록에 없던 recall 컬럼 보충)
pr_rows = {}
for run_id, (model, fn) in FILES.items():
    path = OUT / "predictions" / fn
    if not path.exists():
        print(f"skip (파일 없음): {fn}")
        continue
    preds = load_jsonl(path)
    metrics = {f"{m}@{k}": [] for m in ("P", "R") for k in (5, 10)}
    for r in preds:
        for k in (5, 10):
            res = precision_recall_f1_at_k(r["pred"], r["gold"], k)
            metrics[f"P@{k}"].append(res["precision"])
            metrics[f"R@{k}"].append(res["recall"])
    pr_rows[run_id] = {c: round(float(np.mean(v)), 4) for c, v in metrics.items()}
    print(f"computed: {model} ({len(preds):,} docs)")

# 기록된 지표(experiments.csv)와 병합
exp = pd.read_csv(OUT / "metrics" / "experiments.csv")
exp = exp[exp.run_id.isin(FILES)].set_index("run_id")

rows = []
for run_id, (model, _) in FILES.items():
    if run_id not in pr_rows or run_id not in exp.index:
        continue
    e = exp.loc[run_id]
    rows.append({
        "run_id": run_id,
        "model": model,
        "train_data": "530,809 (full)" if not pd.isna(e.get("train_subset")) else "-(무학습)",
        "eval_docs": 20000,
        **pr_rows[run_id],
        "F1@5": e.get("F1@5"), "F1@10": e.get("F1@10"), "F1@M": e.get("F1@M"),
        "present_F1@5": e.get("present_F1@5"), "absent_R@5": e.get("absent_R@5"),
        "absent_R@10": e.get("absent_R@10"),
        "recall_P": e.get("recall_P"), "recall_R": e.get("recall_R"),
        "recall_M": e.get("recall_M"), "recall_U": e.get("recall_U"),
        "MAP@10": e.get("MAP@10"), "nDCG@10": e.get("nDCG@10"),
        "dup_ratio": e.get("dup_ratio"), "avg_preds_per_doc": e.get("num_pred"),
    })
df = pd.DataFrame(rows)
main_path = OUT / "metrics" / "results_full_dataset.csv"
df.to_csv(main_path, index=False, encoding="utf-8-sig")   # BOM: 엑셀 한글 호환
print("\nsaved:", main_path)

# 후보 pool recall (선택 전 상한선) — 별도 CSV
cr = json.load(open(OUT / "metrics" / "candidate_recall_full.json", encoding="utf-8"))
cr_df = pd.DataFrame(cr).T.round(4)
cr_df.index.name = "candidate_source"
cr_path = OUT / "metrics" / "results_candidate_recall_full.csv"
cr_df.to_csv(cr_path, encoding="utf-8-sig")
print("saved:", cr_path)

# MMR λ 스윕 — 별도 CSV
sweep = json.load(open(OUT / "metrics" / "mmr_sweep_full.json", encoding="utf-8"))
sw_df = pd.DataFrame(sweep).T
sw_df.index.name = "mmr_setting"
sw_path = OUT / "metrics" / "results_mmr_sweep_full.csv"
sw_df.to_csv(sw_path, encoding="utf-8-sig")
print("saved:", sw_path)

pd.set_option("display.width", 250)
print("\n=== results_full_dataset.csv 미리보기 ===")
print(df[["model", "R@5", "R@10", "F1@5", "F1@M", "present_F1@5", "absent_R@10",
          "recall_P", "recall_U", "nDCG@10"]].to_string(index=False))
