# notebooks/ — 파이프라인 정합 워크스루

이 노트북들은 **`src/`·`scripts/`의 실제 파이프라인 코드와 정합한다.** 학습·추론은 `scripts/`가 수행하고,
노트북은 **같은 `src` 모듈을 import**하고 **`results/`에 저장된 실제 결과를 로드**해 설명·시각화한다.
→ 노트북의 수치는 리포트값(test 20,000편, 전체 데이터 학습)과 **정확히 일치**하며 재구현 드리프트가 없다.

| 노트북 | 내용 | 대응 스크립트 |
|---|---|---|
| `00_overview_and_results.ipynb` | 프로젝트 지도 + 전 모델 성능표·플롯 | (grand 통합표 로드) |
| `01_data_and_evaluation.ipynb` | KP20k·PRMU·**단일 평가기** 워크스루 | `src/data.py`, `src/metrics.py` |
| `02_baselines_and_hybrid_pipeline.ipynb` | B0~B4, P1, P2 (GERD 융합+MMR) | `scripts/run_experiments.py` |
| `03_rerankers_pairwise_and_scibert.ipynb` | P5/P6 pairwise, **P7 HybridRanker** | `scripts/pairwise_reranker.py`, `scripts/scibert_hybrid_ranker.py` |
| `04_encoder_tournament_and_ensembles.ipynb` | P8~P15 토너먼트, P13/P16 앙상블 | `scripts/encoder_tournament.py`, `scripts/team_spec_ensemble.py` |
| `05_ensemble_reference_comparison.ipynb` | **채점표(앙상블) 기준** P7 vs KeyBART 랭킹 재현 (nDCG/RBO/Wilcoxon) | 팀 비교 시트 → [`ENSEMBLE_REFERENCE_COMPARISON.md`](../docs/ENSEMBLE_REFERENCE_COMPARISON.md) |

> **5개 노트북 모두 클론 후 바로 실행된다.** 대부분 `results/`의 CSV를 로드하는 방식이라 GPU·재학습 불필요.
> `01`의 KP20k 다운로드 셀만 네트워크가 필요하며(오프라인이면 자동 skip), `05`는 팀 원본 시트가 없으면
> 저장소에 포함된 요약 결과를 표시한다(원본 시트를 `data/keyphrase_ranking_raw.csv`에 두면 전체 재계산).

## 실행 방법

```bash
pip install -r ../requirements.txt
jupyter lab          # 또는 VS Code 에서 열기
```

- 대부분의 셀은 `results/`의 CSV·JSON을 pandas로 읽어 **GPU 없이 즉시 실행**된다.
- `01`의 데이터 로드 셀만 KP20k 다운로드가 필요하며, 오프라인이면 자동으로 건너뛴다.
- 모델을 **직접 학습·재현**하려면 노트북이 아니라 위 표의 스크립트를 실행한다 (루트 [`README.md`](../README.md) 재현 명령 참조).
