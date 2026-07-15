"""KP20k Keyphrase Prediction 프로젝트 소스 패키지.

모듈 구성 (마스터 플랜 10절):
- utils:         시드/디바이스/IO/실험 로깅
- data:          KP20k 로딩, source/target 구성, 데이터 감사
- preprocessing: 정규화·stemming·PRMU 분류·생성 출력 파싱 (단일 규칙)
- metrics:       F1@K/F1@M/MAP/nDCG/PRMU recall/다양성/의미 기반 평가
- extraction:    TF-IDF, KeyBERT 추출 베이스라인
- generation:    Seq2Seq(BART/KeyBART) 학습·디코딩
- reranking:     score fusion, Cross-Encoder
- diversity:     MMR
- pipeline:      GERD 하이브리드 파이프라인
"""

__version__ = "0.1.0"
