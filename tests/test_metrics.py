"""평가 함수 단위 테스트 (마스터 플랜 9절 + Notebook 03 요구 사례 전체).

실행: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diversity import mmr_select
from src.metrics import (
    average_precision_at_k,
    candidate_recall,
    evaluate_corpus,
    exact_duplicate_ratio,
    f1_at_k,
    f1_at_m,
    ndcg_at_k,
    precision_recall_f1_at_k,
    semantic_match_f1,
    stem_duplicate_ratio,
)
from src.preprocessing import (
    classify_prmu,
    clean_candidates,
    is_present,
    normalize_phrase,
    parse_generated_sequence,
    split_present_absent,
    unique_normalized,
)


# ---------------------------------------------------------------------------
# normalize / dedupe
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercase_and_whitespace(self):
        assert normalize_phrase("  Neural   Networks ") == normalize_phrase("neural networks")

    def test_plural_singular_same(self):
        assert normalize_phrase("neural networks") == normalize_phrase("neural network")

    def test_hyphen_kept_as_one_token(self):
        # 공식 데이터 카드: graph-based는 쪼개지 않는다
        assert normalize_phrase("graph-based") != normalize_phrase("graph based")

    def test_empty(self):
        assert normalize_phrase("") == ""
        assert normalize_phrase("   ") == ""

    def test_unique_normalized_order_preserved(self):
        result = unique_normalized(["Neural Network", "neural networks", "classification"])
        assert len(result) == 2
        assert result[0] == normalize_phrase("neural network")


# ---------------------------------------------------------------------------
# F1@K (플랜 9절 필수 테스트 3종 포함)
# ---------------------------------------------------------------------------

class TestF1AtK:
    def test_perfect_match(self):
        result = f1_at_k(["neural network"], ["neural networks"], 1)
        assert result["f1"] == 1.0

    def test_empty_prediction(self):
        result = f1_at_k([], ["neural network"], 5)
        assert result["f1"] == 0.0

    def test_duplicate_prediction(self):
        result = f1_at_k(
            ["neural network", "neural networks", "classification"],
            ["neural network", "classification"],
            2,
        )
        assert result["tp"] == 2

    def test_complete_mismatch(self):
        result = f1_at_k(["apple pie"], ["neural network"], 5)
        assert result["f1"] == 0.0

    def test_case_difference(self):
        result = f1_at_k(["Neural Network"], ["neural network"], 1)
        assert result["f1"] == 1.0

    def test_fewer_predictions_than_k_penalized(self):
        # 예측 1개(정답), K=5 → precision = 1/5 (분모 K 유지)
        result = f1_at_k(["neural network"], ["neural network"], 5)
        assert result["precision"] == pytest.approx(0.2)
        assert result["recall"] == 1.0

    def test_empty_gold_no_crash(self):
        result = f1_at_k(["anything"], [], 5)
        assert result["f1"] == 0.0

    def test_oracle_gold_as_prediction_is_perfect(self):
        gold = ["data aggregation", "sensor networks", "real-time traffic"]
        result = f1_at_m(gold, gold)
        assert result["f1"] == 1.0

    def test_hyphenated_phrase(self):
        assert f1_at_k(["graph-based learning"], ["graph-based learning"], 1)["f1"] == 1.0


class TestRankingMetrics:
    def test_map_prefers_early_hits(self):
        gold = ["a b", "c d"]
        early = average_precision_at_k(["a b", "x", "c d"], gold, 5)
        late = average_precision_at_k(["x", "a b", "c d"], gold, 5)
        assert early > late

    def test_ndcg_perfect_ranking(self):
        gold = ["a b", "c d"]
        assert ndcg_at_k(["a b", "c d"], gold, 2) == pytest.approx(1.0)

    def test_ndcg_empty_gold(self):
        assert ndcg_at_k(["a"], [], 5) == 0.0


# ---------------------------------------------------------------------------
# Present/Absent, PRMU
# ---------------------------------------------------------------------------

DOC = "deep neural networks are used for image classification"


class TestPresentAbsent:
    def test_present_contiguous(self):
        assert is_present("image classification", DOC)
        assert classify_prmu("image classification", DOC) == "P"

    def test_present_stemmed_variant(self):
        assert is_present("neural network", DOC)  # networks → network stem 일치

    def test_reordered(self):
        assert classify_prmu("classification image", DOC) == "R"

    def test_mixed(self):
        assert classify_prmu("medical image classification", DOC) == "M"

    def test_unseen(self):
        assert classify_prmu("computer vision", DOC) == "U"

    def test_split(self):
        present, absent = split_present_absent(
            ["image classification", "computer vision"], DOC
        )
        assert present == ["image classification"]
        assert absent == ["computer vision"]


# ---------------------------------------------------------------------------
# 생성 출력 파싱·후보 정제
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_kp_sep(self):
        raw = "machine learning <kp_sep> neural network <kp_sep> classification"
        assert parse_generated_sequence(raw) == [
            "machine learning", "neural network", "classification",
        ]

    def test_parse_present_absent_tokens(self):
        raw = "<present> a b <kp_sep> c d <absent> e f"
        assert parse_generated_sequence(raw) == ["a b", "c d", "e f"]

    def test_parse_semicolon_fallback(self):
        assert parse_generated_sequence("a b; c d") == ["a b", "c d"]

    def test_clean_removes_unk_and_long(self):
        cands = ["good phrase", "has <unk> token", "one two three four five six seven"]
        out = clean_candidates(cands, max_words=5)
        assert out == ["good phrase"]

    def test_clean_dedupes_by_stem(self):
        out = clean_candidates(["neural network", "Neural Networks", "svm"])
        assert len(out) == 2


# ---------------------------------------------------------------------------
# 다양성·의미 지표
# ---------------------------------------------------------------------------

class TestDiversity:
    def test_stem_duplicate_ratio(self):
        preds = ["neural network", "neural networks", "svm"]
        assert stem_duplicate_ratio(preds) == pytest.approx(1 / 3)

    def test_exact_duplicate_ratio_zero(self):
        assert exact_duplicate_ratio(["a", "b", "c"]) == 0.0

    def test_mmr_first_pick_is_max_relevance(self):
        emb = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        order = mmr_select(emb, [0.9, 0.8, 0.5], top_k=2, lambda_=0.5)
        assert order[0] == 0
        # 두 번째는 중복(idx1)보다 다양한 idx2를 선택해야 한다
        assert order[1] == 2

    def test_mmr_lambda_one_is_pure_relevance(self):
        emb = np.eye(3)
        order = mmr_select(emb, [0.1, 0.9, 0.5], top_k=3, lambda_=1.0)
        assert order == [1, 2, 0]


class TestSemanticF1:
    def test_identical_embeddings_match(self):
        e = np.array([[1.0, 0.0], [0.0, 1.0]])
        r = semantic_match_f1(e, e, threshold=0.9)
        assert r["f1"] == 1.0

    def test_orthogonal_no_match(self):
        p = np.array([[1.0, 0.0]])
        g = np.array([[0.0, 1.0]])
        assert semantic_match_f1(p, g, threshold=0.5)["f1"] == 0.0

    def test_empty_pred(self):
        g = np.array([[1.0, 0.0]])
        assert semantic_match_f1(np.zeros((0, 2)), g)["f1"] == 0.0


# ---------------------------------------------------------------------------
# 코퍼스 평가·후보 recall
# ---------------------------------------------------------------------------

class TestCorpusEval:
    def test_macro_average(self):
        preds = [["a b"], ["x y"]]
        golds = [["a b"], ["c d"]]
        macro = evaluate_corpus(preds, golds, ks=(1,))
        assert macro["F1@1"] == pytest.approx(0.5)
        assert macro["num_docs"] == 2

    def test_present_absent_split_with_prmu(self):
        doc = "deep neural networks are used for image classification"
        macro = evaluate_corpus(
            [["image classification", "computer vision"]],
            [["image classification", "computer vision"]],
            all_prmu=[["P", "U"]],
            all_doc_texts=[doc],
            ks=(5,),
        )
        assert macro["recall_P"] == 1.0
        assert macro["recall_U"] == 1.0
        assert macro["present_F1@5"] > 0

    def test_candidate_recall(self):
        out = candidate_recall([["a b", "c d", "e f"]], [["c d", "zz"]], ks=(2, 3))
        assert out["cand_recall@2"] == pytest.approx(0.5)
        assert out["cand_recall@3"] == pytest.approx(0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
