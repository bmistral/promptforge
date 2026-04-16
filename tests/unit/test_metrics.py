"""
Tests unitaires pour core/metrics.py.

Couvre :
- exact_match       : correspondance exacte, casse, espaces, non-correspondance
- regex_validity    : regex valide + passant, invalide, valide mais échouant
- rouge_l           : texte identique, vide, partiellement similaire
- f1_token          : overlap parfait, nul, partiel, ordre différent
- length_ratio_score: ratio dans la plage, en dessous, au dessus, référence vide
- semantic_similarity : mock SentenceTransformer, lazy loading confirmé
- code_execution    : fonction correcte, erreur syntaxe, timeout boucle infinie
- compute_deterministic_score : dataset avec métriques, sans métriques, inconnu

Aucun appel réseau : SentenceTransformer est toujours mocké.
"""

import importlib
import time
from unittest.mock import MagicMock, patch

import pytest

from core.metrics import (
    compute_deterministic_score,
    code_execution,
    exact_match,
    f1_token,
    length_ratio_score,
    regex_validity,
    rouge_l,
    semantic_similarity,
)


# ─── exact_match ──────────────────────────────────────────────────────────────


class TestExactMatch:
    def test_identical_strings_return_one(self):
        assert exact_match("positif", "positif") == 1.0

    def test_case_insensitive_match_returns_one(self):
        assert exact_match("Positif", "positif") == 1.0

    def test_strips_leading_trailing_whitespace(self):
        assert exact_match("  positif  ", "positif") == 1.0

    def test_different_values_return_zero(self):
        assert exact_match("positif", "négatif") == 0.0

    def test_empty_strings_match(self):
        assert exact_match("", "") == 1.0

    def test_empty_vs_nonempty_returns_zero(self):
        assert exact_match("", "something") == 0.0


# ─── regex_validity ───────────────────────────────────────────────────────────


class TestRegexValidity:
    def test_valid_regex_all_cases_pass_returns_one(self):
        score = regex_validity(
            r"^\d{5}$",
            [
                {"input": "75001", "should_match": True},
                {"input": "abc", "should_match": False},
            ],
        )
        assert score == 1.0

    def test_invalid_regex_returns_zero(self):
        score = regex_validity(
            "[invalid",
            [{"input": "test", "should_match": True}],
        )
        assert score == 0.0

    def test_valid_regex_but_failing_cases_returns_partial(self):
        # r"^\d{3}$" ne matche pas "12345" (5 chiffres) mais ne matche pas "abc"
        score = regex_validity(
            r"^\d{3}$",
            [
                {"input": "12345", "should_match": True},   # échoue
                {"input": "abc", "should_match": False},    # réussit
            ],
        )
        assert score == 0.5

    def test_valid_regex_all_cases_fail_returns_zero(self):
        score = regex_validity(
            r"^\d{3}$",
            [
                {"input": "12345", "should_match": True},
                {"input": "123", "should_match": False},
            ],
        )
        assert score == 0.0

    def test_valid_regex_no_test_cases_returns_one(self):
        assert regex_validity(r"^\d+$", []) == 1.0


# ─── rouge_l ──────────────────────────────────────────────────────────────────


class TestRougeL:
    def test_identical_texts_return_near_one(self):
        score = rouge_l("Le chat est sur le tapis.", "Le chat est sur le tapis.")
        assert score >= 0.99

    def test_empty_generated_returns_zero(self):
        assert rouge_l("", "Le chat est sur le tapis.") == 0.0

    def test_empty_reference_returns_zero(self):
        assert rouge_l("Le chat est sur le tapis.", "") == 0.0

    def test_partial_overlap_returns_intermediate_score(self):
        score = rouge_l(
            "Le chat est sur le tapis rouge.",
            "Le chien est sous la table bleue.",
        )
        assert 0.0 < score < 1.0

    def test_completely_different_texts_return_low_score(self):
        score = rouge_l("soleil lune étoiles", "voiture autoroute péage")
        assert score < 0.2


# ─── f1_token ─────────────────────────────────────────────────────────────────


class TestF1Token:
    def test_perfect_overlap_returns_one(self):
        assert f1_token("chat chien oiseau", "chat chien oiseau") == 1.0

    def test_no_overlap_returns_zero(self):
        assert f1_token("chat chien", "rouge bleu vert") == 0.0

    def test_partial_overlap_returns_intermediate(self):
        score = f1_token("chat chien oiseau lapin", "chat chien renard")
        assert 0.0 < score < 1.0

    def test_order_does_not_matter(self):
        s1 = f1_token("a b c", "c b a")
        s2 = f1_token("a b c", "a b c")
        assert s1 == s2 == 1.0

    def test_case_insensitive(self):
        assert f1_token("Chat Chien", "chat chien") == 1.0

    def test_both_empty_returns_one(self):
        assert f1_token("", "") == 1.0

    def test_one_empty_returns_zero(self):
        assert f1_token("", "chat") == 0.0


# ─── length_ratio_score ───────────────────────────────────────────────────────


class TestLengthRatioScore:
    def test_ratio_in_range_returns_one(self):
        # 5 mots / 3 mots = 1.67 → dans [1.2, 3.0]
        assert length_ratio_score("a b c d e", "a b c") == 1.0

    def test_ratio_at_lower_bound_returns_one(self):
        # 6 mots / 5 mots = 1.2 → exactement à la borne
        assert length_ratio_score("a b c d e f", "a b c d e") == 1.0

    def test_ratio_at_upper_bound_returns_one(self):
        # 9 mots / 3 mots = 3.0 → exactement à la borne
        assert length_ratio_score("a b c d e f g h i", "a b c") == 1.0

    def test_ratio_below_range_returns_half(self):
        # 2 mots / 5 mots = 0.4 → sous 1.2
        assert length_ratio_score("a b", "a b c d e") == 0.5

    def test_ratio_above_range_returns_half(self):
        # 10 mots / 2 mots = 5.0 → au dessus de 3.0
        assert length_ratio_score("a b c d e f g h i j", "a b") == 0.5

    def test_empty_reference_returns_half(self):
        assert length_ratio_score("quelques mots", "") == 0.5


# ─── semantic_similarity ─────────────────────────────────────────────────────


class TestSemanticSimilarity:
    @patch("core.metrics._sentence_model", new=None)
    @patch("core.metrics.SentenceTransformer")
    def test_semantic_similarity_uses_mock_model(self, mock_st):
        mock_instance = MagicMock()
        mock_instance.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_st.return_value = mock_instance
        score = semantic_similarity("texte A", "texte B")
        mock_st.assert_called_once_with("paraphrase-multilingual-MiniLM-L12-v2")

    @patch("core.metrics._sentence_model", new=None)
    @patch("core.metrics.SentenceTransformer")
    def test_semantic_similarity_model_loaded_once(self, mock_st):
        """Le modèle n'est chargé qu'une seule fois malgré plusieurs appels."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        mock_st.return_value = mock_instance
        semantic_similarity("texte A", "texte B")
        semantic_similarity("texte C", "texte D")
        mock_st.assert_called_once()

    def test_semantic_similarity_does_not_download_model_at_import(self):
        """
        Vérifie que l'import de core.metrics ne déclenche pas
        le chargement du modèle Sentence Transformers.
        Le modèle ne doit être chargé qu'au premier appel à semantic_similarity().
        """
        import core.metrics as m

        importlib.reload(m)
        assert m._sentence_model is None  # lazy loading confirmé

    @patch("core.metrics.SentenceTransformer", new=None)
    def test_semantic_similarity_returns_zero_if_lib_unavailable(self):
        """Retourne 0.0 si sentence_transformers n'est pas installé."""
        score = semantic_similarity("texte A", "texte B")
        assert score == 0.0

    @patch("core.metrics._sentence_model", new=None)
    @patch("core.metrics.SentenceTransformer")
    def test_semantic_similarity_identical_embeddings_return_one(self, mock_st):
        import numpy as np

        emb = np.array([0.5, 0.5, 0.5])
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([emb, emb])
        mock_st.return_value = mock_instance
        score = semantic_similarity("même texte", "même texte")
        assert abs(score - 1.0) < 1e-6


# ─── code_execution ───────────────────────────────────────────────────────────


class TestCodeExecution:
    def test_correct_function_returns_one(self):
        code = "def add(a, b):\n    return a + b"
        test_cases = [{"input": [1, 2], "expected_output": 3}]
        assert code_execution(code, test_cases) == 1.0

    def test_multiple_test_cases_partial_pass(self):
        code = "def identity(x):\n    return x"
        test_cases = [
            {"input": 42, "expected_output": 42},
            {"input": "ok", "expected_output": "not ok"},
        ]
        assert code_execution(code, test_cases) == 0.5

    def test_syntax_error_returns_zero(self):
        code = "def bad_func(x\n    return x"
        score = code_execution(code, [{"input": 1, "expected_output": 1}])
        assert score == 0.0

    def test_no_function_defined_returns_zero(self):
        code = "x = 42"
        score = code_execution(code, [{"input": None, "expected_output": 42}])
        assert score == 0.0

    def test_empty_test_cases_returns_zero(self):
        assert code_execution("def f(x): return x", []) == 0.0

    def test_empty_code_returns_zero(self):
        assert code_execution("", [{"input": 1, "expected_output": 1}]) == 0.0

    def test_code_execution_timeout_kills_infinite_loop(self):
        infinite_loop_code = "while True: pass"
        start = time.time()
        score = code_execution(
            infinite_loop_code, [{"input": None, "expected_output": None}]
        )
        elapsed = time.time() - start
        assert score == 0.0
        assert elapsed < 8.0  # timeout 5 s + marge de 3 s max


# ─── compute_deterministic_score ─────────────────────────────────────────────


class TestComputeDeterministicScore:
    def test_known_dataset_exact_match_hit(self):
        score = compute_deterministic_score("sentiment", "positif", "positif")
        assert score == 1.0

    def test_known_dataset_exact_match_miss(self):
        score = compute_deterministic_score("sentiment", "négatif", "positif")
        assert score == 0.0

    def test_dataset_without_metrics_returns_none(self):
        assert compute_deterministic_score("address", "anything", "reference") is None

    def test_empty_metrics_dataset_returns_none(self):
        assert compute_deterministic_score("security", "anything", "reference") is None

    def test_unknown_dataset_returns_none(self):
        assert compute_deterministic_score("unknown_dataset", "text", "ref") is None

    def test_code_dataset_without_test_cases_returns_none(self):
        # code metric needs test_cases; without them the weighted sum is empty → None
        score = compute_deterministic_score("code", "def f(x): return x", "def f(x): return x")
        assert score is None

    def test_code_dataset_with_test_cases_returns_float(self):
        code = "def double(x):\n    return x * 2"
        test_cases = [{"input": 3, "expected_output": 6}]
        score = compute_deterministic_score("code", code, "", test_cases=test_cases)
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_summarization_dataset_returns_float_in_range(self):
        score = compute_deterministic_score(
            "summarization",
            "Le chat dort sur le tapis.",
            "Le chat dort sur le tapis.",
        )
        assert score is not None
        assert score >= 0.99

    def test_feedback_dataset_weighted_average(self):
        """feedback utilise semantic_similarity (0.7) + length_ratio (0.3)."""
        with patch("core.metrics._sentence_model", new=None), patch(
            "core.metrics.SentenceTransformer"
        ) as mock_st:
            import numpy as np

            emb = np.array([1.0, 0.0, 0.0])
            mock_instance = MagicMock()
            mock_instance.encode.return_value = np.array([emb, emb])
            mock_st.return_value = mock_instance

            # generated: 5 mots, reference: 3 mots → ratio 1.67 → length_ratio=1.0
            generated = "a b c d e"
            reference = "a b c"
            score = compute_deterministic_score("feedback", generated, reference)
            assert score is not None
            # semantic_similarity=1.0, length_ratio=1.0 → score=1.0
            assert abs(score - 1.0) < 1e-6
