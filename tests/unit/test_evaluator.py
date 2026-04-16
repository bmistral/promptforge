"""
Tests unitaires pour core/evaluator.py.

Couvre :
- _parse_score : JSON valide, score manquant, feedback manquant, score hors range (clamp),
                 score entier → float, JSON invalide, JSON dans des fences markdown
- evaluate     : mode simulation — mock de l'API, vérification du score retourné
                 et de l'accumulation des coûts

Aucun appel réel à l'API Anthropic : client mocké via unittest.mock.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.evaluator import PromptEvaluator


@pytest.fixture
def evaluator():
    """Crée un PromptEvaluator avec le client Anthropic mocké.

    Le patch est actif pendant la construction pour éviter toute validation
    de clé API. ev.client reste un MagicMock utilisable dans les tests.
    """
    with patch("core.evaluator.anthropic.AsyncAnthropic"):
        ev = PromptEvaluator(model="claude-haiku-4-5-20251001", execute=False)
    return ev


def _make_mock_response(json_text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Construit un objet response Anthropic mocké pour un texte JSON donné."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json_text)]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.usage.cache_creation_input_tokens = 0
    return mock_response


class TestParseScore:
    def test_parse_score_valid_json_returns_score_and_feedback(self, evaluator):
        score, feedback = evaluator._parse_score('{"score": 7.5, "feedback": "bon prompt"}')
        assert score == 7.5
        assert feedback == "bon prompt"

    def test_parse_score_missing_score_key_defaults_to_five(self, evaluator):
        score, feedback = evaluator._parse_score('{"feedback": "feedback sans score"}')
        assert score == 5.0
        assert feedback == "feedback sans score"

    def test_parse_score_missing_feedback_key_returns_empty_string(self, evaluator):
        score, feedback = evaluator._parse_score('{"score": 8.0}')
        assert score == 8.0
        assert feedback == ""

    def test_parse_score_score_above_max_clamps_to_ten(self, evaluator):
        score, _ = evaluator._parse_score('{"score": 15.0, "feedback": "trop haut"}')
        assert score == 10.0

    def test_parse_score_score_below_min_clamps_to_zero(self, evaluator):
        score, _ = evaluator._parse_score('{"score": -3.0, "feedback": "négatif"}')
        assert score == 0.0

    def test_parse_score_score_at_boundary_ten_not_clamped(self, evaluator):
        score, _ = evaluator._parse_score('{"score": 10.0, "feedback": "parfait"}')
        assert score == 10.0

    def test_parse_score_score_at_boundary_zero_not_clamped(self, evaluator):
        score, _ = evaluator._parse_score('{"score": 0.0, "feedback": "nul"}')
        assert score == 0.0

    def test_parse_score_integer_score_returned_as_float(self, evaluator):
        score, _ = evaluator._parse_score('{"score": 8, "feedback": "entier"}')
        assert isinstance(score, float)
        assert score == 8.0

    def test_parse_score_invalid_json_returns_default_score_five(self, evaluator):
        score, feedback = evaluator._parse_score("ceci n'est pas du JSON")
        assert score == 5.0
        assert feedback == "Erreur d'évaluation"

    def test_parse_score_empty_string_returns_default_score_five(self, evaluator):
        score, feedback = evaluator._parse_score("")
        assert score == 5.0
        assert feedback == "Erreur d'évaluation"

    def test_parse_score_json_in_markdown_fences_parsed_correctly(self, evaluator):
        raw = '```json\n{"score": 9.0, "feedback": "excellent"}\n```'
        score, feedback = evaluator._parse_score(raw)
        assert score == 9.0
        assert feedback == "excellent"

    def test_parse_score_json_with_surrounding_text_parsed_correctly(self, evaluator):
        raw = 'Voici mon évaluation : {"score": 6.5, "feedback": "correct"} merci.'
        score, feedback = evaluator._parse_score(raw)
        assert score == 6.5
        assert feedback == "correct"


class TestEvaluateSimulated:
    async def test_evaluate_simulated_returns_parsed_score_from_api_response(self, evaluator):
        evaluator.client.messages.create = AsyncMock(
            return_value=_make_mock_response('{"score": 7.5, "feedback": "bien structuré"}')
        )

        score, feedback = await evaluator.evaluate(
            prompt="Résume ce texte en une phrase.",
            examples=[{"input": "texte long", "expected_output": "résumé court"}],
            task_description="Résumé",
        )

        assert score == 7.5
        assert feedback == "bien structuré"

    async def test_evaluate_simulated_calls_api_exactly_once(self, evaluator):
        evaluator.client.messages.create = AsyncMock(
            return_value=_make_mock_response('{"score": 6.0, "feedback": "ok"}')
        )

        await evaluator.evaluate(
            prompt="Mon prompt",
            examples=[{"input": "x", "expected_output": "y"}],
        )

        evaluator.client.messages.create.assert_called_once()

    async def test_evaluate_simulated_tracks_api_cost_in_tracker(self, evaluator):
        evaluator.client.messages.create = AsyncMock(
            return_value=_make_mock_response(
                '{"score": 6.0, "feedback": "ok"}',
                input_tokens=200,
                output_tokens=80,
            )
        )

        await evaluator.evaluate(
            prompt="Mon prompt",
            examples=[{"input": "x", "expected_output": "y"}],
        )

        summary = evaluator.tracker.summary()
        assert summary["api_calls"] == 1
        assert summary["input_tokens"] == 200
        assert summary["output_tokens"] == 80

    async def test_evaluate_simulated_clamps_score_above_ten(self, evaluator):
        evaluator.client.messages.create = AsyncMock(
            return_value=_make_mock_response('{"score": 12.0, "feedback": "trop haut"}')
        )

        score, _ = await evaluator.evaluate(
            prompt="prompt",
            examples=[{"input": "x", "expected_output": "y"}],
        )

        assert score == 10.0

    async def test_evaluate_simulated_returns_default_on_malformed_json(self, evaluator):
        evaluator.client.messages.create = AsyncMock(
            return_value=_make_mock_response("réponse non-JSON inattendue")
        )

        score, feedback = await evaluator.evaluate(
            prompt="prompt",
            examples=[{"input": "x", "expected_output": "y"}],
        )

        assert score == 5.0
        assert feedback == "Erreur d'évaluation"

    async def test_evaluate_without_task_description_still_returns_score(self, evaluator):
        evaluator.client.messages.create = AsyncMock(
            return_value=_make_mock_response('{"score": 5.5, "feedback": "neutre"}')
        )

        score, feedback = await evaluator.evaluate(
            prompt="prompt sans contexte",
            examples=[{"input": "x", "expected_output": "y"}],
            task_description=None,
        )

        assert score == 5.5
