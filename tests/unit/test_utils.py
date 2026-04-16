"""
Tests unitaires pour core/utils.py.

Couvre :
- parse_json_response : JSON propre, fences markdown, JSON embarqué, JSON malformé
- deduplicate         : suppression de doublons, préservation d'ordre, cas limites
"""

import json
import pytest

from core.utils import deduplicate, parse_json_response


class TestParseJsonResponse:
    def test_parse_json_response_valid_json_returns_dict(self):
        raw = '{"key": "value", "number": 42}'
        result = parse_json_response(raw)
        assert result == {"key": "value", "number": 42}

    def test_parse_json_response_strips_leading_trailing_whitespace(self):
        raw = '  \n{"score": 7.5}\n  '
        result = parse_json_response(raw)
        assert result == {"score": 7.5}

    def test_parse_json_response_json_in_markdown_fences_is_parsed(self):
        raw = '```json\n{"score": 8.0, "feedback": "bien"}\n```'
        result = parse_json_response(raw)
        assert result["score"] == 8.0
        assert result["feedback"] == "bien"

    def test_parse_json_response_json_with_surrounding_text_is_extracted(self):
        raw = 'Voici le résultat : {"score": 6.5, "feedback": "ok"} (évaluation terminée)'
        result = parse_json_response(raw)
        assert result["score"] == 6.5

    def test_parse_json_response_nested_objects_parsed_correctly(self):
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_json_response(raw)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_parse_json_response_extracts_between_first_and_last_brace(self):
        """Le parsing doit utiliser la première { et la dernière } pour tolérer
        du texte autour même si le contenu contient des accolades."""
        raw = 'Avant {"score": 9.0, "feedback": "test {accolades}"} après'
        result = parse_json_response(raw)
        assert result["score"] == 9.0

    def test_parse_json_response_invalid_json_raises_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            parse_json_response("ceci n'est pas du JSON valide")

    def test_parse_json_response_plain_string_no_braces_raises_error(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_json_response("score: 5")

    def test_parse_json_response_empty_string_raises_error(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_json_response("")

    def test_parse_json_response_malformed_json_raises_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            parse_json_response('{"score": 7.5, "feedback":}')


class TestDeduplicate:
    def test_deduplicate_removes_duplicate_strings(self):
        result = deduplicate(["a", "b", "a", "c"])
        assert result == ["a", "b", "c"]

    def test_deduplicate_preserves_original_order(self):
        result = deduplicate(["c", "a", "b", "a", "c"])
        assert result == ["c", "a", "b"]

    def test_deduplicate_empty_list_returns_empty_list(self):
        assert deduplicate([]) == []

    def test_deduplicate_list_without_duplicates_is_unchanged(self):
        items = ["x", "y", "z"]
        result = deduplicate(items)
        assert result == ["x", "y", "z"]

    def test_deduplicate_all_identical_elements_returns_single_element(self):
        result = deduplicate(["a", "a", "a"])
        assert result == ["a"]

    def test_deduplicate_works_with_integers(self):
        result = deduplicate([1, 2, 1, 3, 2])
        assert result == [1, 2, 3]

    def test_deduplicate_returns_new_list_not_same_reference(self):
        original = ["a", "b", "a"]
        result = deduplicate(original)
        assert result is not original

    def test_deduplicate_single_element_list_unchanged(self):
        assert deduplicate(["unique"]) == ["unique"]
