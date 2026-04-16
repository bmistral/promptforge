"""
Tests unitaires pour core/cost_tracker.py.

Couvre :
- _resolve_prices  : correspondance exacte, partielle, fallback
- CostTracker.track            : calcul de coût, accumulation, log de session
- CostTracker.track_from_usage : lecture des attributs via getattr, gestion des None
- CostTracker.alert_if_over_budget : déclenchement au-dessus du seuil 80%, silence dessous,
                                     absence d'alerte sans budget configuré
- CostTracker.summary          : clés attendues, valeurs initiales, cache_hit_pct
"""

import pytest
from unittest.mock import MagicMock, patch

from core.cost_tracker import CostTracker, _resolve_prices


class TestResolvePrices:
    def test_resolve_prices_haiku_matches_exactly(self):
        p_in, p_out, p_cr, p_cw = _resolve_prices("claude-haiku-4-5-20251001")
        assert p_in == 1.00
        assert p_out == 5.00
        assert p_cr == 0.10
        assert p_cw == 1.25

    def test_resolve_prices_sonnet_matches_partial_key(self):
        p_in, p_out, p_cr, p_cw = _resolve_prices("claude-sonnet-4-6-20251022")
        assert p_in == 3.00
        assert p_out == 15.00

    def test_resolve_prices_opus_matches_partial_key(self):
        p_in, p_out, _, _ = _resolve_prices("claude-opus-4-6")
        assert p_in == 5.00
        assert p_out == 25.00

    def test_resolve_prices_unknown_model_falls_back_to_sonnet_rates(self):
        p_in, p_out, p_cr, p_cw = _resolve_prices("modele-inconnu-xyz")
        assert p_in == 3.00
        assert p_out == 15.00
        assert p_cr == 0.30
        assert p_cw == 3.75


class TestCostTrackerTrack:
    def test_track_returns_correct_cost_for_input_tokens(self):
        tracker = CostTracker()
        # haiku: 1.00 $ / million tokens d'entrée
        cost = tracker.track("claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(1.00)

    def test_track_returns_correct_cost_for_output_tokens(self):
        tracker = CostTracker()
        # haiku: 5.00 $ / million tokens de sortie
        cost = tracker.track("claude-haiku-4-5-20251001", input_tokens=0, output_tokens=1_000_000)
        assert cost == pytest.approx(5.00)

    def test_track_calculates_cache_write_cost_correctly(self):
        tracker = CostTracker()
        # haiku cache_write: 1.25 $ / million tokens
        cost = tracker.track("claude-haiku-4-5-20251001", input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000)
        assert cost == pytest.approx(1.25)

    def test_track_calculates_cache_read_cost_correctly(self):
        tracker = CostTracker()
        # haiku cache_read: 0.10 $ / million tokens
        cost = tracker.track("claude-haiku-4-5-20251001", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
        assert cost == pytest.approx(0.10)

    def test_track_accumulates_total_cost_across_multiple_calls(self):
        tracker = CostTracker()
        tracker.track("claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=0)
        tracker.track("claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=0)
        assert tracker._total_cost == pytest.approx(2.00)

    def test_track_increments_api_calls_counter(self):
        tracker = CostTracker()
        tracker.track("claude-haiku-4-5-20251001", input_tokens=100, output_tokens=50)
        tracker.track("claude-haiku-4-5-20251001", input_tokens=100, output_tokens=50)
        assert tracker._api_calls == 2

    def test_track_accumulates_all_token_types(self):
        tracker = CostTracker()
        tracker.track(
            "claude-haiku-4-5-20251001",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
        )
        assert tracker._total_input == 100
        assert tracker._total_output == 50
        assert tracker._total_cache_read == 10
        assert tracker._total_cache_write == 5

    def test_track_appends_entry_to_session_log(self):
        tracker = CostTracker()
        tracker.track("claude-haiku-4-5-20251001", input_tokens=100, output_tokens=50)
        assert len(tracker.session_log) == 1
        entry = tracker.session_log[0]
        assert entry["model"] == "claude-haiku-4-5-20251001"
        assert entry["input_tokens"] == 100
        assert entry["output_tokens"] == 50

    def test_track_fresh_tracker_starts_at_zero(self):
        tracker = CostTracker()
        assert tracker._total_cost == 0.0
        assert tracker._api_calls == 0
        assert tracker._total_input == 0


class TestTrackFromUsage:
    def test_track_from_usage_reads_standard_usage_attributes(self):
        tracker = CostTracker()
        usage = MagicMock()
        usage.input_tokens = 500
        usage.output_tokens = 200
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0

        tracker.track_from_usage("claude-haiku-4-5-20251001", usage)

        assert tracker._total_input == 500
        assert tracker._total_output == 200

    def test_track_from_usage_handles_none_attributes_as_zero(self):
        tracker = CostTracker()
        usage = MagicMock()
        usage.input_tokens = None
        usage.output_tokens = 100
        usage.cache_read_input_tokens = None
        usage.cache_creation_input_tokens = None

        tracker.track_from_usage("claude-haiku-4-5-20251001", usage)

        assert tracker._total_input == 0
        assert tracker._total_output == 100

    def test_track_from_usage_maps_cache_creation_to_write(self):
        tracker = CostTracker()
        usage = MagicMock()
        usage.input_tokens = 0
        usage.output_tokens = 0
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 1_000

        tracker.track_from_usage("claude-haiku-4-5-20251001", usage)

        assert tracker._total_cache_write == 1_000


class TestAlertIfOverBudget:
    def test_alert_if_over_budget_triggers_when_threshold_exceeded(self):
        # mensuel=$30 → journalier=$1.00 → seuil 80%=$0.80
        tracker = CostTracker(monthly_budget_usd=30.0)
        tracker._total_cost = 0.85  # > 0.80 → alerte attendue

        with patch("core.cost_tracker._console") as mock_console:
            tracker.alert_if_over_budget()
            mock_console.print.assert_called_once()
            message = mock_console.print.call_args[0][0]
            assert "Alerte budget" in message

    def test_alert_if_over_budget_no_alert_when_below_threshold(self):
        # mensuel=$30 → journalier=$1.00 → seuil 80%=$0.80
        tracker = CostTracker(monthly_budget_usd=30.0)
        tracker._total_cost = 0.50  # < 0.80 → pas d'alerte

        with patch("core.cost_tracker._console") as mock_console:
            tracker.alert_if_over_budget()
            mock_console.print.assert_not_called()

    def test_alert_if_over_budget_skips_when_no_budget_configured(self):
        tracker = CostTracker()  # monthly_budget_usd=None
        tracker._total_cost = 9999.0  # coût arbitrairement élevé

        with patch("core.cost_tracker._console") as mock_console:
            tracker.alert_if_over_budget()
            mock_console.print.assert_not_called()

    def test_alert_if_over_budget_triggers_exactly_at_threshold(self):
        # seuil exact : 80% du budget journalier
        tracker = CostTracker(monthly_budget_usd=30.0)
        tracker._total_cost = 0.80  # exactement au seuil

        with patch("core.cost_tracker._console") as mock_console:
            tracker.alert_if_over_budget()
            mock_console.print.assert_called_once()


class TestSummary:
    def test_summary_returns_all_expected_keys(self):
        tracker = CostTracker()
        s = tracker.summary()
        expected_keys = {
            "total_cost_usd",
            "api_calls",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cache_hit_pct",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_empty_tracker_returns_all_zeros(self):
        tracker = CostTracker()
        s = tracker.summary()
        assert s["total_cost_usd"] == 0.0
        assert s["api_calls"] == 0
        assert s["input_tokens"] == 0
        assert s["output_tokens"] == 0
        assert s["cache_hit_pct"] == 0.0

    def test_summary_cache_hit_pct_calculated_correctly(self):
        tracker = CostTracker()
        # 900 input + 100 cache_read = 1000 tokens facturés
        # cache_hit_pct = 100 / 1000 * 100 = 10.0 %
        tracker.track(
            "claude-haiku-4-5-20251001",
            input_tokens=900,
            output_tokens=0,
            cache_read_tokens=100,
        )
        s = tracker.summary()
        assert s["cache_hit_pct"] == 10.0

    def test_summary_reflects_accumulated_token_counts(self):
        tracker = CostTracker()
        tracker.track("claude-haiku-4-5-20251001", input_tokens=100, output_tokens=50)
        tracker.track("claude-haiku-4-5-20251001", input_tokens=200, output_tokens=100)
        s = tracker.summary()
        assert s["input_tokens"] == 300
        assert s["output_tokens"] == 150
        assert s["api_calls"] == 2

    def test_summary_total_cost_rounded_to_six_decimals(self):
        tracker = CostTracker()
        tracker.track("claude-haiku-4-5-20251001", input_tokens=1, output_tokens=1)
        s = tracker.summary()
        # Vérifie que la valeur est bien un float arrondi (pas de longue mantisse)
        assert isinstance(s["total_cost_usd"], float)
        assert len(str(s["total_cost_usd"]).split(".")[-1]) <= 6
