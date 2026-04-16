"""
Suivi des coûts API en temps réel avec alertes budget.
"""

from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table

_console = Console(stderr=True)

# Prix par million de tokens (input, output, cache_read, cache_write) par modèle
_PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-haiku-4-5":  (1.00,  5.00,  0.10, 1.25),
    "claude-sonnet-4-6": (3.00, 15.00,  0.30, 3.75),
    "claude-opus-4-6":   (5.00, 25.00,  0.50, 6.25),
}
_FALLBACK_PRICES = (3.00, 15.00, 0.30, 3.75)  # sonnet


def _resolve_prices(model: str) -> tuple[float, float, float, float]:
    """
    Résout les prix pour un modèle donné en cherchant une correspondance partielle.

    Args:
        model: Identifiant du modèle.

    Returns:
        Tuple (input_price, output_price, cache_read_price, cache_write_price) par million de tokens.
    """
    for key, prices in _PRICES.items():
        if key in model:
            return prices
    return _FALLBACK_PRICES


class CostTracker:
    """
    Suit les coûts API par session avec alertes si le budget mensuel est dépassé.

    Usage:
        tracker = CostTracker(monthly_budget_usd=20.0)
        tracker.track(model, input_tokens, output_tokens, cache_read, cache_write)
        tracker.alert_if_over_budget()
        summary = tracker.summary()
    """

    def __init__(self, monthly_budget_usd: Optional[float] = None):
        """
        Initialise un suivi de coûts API.

        Args:
            monthly_budget_usd: Budget mensuel en USD (optionnel). Si défini, des alertes
                                sont affichées si le seuil de 80 % du budget journalier est atteint.
        """
        self.monthly_budget_usd = monthly_budget_usd
        self.session_log: list[dict] = []
        self._total_cost: float = 0.0
        self._total_input: int = 0
        self._total_output: int = 0
        self._total_cache_read: int = 0
        self._total_cache_write: int = 0
        self._api_calls: int = 0

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """
        Enregistre un appel API et retourne son coût estimé en USD.

        Args:
            model: Identifiant du modèle Claude utilisé.
            input_tokens: Tokens d'entrée facturés.
            output_tokens: Tokens de sortie.
            cache_read_tokens: Tokens lus depuis le cache (0.1× input_price).
            cache_write_tokens: Tokens écrits dans le cache (1.25× input_price).

        Returns:
            Coût estimé de cet appel en USD.
        """
        p_in, p_out, p_cr, p_cw = _resolve_prices(model)
        cost = (
            input_tokens * p_in / 1_000_000
            + output_tokens * p_out / 1_000_000
            + cache_read_tokens * p_cr / 1_000_000
            + cache_write_tokens * p_cw / 1_000_000
        )

        self._total_cost += cost
        self._total_input += input_tokens
        self._total_output += output_tokens
        self._total_cache_read += cache_read_tokens
        self._total_cache_write += cache_write_tokens
        self._api_calls += 1

        self.session_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": round(cost, 6),
        })

        return cost

    def track_from_usage(self, model: str, usage) -> float:
        """
        Raccourci pour enregistrer depuis un objet usage de réponse Anthropic SDK.

        Args:
            model: Identifiant du modèle utilisé.
            usage: Objet `response.usage` retourné par l'API.

        Returns:
            Coût estimé de cet appel en USD.
        """
        return self.track(
            model=model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    def alert_if_over_budget(self) -> None:
        """
        Affiche un avertissement rich si le coût de session dépasse 80 % du budget journalier.
        Le budget journalier est estimé à monthly_budget / 30.
        Ne fait rien si aucun budget n'est configuré.
        """
        if not self.monthly_budget_usd:
            return
        daily_budget = self.monthly_budget_usd / 30
        threshold = daily_budget * 0.80
        if self._total_cost >= threshold:
            pct = self._total_cost / daily_budget * 100
            _console.print(
                f"[bold yellow]⚠️  Alerte budget :[/bold yellow] "
                f"${self._total_cost:.4f} dépensés cette session "
                f"({pct:.0f}% du budget journalier de ${daily_budget:.2f})"
            )

    def summary(self) -> dict:
        """
        Retourne un dictionnaire récapitulatif de la session courante.

        Returns:
            Dict avec total_cost_usd, api_calls, input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens, cache_hit_pct.
        """
        total_billed_input = self._total_input + self._total_cache_write + self._total_cache_read
        cache_hit_pct = (
            round(self._total_cache_read / total_billed_input * 100, 1)
            if total_billed_input > 0
            else 0.0
        )
        return {
            "total_cost_usd": round(self._total_cost, 6),
            "api_calls": self._api_calls,
            "input_tokens": self._total_input,
            "output_tokens": self._total_output,
            "cache_read_tokens": self._total_cache_read,
            "cache_write_tokens": self._total_cache_write,
            "cache_hit_pct": cache_hit_pct,
        }

    def print_summary(self) -> None:
        """Affiche un tableau rich récapitulatif de la session."""
        s = self.summary()
        table = Table(title="Coûts API — session", show_header=True, header_style="bold cyan")
        table.add_column("Métrique", style="dim")
        table.add_column("Valeur", justify="right")

        table.add_row("Appels API", str(s["api_calls"]))
        table.add_row("Tokens entrée", f"{s['input_tokens']:,}")
        table.add_row("Tokens sortie", f"{s['output_tokens']:,}")
        table.add_row("Cache read", f"{s['cache_read_tokens']:,} ({s['cache_hit_pct']}%)")
        table.add_row("Cache write", f"{s['cache_write_tokens']:,}")
        table.add_row("[bold]Coût estimé[/bold]", f"[bold green]~${s['total_cost_usd']:.4f}[/bold green]")

        _console.print(table)