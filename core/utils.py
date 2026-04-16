"""
Utilitaires partagés : parsing JSON robuste, déduplication, tracking des coûts API.
"""

import json
import re
from dataclasses import dataclass, field


def parse_json_response(raw: str) -> dict:
    """
    Parse une réponse JSON de Claude en nettoyant les balises markdown éventuelles.

    Args:
        raw: Texte brut retourné par Claude.

    Returns:
        Dictionnaire Python parsé.

    Raises:
        json.JSONDecodeError: Si le contenu n'est pas du JSON valide après nettoyage.
    """
    text = raw.strip()
    # Extract the JSON object between first { and last } to handle cases where
    # variant strings themselves contain markdown code fences (e.g. ```python).
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    return json.loads(text)


def deduplicate(items: list) -> list:
    """
    Supprime les doublons d'une liste en préservant l'ordre d'apparition.

    Args:
        items: Liste potentiellement contenant des doublons.

    Returns:
        Liste sans doublons, dans l'ordre original.
    """
    seen: set = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


@dataclass
class CostStats:
    """
    Accumule les statistiques de tokens et estime le coût des appels API Anthropic.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    api_calls: int = 0

    # Prix par million de tokens (input, output) selon le modèle
    _PRICES: dict = field(default_factory=lambda: {
        "claude-haiku-4-5": (1.0, 5.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-opus-4-6": (5.0, 25.0),
    })

    def add(self, usage) -> None:
        """
        Intègre les données d'usage d'une réponse API.

        Args:
            usage: Objet contenant les attributs input_tokens, output_tokens,
                   cache_creation_input_tokens, cache_read_input_tokens.
        """
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.api_calls += 1

    def merge(self, other: "CostStats") -> None:
        """
        Fusionne les stats d'un autre CostStats dans celui-ci.

        Args:
            other: Instance CostStats à fusionner.
        """
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.api_calls += other.api_calls

    def estimate_cost_usd(self, model: str) -> float:
        """
        Estime le coût en USD selon le modèle.

        Cache reads sont facturés à 0.1× le prix d'entrée.
        Cache creation est facturée à 1.25× le prix d'entrée.

        Args:
            model: Nom du modèle Claude utilisé.

        Returns:
            Coût estimé en USD.
        """
        input_price, output_price = next(
            (v for k, v in self._PRICES.items() if k in model),
            (3.0, 15.0),  # fallback sonnet
        )
        return (
            self.input_tokens * input_price / 1_000_000
            + self.output_tokens * output_price / 1_000_000
            + self.cache_read_tokens * input_price * 0.1 / 1_000_000
            + self.cache_creation_tokens * input_price * 1.25 / 1_000_000
        )

    def summary(self, model: str) -> str:
        """
        Retourne un résumé lisible des tokens consommés et du coût estimé.

        Args:
            model: Nom du modèle Claude utilisé.

        Returns:
            Chaîne formatée avec les statistiques d'usage et le coût estimé.
        """
        total_input = self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens
        cache_pct = (
            round(self.cache_read_tokens / total_input * 100) if total_input > 0 else 0
        )
        cost = self.estimate_cost_usd(model)
        return (
            f"Appels API : {self.api_calls} | "
            f"Tokens entrée : {total_input:,} (cache hit : {cache_pct}%) | "
            f"Tokens sortie : {self.output_tokens:,} | "
            f"Coût estimé : ~${cost:.4f}"
        )