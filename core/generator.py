"""
Génération de variantes de prompts via Claude.
"""

import anthropic
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from core.cost_tracker import CostTracker
from core.utils import parse_json_response


GENERATION_SYSTEM = """Tu es un expert en prompt engineering.
Ton rôle est de générer des variantes améliorées d'un prompt donné.
Chaque variante doit explorer une approche différente :
- reformulation, ajout de contexte, instructions plus précises,
- exemples intégrés (few-shot), chaîne de pensée (chain-of-thought),
- rôle assigné au modèle, contraintes de format, etc.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, sans markdown.
Format attendu : {"variants": ["variante1", "variante2", ...]}
"""


class PromptGenerator:
    """
    Générateur de variantes de prompts via l'API Claude.
    
    Utilise Claude pour explorer différentes stratégies de prompt engineering
    et générer des variantes améliorées d'un prompt donné.
    """
    
    def __init__(self, model: str = "claude-haiku-4-5-20251001", tracker: Optional[CostTracker] = None):
        """
        Initialise le générateur de prompts.

        Args:
            model: Identifiant du modèle Claude à utiliser.
            tracker: Instance CostTracker partagée (optionnel).
        """
        self.client = anthropic.AsyncAnthropic(max_retries=3)
        self.model = model
        self.tracker = tracker or CostTracker()

    async def generate_variants(
        self,
        prompt: str,
        task_description: Optional[str] = None,
        n: int = 3,
    ) -> list[str]:
        """
        Génère N variantes améliorées d'un prompt en explorant différentes
        stratégies de prompt engineering.

        Args:
            prompt: Le prompt de départ à améliorer.
            task_description: Description optionnelle de la tâche pour guider les variantes.
            n: Nombre de variantes à générer.

        Returns:
            Liste de variantes (peut être < n si le LLM en génère moins).
        """
        task_context = f"\nContexte de la tâche : {task_description}" if task_description else ""

        user_message = f"""Prompt original à améliorer :
---
{prompt}
---
{task_context}

Génère exactement {n} variantes améliorées de ce prompt.
Chaque variante doit explorer une stratégie de prompt engineering différente.
Réponds uniquement avec le JSON demandé."""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=[{
                "type": "text",
                "text": GENERATION_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
        )
        self.tracker.track_from_usage(self.model, response.usage)

        try:
            data = parse_json_response(response.content[0].text)
            variants = data.get("variants", [])
            return variants[:n]
        except Exception:
            print("⚠️ Erreur parsing JSON génération, fallback sur prompt original")
            return [prompt]