"""
Évaluation des prompts via LLM-as-judge (Claude).

Deux modes disponibles :
- Simulation (défaut) : Claude évalue le prompt en simulant mentalement les sorties. Rapide et économique.
- Exécution réelle (--execute) : le prompt est d'abord exécuté sur chaque exemple, puis le juge
  compare les vraies sorties aux sorties attendues. Plus fiable, plus coûteux.
"""

import asyncio
import anthropic
from typing import Optional

from core.cost_tracker import CostTracker
from core.utils import parse_json_response


JUDGE_SYSTEM = """Tu es un juge expert en qualité de prompts LLM.
Tu évalues la qualité d'un prompt en te basant sur des exemples fournis.

Pour chaque évaluation, tu dois scorer selon ces critères :
   - Clarté et précision des instructions (0-3)
   - Qualité et pertinence des sorties produites (0-4)
   - Robustesse et généralisation (0-3)

Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, sans markdown.
Format : {"score": <float 0-10>, "feedback": "<explication concise>"}
"""


class PromptEvaluator:
    """Évaluateur de prompts LLM utilisant Claude comme juge expert."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", execute: bool = False, tracker: Optional[CostTracker] = None):
        """
        Initialise l'évaluateur de prompts.

        Args:
            model: Modèle Claude à utiliser (défaut : claude-haiku-4-5-20251001).
            execute: Si True, exécute réellement le prompt sur les exemples.
            tracker: Instance CostTracker partagée (optionnel).
        """
        self.client = anthropic.AsyncAnthropic(max_retries=3)
        self.model = model
        self.execute = execute
        self.tracker = tracker or CostTracker()

    async def evaluate(
        self,
        prompt: str,
        examples: list[dict],
        task_description: Optional[str] = None,
    ) -> tuple[float, str]:
        """
        Évalue un prompt candidat sur des exemples entrée/sortie attendus.

        En mode exécution (execute=True), exécute réellement le prompt sur chaque
        exemple et soumet les vraies sorties au juge pour comparaison.
        En mode simulation (défaut), demande au juge de simuler mentalement les sorties.

        Args:
            prompt: Le prompt à évaluer.
            examples: Liste de {"input": ..., "expected_output": ...}.
            task_description: Contexte optionnel de la tâche.

        Returns:
            Tuple (score float 0-10, feedback string).
        """
        if self.execute and examples:
            return await self._evaluate_with_execution(prompt, examples, task_description)
        return await self._evaluate_simulated(prompt, examples, task_description)

    async def _execute_prompt(self, prompt: str, input_text: str) -> str:
        """
        Exécute réellement le prompt sur un exemple et retourne la sortie brute.

        Args:
            prompt: Le prompt candidat à tester.
            input_text: L'entrée de l'exemple.

        Returns:
            La sortie produite par Claude avec ce prompt.
        """
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": f"{prompt}\n\n{input_text}"}],
        )
        self.tracker.track_from_usage(self.model, response.usage)
        return response.content[0].text.strip()

    async def _evaluate_with_execution(
        self,
        prompt: str,
        examples: list[dict],
        task_description: Optional[str],
    ) -> tuple[float, str]:
        """
        Exécute le prompt sur les exemples en parallèle, puis soumet les vraies
        sorties au juge pour une évaluation basée sur les résultats réels.

        Args:
            prompt: Le prompt à évaluer.
            examples: Liste des exemples d'entrée/sortie attendue.
            task_description: Description optionnelle de la tâche.

        Returns:
            Tuple (score float 0-10, feedback string).
        """
        sample = examples[:3]

        actual_outputs = await asyncio.gather(
            *[self._execute_prompt(prompt, ex["input"]) for ex in sample]
        )

        task_context = f"Tâche : {task_description}\n\n" if task_description else ""
        comparison_str = "\n".join([
            f"Exemple {i + 1}:\n"
            f"  Input           : {ex['input']}\n"
            f"  Sortie attendue : {ex['expected_output']}\n"
            f"  Sortie réelle   : {actual}"
            for i, (ex, actual) in enumerate(zip(sample, actual_outputs))
        ])

        judge_message = (
            f"{task_context}Prompt évalué :\n---\n{prompt}\n---\n\n"
            f"Résultats réels sur les exemples :\n{comparison_str}\n\n"
            "Évalue la qualité de ce prompt en te basant sur les sorties réelles obtenues.\n"
            "Réponds uniquement avec le JSON demandé."
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=[{
                "type": "text",
                "text": JUDGE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": judge_message}],
        )
        self.tracker.track_from_usage(self.model, response.usage)
        return self._parse_score(response.content[0].text)

    async def _evaluate_simulated(
        self,
        prompt: str,
        examples: list[dict],
        task_description: Optional[str],
    ) -> tuple[float, str]:
        """
        Évaluation par simulation mentale du juge. Mode rapide et économique.

        Args:
            prompt: Le prompt à évaluer.
            examples: Liste des exemples d'entrée/sortie attendue.
            task_description: Description optionnelle de la tâche.

        Returns:
            Tuple (score float 0-10, feedback string).
        """
        task_context = f"Tâche : {task_description}\n\n" if task_description else ""
        examples_str = "\n".join([
            f"Exemple {i + 1}:\n  Input: {ex['input']}\n  Sortie attendue: {ex['expected_output']}"
            for i, ex in enumerate(examples[:3])
        ])

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=[{
                "type": "text",
                "text": JUDGE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{task_context}Exemples de référence :\n{examples_str}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Prompt à évaluer :\n---\n{prompt}\n---\n\n"
                            "Évalue ce prompt selon les critères définis.\n"
                            "Réponds uniquement avec le JSON demandé."
                        ),
                    },
                ],
            }],
        )
        self.tracker.track_from_usage(self.model, response.usage)
        return self._parse_score(response.content[0].text)

    def _parse_score(self, raw: str) -> tuple[float, str]:
        """
        Parse la réponse JSON du juge et retourne (score, feedback).

        Args:
            raw: Réponse JSON brute du juge.

        Returns:
            Tuple (score float 0-10 borné, feedback string).
        """
        try:
            data = parse_json_response(raw)
            score = float(data.get("score", 5.0))
            feedback = data.get("feedback", "")
            return min(max(score, 0.0), 10.0), feedback
        except Exception:
            print("⚠️ Erreur parsing JSON évaluation, score par défaut 5.0")
            return 5.0, "Erreur d'évaluation"