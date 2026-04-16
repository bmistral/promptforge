"""
Mutation et croisement des meilleurs prompts pour la génération suivante.
"""

import anthropic
from typing import Optional

from core.cost_tracker import CostTracker
from core.utils import parse_json_response


OPTIMIZER_SYSTEM = """Tu es un expert en prompt engineering évolutionnaire.
On te donne les meilleurs prompts d'une génération avec leurs feedbacks.
Ton rôle est de les combiner et les améliorer pour produire de nouveaux prompts encore meilleurs.

Stratégies à appliquer :
- Combine les points forts de chaque prompt parent
- Corrige les faiblesses identifiées dans les feedbacks
- Explore de nouvelles variations inspirées des meilleurs éléments
- Assure-toi que chaque nouveau prompt est distinct

Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, sans markdown.
Format : {"mutated_prompts": ["prompt1", "prompt2", ...]}
"""


class PromptOptimizer:
    """Optimiseur de prompts par mutation et croisement génétique."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", tracker: Optional[CostTracker] = None):
        """
        Initialise l'optimiseur de prompts.

        Args:
            model: Modèle Claude à utiliser pour l'optimisation.
            tracker: Instance CostTracker partagée (optionnel).
        """
        self.client = anthropic.AsyncAnthropic(max_retries=3)
        self.model = model
        self.tracker = tracker or CostTracker()

    async def mutate(
        self,
        top_prompts: list[str],
        feedbacks: list[str],
        task_description: Optional[str] = None,
        n_output: int = 2,
    ) -> list[str]:
        """
        Génère de nouveaux prompts par mutation et croisement des meilleurs candidats.

        Combine les points forts des prompts parents et corrige les faiblesses
        identifiées dans les feedbacks du juge.

        Args:
            top_prompts: Les meilleurs prompts de la génération actuelle.
            feedbacks: Les feedbacks du juge pour chaque prompt parent.
            task_description: Contexte optionnel de la tâche.
            n_output: Nombre de prompts mutés à produire.

        Returns:
            Liste de prompts mutés (retombe sur les parents si parsing échoue).
        """
        task_context = f"Tâche : {task_description}\n\n" if task_description else ""
        parents_str = "\n\n".join([
            f"Parent {i + 1}:\nPrompt: {p}\nFeedback: {f}"
            for i, (p, f) in enumerate(zip(top_prompts, feedbacks))
        ])

        user_message = (
            f"{task_context}Meilleurs prompts de la génération actuelle avec leurs feedbacks :\n\n"
            f"{parents_str}\n\n"
            f"Génère exactement {n_output} nouveaux prompts améliorés par mutation/croisement.\n"
            "Réponds uniquement avec le JSON demandé."
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=[{
                "type": "text",
                "text": OPTIMIZER_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
        )
        self.tracker.track_from_usage(self.model, response.usage)

        try:
            data = parse_json_response(response.content[0].text)
            mutated = data.get("mutated_prompts", top_prompts)
            return mutated[:n_output]
        except Exception:
            print("⚠️ Erreur parsing JSON mutation, retour aux parents")
            return top_prompts[:n_output]