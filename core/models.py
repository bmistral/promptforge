"""
Modèles de données pour PromptForge.
"""

from dataclasses import dataclass


@dataclass
class IterationResult:
    """Résultats d'une itération d'optimisation de prompt.
    
    Attributes:
        iteration: Numéro de l'itération.
        candidates: Liste des prompts candidats générés.
        best_prompt: Meilleur prompt trouvé dans cette itération.
        best_score: Score du meilleur prompt.
        avg_score: Score moyen de tous les candidats.
    """
    iteration: int
    candidates: list[dict]
    best_prompt: str
    best_score: float
    avg_score: float

    def to_dict(self) -> dict:
        """Convertit le résultat d'itération en dictionnaire.
        
        Returns:
            Dictionnaire contenant les résultats d'itération avec scores arrondis à 4 décimales.
        """
        return {
            "iteration": self.iteration,
            "candidates": self.candidates,
            "best_prompt": self.best_prompt,
            "best_score": round(self.best_score, 4),
            "avg_score": round(self.avg_score, 4),
        }


@dataclass
class OptimizationRun:
    """Résultats complets d'une exécution d'optimisation de prompt.
    
    Attributes:
        initial_prompt: Prompt initial avant optimisation.
        task_description: Description de la tâche à optimiser.
        examples: Exemples d'entrée/sortie pour évaluer les prompts.
        iterations: Liste des résultats de chaque itération.
        best_prompt: Meilleur prompt trouvé après optimisation.
        best_score: Score du meilleur prompt final.
        timestamp: Horodatage de l'exécution.
        cost_summary: Résumé des coûts d'utilisation des API.
    """
    initial_prompt: str
    task_description: str
    examples: list[dict]
    iterations: list[IterationResult]
    best_prompt: str
    best_score: float
    timestamp: str
    cost_summary: dict = None

    def to_dict(self) -> dict:
        """Convertit l'exécution d'optimisation en dictionnaire.
        
        Returns:
            Dictionnaire contenant les résultats complets de l'optimisation avec scores arrondis à 4 décimales.
        """
        return {
            "initial_prompt": self.initial_prompt,
            "task_description": self.task_description,
            "examples": self.examples,
            "iterations": [it.to_dict() for it in self.iterations],
            "best_prompt": self.best_prompt,
            "best_score": round(self.best_score, 4),
            "timestamp": self.timestamp,
            "cost_summary": self.cost_summary or {},
        }