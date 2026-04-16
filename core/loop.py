"""
PromptForge — Boucle principale d'optimisation agentique.

- Évaluations parallèles via asyncio.gather()
- Checkpoint JSON après chaque itération (reprise possible avec --resume)
- Tracking des coûts API en temps réel via CostTracker partagé
- Distribution correcte des variantes entre les parents
- Déduplication des candidats sans perte d'ordre
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Optional

from core.cost_tracker import CostTracker
from core.evaluator import PromptEvaluator
from core.generator import PromptGenerator
from core.models import OptimizationRun, IterationResult
from core.optimizer import PromptOptimizer
from core.utils import deduplicate


class PromptForge:
    """
    Orchestre la boucle Génère → Évalue → Sélectionne → Mute pour optimiser un prompt.

    Toutes les évaluations d'une itération sont exécutées en parallèle.
    Un checkpoint est sauvegardé après chaque itération pour permettre la reprise.
    Un CostTracker partagé accumule les coûts des trois composants.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        n_variants: int = 5,
        n_iterations: int = 4,
        top_k: int = 2,
        execute: bool = False,
        results_dir: str = "results/optimization_runs",
    ):
        monthly_budget = float(os.environ.get("MONTHLY_BUDGET_USD", 0) or 0) or None
        self.tracker = CostTracker(monthly_budget_usd=monthly_budget)
        self.generator = PromptGenerator(model=model, tracker=self.tracker)
        self.evaluator = PromptEvaluator(model=model, execute=execute, tracker=self.tracker)
        self.optimizer = PromptOptimizer(model=model, tracker=self.tracker)
        self.model = model
        self.n_variants = n_variants
        self.n_iterations = n_iterations
        self.top_k = top_k
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    # ─── Point d'entrée public ────────────────────────────────────────────────

    def run(
        self,
        initial_prompt: str,
        examples: list[dict],
        task_description: Optional[str] = None,
        verbose: bool = True,
        checkpoint_path: Optional[str] = None,
    ) -> OptimizationRun:
        """
        Lance (ou reprend) l'optimisation complète de manière synchrone.

        Args:
            initial_prompt: Le prompt de départ.
            examples: Liste de {"input": ..., "expected_output": ...}.
            task_description: Description optionnelle de la tâche.
            verbose: Affiche la progression en temps réel.
            checkpoint_path: Chemin vers un checkpoint existant pour reprendre un run.

        Returns:
            OptimizationRun avec l'historique complet et le meilleur prompt trouvé.
        """
        return asyncio.run(
            self.arun(initial_prompt, examples, task_description, verbose, checkpoint_path)
        )

    async def arun(
        self,
        initial_prompt: str,
        examples: list[dict],
        task_description: Optional[str] = None,
        verbose: bool = True,
        checkpoint_path: Optional[str] = None,
    ) -> OptimizationRun:
        """
        Version async de run() — à utiliser avec ``await`` dans Jupyter ou tout contexte
        où une boucle événementielle est déjà active (évite le conflit de contexte Python 3.12).

        Args:
            initial_prompt: Le prompt de départ.
            examples: Liste de {"input": ..., "expected_output": ...}.
            task_description: Description optionnelle de la tâche.
            verbose: Affiche la progression en temps réel.
            checkpoint_path: Chemin vers un checkpoint existant pour reprendre un run.

        Returns:
            OptimizationRun avec l'historique complet et le meilleur prompt trouvé.
        """
        return await self._run_async(
            initial_prompt, examples, task_description, verbose, checkpoint_path
        )

    # ─── Boucle async principale ──────────────────────────────────────────────

    async def _run_async(
        self,
        initial_prompt: str,
        examples: list[dict],
        task_description: Optional[str],
        verbose: bool,
        checkpoint_path: Optional[str],
    ) -> OptimizationRun:
        """
        Boucle d'optimisation async avec parallélisation des évaluations.

        Orchestre les quatre phases (génération, évaluation, sélection, mutation)
        pendant n_iterations cycles, avec checkpoints après chaque itération.
        """

        run, current_pool, start_iteration = self._init_or_resume(
            initial_prompt, examples, task_description, checkpoint_path
        )
        run_id = run.timestamp.replace(":", "-")

        for iteration in range(start_iteration, self.n_iterations + 1):
            if verbose:
                print(f"\n{'='*50}")
                mode = " [exécution réelle]" if self.evaluator.execute else ""
                print(f"🔄 Itération {iteration}/{self.n_iterations}{mode}")
                print(f"{'='*50}")

            # 1. GÉNÉRATION — distribuée équitablement entre les parents
            variants = await self._generate_all(current_pool, task_description, verbose)
            all_candidates = deduplicate(current_pool + variants)

            if verbose:
                print(f"📝 {len(all_candidates)} candidats (dont {len(current_pool)} parents conservés)")

            # 2. ÉVALUATION — toutes les évaluations en parallèle
            scored_candidates = await self._evaluate_all(
                all_candidates, examples, task_description, verbose
            )

            # 3. SÉLECTION — top K par score décroissant
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored_candidates[: self.top_k]
            best_prompt, best_score, best_feedback = top_candidates[0]

            if verbose:
                print(f"\n🏆 Meilleur score itération {iteration}: {best_score:.2f}/10")
                print(f"💬 Feedback: {best_feedback[:150]}...")

            # Mise à jour du meilleur global
            if best_score > run.best_score:
                run.best_score = best_score
                run.best_prompt = best_prompt

            # Sauvegarde de l'itération
            avg = sum(s for _, s, _ in scored_candidates) / len(scored_candidates)
            iteration_result = IterationResult(
                iteration=iteration,
                candidates=[
                    {"prompt": p, "score": s, "feedback": f}
                    for p, s, f in scored_candidates
                ],
                best_prompt=best_prompt,
                best_score=best_score,
                avg_score=avg,
            )
            run.iterations.append(iteration_result)

            # 4. MUTATION — prépare le pool suivant (sauf dernière itération)
            if iteration < self.n_iterations:
                current_pool = await self.optimizer.mutate(
                    top_prompts=[p for p, _, _ in top_candidates],
                    feedbacks=[f for _, _, f in top_candidates],
                    task_description=task_description,
                    n_output=self.top_k,
                )

            # Checkpoint après chaque itération
            self._save_checkpoint(run, current_pool, iteration + 1, run_id)

        # Sauvegarde du run final et suppression du checkpoint
        self._save_run(run, run_id)
        self._delete_checkpoint(run_id)

        # Résumé des coûts
        run.cost_summary = self.tracker.summary()
        self.tracker.alert_if_over_budget()

        if verbose:
            print(f"\n{'='*50}")
            print("✅ Optimisation terminée !")
            print(f"📈 Score initial estimé : {run.iterations[0].avg_score:.2f}/10")
            print(f"🚀 Meilleur score final  : {run.best_score:.2f}/10")
            self.tracker.print_summary()
            print(f"{'='*50}")
            print(f"\n🏆 Prompt optimisé :\n{run.best_prompt}")

        return run

    # ─── Helpers async ────────────────────────────────────────────────────────

    async def _generate_all(
        self,
        current_pool: list[str],
        task_description: Optional[str],
        verbose: bool,
    ) -> list[str]:
        """
        Génère les variantes depuis tous les parents en parallèle.

        Distribue n_variants équitablement entre les parents
        (le reste va aux premiers parents).

        Args:
            current_pool: Liste des prompts parents.
            task_description: Description optionnelle de la tâche.
            verbose: Affiche les détails d'exécution.

        Returns:
            Liste de tous les variantes générées.
        """
        pool_size = len(current_pool)
        base = self.n_variants // pool_size
        remainder = self.n_variants % pool_size

        tasks = [
            self.generator.generate_variants(
                prompt=parent,
                task_description=task_description,
                n=base + (1 if i < remainder else 0),
            )
            for i, parent in enumerate(current_pool)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        variants = []
        for r in results:
            if isinstance(r, Exception):
                print(f"⚠️ Erreur génération ignorée : {r}")
            else:
                variants.extend(r)
        return variants

    async def _evaluate_all(
        self,
        candidates: list[str],
        examples: list[dict],
        task_description: Optional[str],
        verbose: bool,
    ) -> list[tuple[str, float, str]]:
        """
        Évalue tous les candidats en parallèle.

        Les erreurs individuelles reçoivent un score de 5.0 par défaut
        avec un feedback d'erreur.

        Args:
            candidates: Liste des prompts à évaluer.
            examples: Exemples pour l'évaluation.
            task_description: Description optionnelle de la tâche.
            verbose: Affiche le score de chaque candidat.

        Returns:
            Liste de tuples (prompt, score, feedback).
        """
        tasks = [
            self.evaluator.evaluate(c, examples, task_description)
            for c in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scored = []
        for candidate, result in zip(candidates, results):
            if isinstance(result, Exception):
                print(f"⚠️ Erreur évaluation ignorée : {result}")
                scored.append((candidate, 5.0, "Erreur d'évaluation"))
            else:
                score, feedback = result
                scored.append((candidate, score, feedback))
                if verbose:
                    print(f"  ✓ Score : {score:.2f}/10")

        return scored

    # ─── Checkpoint ──────────────────────────────────────────────────────────

    def _checkpoint_path(self, run_id: str) -> str:
        """Retourne le chemin du fichier checkpoint pour un run donné."""
        return os.path.join(self.results_dir, f"checkpoint_{run_id}.json")

    def _save_checkpoint(
        self, run: OptimizationRun, current_pool: list[str], next_iteration: int, run_id: str
    ) -> None:
        """
        Sauvegarde l'état courant pour permettre la reprise en cas d'interruption.

        Args:
            run: L'OptimizationRun actuel.
            current_pool: Le pool de prompts pour la prochaine itération.
            next_iteration: Le numéro de la prochaine itération.
            run_id: Identifiant unique du run.
        """
        checkpoint = {
            "run_id": run_id,
            "next_iteration": next_iteration,
            "current_pool": current_pool,
            "config": {
                "model": self.model,
                "n_variants": self.n_variants,
                "n_iterations": self.n_iterations,
                "top_k": self.top_k,
                "execute": self.evaluator.execute,
            },
            "run": run.to_dict(),
        }
        path = self._checkpoint_path(run_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def _delete_checkpoint(self, run_id: str) -> None:
        """Supprime le fichier checkpoint d'un run."""
        path = self._checkpoint_path(run_id)
        if os.path.exists(path):
            os.remove(path)

    def _init_or_resume(
        self,
        initial_prompt: str,
        examples: list[dict],
        task_description: Optional[str],
        checkpoint_path: Optional[str],
    ) -> tuple[OptimizationRun, list[str], int]:
        """
        Initialise un nouveau run ou reprend depuis un checkpoint existant.

        Args:
            initial_prompt: Le prompt de départ.
            examples: Liste des exemples.
            task_description: Description optionnelle de la tâche.
            checkpoint_path: Chemin du checkpoint pour la reprise (optionnel).

        Returns:
            Tuple (run, current_pool, start_iteration).
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, encoding="utf-8") as f:
                ckpt = json.load(f)

            print(f"♻️  Reprise depuis le checkpoint : {checkpoint_path}")
            run_dict = ckpt["run"]
            iterations = [
                IterationResult(
                    iteration=it["iteration"],
                    candidates=it["candidates"],
                    best_prompt=it["best_prompt"],
                    best_score=it["best_score"],
                    avg_score=it["avg_score"],
                )
                for it in run_dict["iterations"]
            ]
            run = OptimizationRun(
                initial_prompt=run_dict["initial_prompt"],
                task_description=run_dict["task_description"],
                examples=run_dict["examples"],
                iterations=iterations,
                best_prompt=run_dict["best_prompt"],
                best_score=run_dict["best_score"],
                timestamp=run_dict["timestamp"],
            )
            return run, ckpt["current_pool"], ckpt["next_iteration"]

        run = OptimizationRun(
            initial_prompt=initial_prompt,
            task_description=task_description or "",
            examples=examples,
            iterations=[],
            best_prompt=initial_prompt,
            best_score=0.0,
            timestamp=datetime.now().isoformat(),
        )
        return run, [initial_prompt], 1

    # ─── Persistance du run final ─────────────────────────────────────────────

    def _save_run(self, run: OptimizationRun, run_id: str) -> None:
        """
        Sauvegarde le run final en JSON.

        Args:
            run: L'OptimizationRun complet.
            run_id: Identifiant unique du run.
        """
        filename = os.path.join(self.results_dir, f"run_{run_id}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n\U0001f4be Run sauvegardé : {filename}")
