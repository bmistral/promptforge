"""
PromptForge — Point d'entrée CLI.

Usage:
    python main.py --dataset summarization
    python main.py --dataset sentiment --iterations 3 --variants 4
    python main.py --prompt "Résume ce texte." --task "Résumé court"
    python main.py --dataset code --execute
    python main.py --resume results/optimization_runs/checkpoint_2024-...json
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from core.loop import PromptForge
from datasets.examples import DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PromptForge — Optimisation automatique de prompts par boucle agentique"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        default="summarization",
        help="Dataset de test à utiliser (default: summarization)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt initial personnalisé (remplace le dataset)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Description de la tâche (optionnel avec --prompt)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=4,
        help="Nombre d'itérations d'optimisation (default: 4)",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=5,
        help="Nombre de variantes générées par itération (default: 5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Nombre de meilleurs prompts conservés (default: 2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Modèle Claude à utiliser (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help=(
            "Exécute réellement chaque prompt candidat sur les exemples avant de le noter. "
            "Plus fiable, mais ~3× plus d'appels API."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Reprend un run interrompu depuis le fichier checkpoint indiqué.",
    )

    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if args.iterations < 1:
        parser.error("--iterations doit être >= 1")
    if args.variants < 1:
        parser.error("--variants doit être >= 1")
    if args.top_k < 1:
        parser.error("--top-k doit être >= 1")
    if args.top_k >= args.variants:
        print(
            f"⚠️  Attention : --top-k ({args.top_k}) >= --variants ({args.variants}). "
            "Tous les candidats seront conservés, la sélection sera inefficace.",
            file=sys.stderr,
        )
    if args.resume and not os.path.isfile(args.resume):
        parser.error(f"Fichier checkpoint introuvable : {args.resume}")
    if args.prompt and not args.task:
        print(
            "⚠️  Mode prompt personnalisé sans --task : l'évaluation sera moins précise.",
            file=sys.stderr,
        )

    return args


def main() -> None:
    args = parse_args()

    # ── Chargement du dataset ou du prompt personnalisé ───────────────────────
    if args.resume:
        # En mode reprise, le prompt/task/examples sont dans le checkpoint
        initial_prompt = ""
        task_description = ""
        examples: list[dict] = []
        print(f"♻️  Reprise depuis : {args.resume}")
    elif args.prompt:
        initial_prompt = args.prompt
        task_description = args.task
        examples = []
        print("📝 Mode prompt personnalisé — aucun exemple fourni.")
    else:
        dataset = DATASETS[args.dataset]
        initial_prompt = dataset["initial_prompt"]
        task_description = dataset["task"]
        examples = dataset["examples"]
        print(f"📂 Dataset : {args.dataset} ({len(examples)} exemples)")

    if not args.resume:
        print(f"\n🎯 Tâche    : {task_description}")
        print(f"📝 Prompt   : {initial_prompt}")
    print(
        f"⚙️  Config   : {args.iterations} itérations, {args.variants} variantes, "
        f"top-{args.top_k}, exécution={'oui' if args.execute else 'non'}"
    )
    print(f"🤖 Modèle   : {args.model}\n")

    # ── Lancement ─────────────────────────────────────────────────────────────
    forge = PromptForge(
        model=args.model,
        n_variants=args.variants,
        n_iterations=args.iterations,
        top_k=args.top_k,
        execute=args.execute,
    )

    run = forge.run(
        initial_prompt=initial_prompt,
        examples=examples,
        task_description=task_description,
        verbose=True,
        checkpoint_path=args.resume,
    )

    print(f"\n{'='*50}")
    print("📊 RÉSUMÉ FINAL")
    print(f"{'='*50}")
    print(f"Prompt initial  : {run.initial_prompt}")
    print(f"Prompt optimisé : {run.best_prompt}")
    print(f"Score final     : {run.best_score:.2f}/10")


if __name__ == "__main__":
    main()
