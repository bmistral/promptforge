# PromptForge 🔨

> **Auto-optimisation de prompts par boucle agentique — inspiré de DSPy et OPRO**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![Anthropic](https://img.shields.io/badge/Claude-API-orange)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Un prompt mal écrit peut diviser par 3 la qualité d'un LLM. **PromptForge** prend un prompt naïf, génère des variantes, les évalue automatiquement via un LLM-as-judge, et itère jusqu'à trouver le prompt optimal — sans intervention humaine.

---

## 🎯 Le problème

Écrire un bon prompt est difficile et itératif. La plupart des développeurs testent leurs prompts manuellement, sans méthode systématique. PromptForge automatise ce processus en appliquant un **algorithme évolutionnaire aux prompts**.

---

## 🔄 La boucle d'optimisation

```
Prompt naïf
     │
     ▼
┌─────────────────────────────────────────┐
│  GÉNÉRATION  (async, en parallèle)      │
│  Claude génère N variantes du prompt    │
│  (few-shot, CoT, rôle, contraintes...)  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  ÉVALUATION (LLM-as-Judge, parallèle)   │
│  Score chaque variante sur :            │
│  • Clarté des instructions    (0-3)     │
│  • Qualité des sorties        (0-4)     │
│  • Robustesse / généralisation (0-3)    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  SÉLECTION + MUTATION                   │
│  Top-K conservés → croisement/mutation  │
└──────────────┬──────────────────────────┘
               │
        Répéter N fois
               │
               ▼
        Prompt optimisé
        + rapport complet + coûts API
```

---

## ✨ Fonctionnalités

- **Génération automatique** de variantes (few-shot, chain-of-thought, rôle, format...)
- **LLM-as-judge** : Claude évalue objectivement chaque prompt sur vos exemples
- **Algorithme évolutionnaire** : sélection des meilleurs, mutation, croisement
- **10 datasets intégrés** : résumé, sentiment, extraction, code, adresse, langue, ticket, regex, feedback, sécurité
- **Deux modes d'évaluation** : simulation mentale (rapide) ou exécution réelle avec `--execute` (~3× plus fiable)
- **Parallélisation async** : toutes les évaluations d'une itération s'exécutent en parallèle
- **Prompt caching** : mise en cache du système et des exemples pour réduire les coûts
- **Suivi des coûts** en temps réel avec alerte si le budget est dépassé
- **Retry automatique** avec backoff exponentiel sur tous les appels API (tenacity)
- **Checkpoint / reprise** : sauvegarde après chaque itération, `--resume` pour reprendre un run interrompu
- **Historique complet** des runs sauvegardé en JSON

---

## 🚀 Démarrage rapide

### Installation

```bash
git clone https://github.com/bmistral/promptforge.git
cd promptforge

# Avec uv (recommandé)
uv sync

# Ou avec pip
pip install -e .
pip install -e ".[notebook]"  # pour les notebooks Jupyter
pip install -e ".[dev]"       # pour pytest / ruff / mypy
```

### Configuration

```bash
cp .env.example .env
# Éditez .env et renseignez votre clé :
# ANTHROPIC_API_KEY=sk-ant-...
# MONTHLY_BUDGET_USD=20   ← alerte si dépassé
```

### Lancement

```bash
# Dataset de résumé (par défaut)
python main.py --dataset summarization

# Classification de sentiment, 3 itérations, évaluation réelle
python main.py --dataset sentiment --iterations 3 --execute

# Avec votre propre prompt
python main.py --prompt "Analyse ce texte." --task "Analyse de sentiment"

# Reprendre un run interrompu
python main.py --resume results/optimization_runs/checkpoint_2024-....json
```

---

## 📊 Exemple de résultat

**Prompt initial :** `Résume ce texte.`

**Prompt optimisé après 4 itérations :**
```
En tant qu'expert en synthèse d'information, résume le texte suivant en UNE seule phrase
(30 mots maximum). Ta phrase doit capturer l'idée principale et les données chiffrées clés
si présentes. Format : une phrase directe, sans introduction.
```

**Progression des scores :**

| Itération | Score moyen | Meilleur score |
|-----------|-------------|----------------|
| 1         | 5.2 / 10    | 6.8 / 10       |
| 2         | 6.7 / 10    | 7.9 / 10       |
| 3         | 7.5 / 10    | 8.4 / 10       |
| 4         | 8.1 / 10    | 8.9 / 10       |

---

## 🗂️ Datasets intégrés

10 cas d'usage prêts à l'emploi, du plus simple au plus complexe :

| Dataset | `--dataset` | Prompt initial naïf | Tâche |
|---|---|---|---|
| Résumé | `summarization` | `Résume ce texte.` | Résumer en une phrase concise |
| Sentiment | `sentiment` | `Quel est le sentiment de ce texte ?` | Classifier positif / négatif / neutre |
| Extraction | `extraction` | `Extrais les informations importantes.` | Extraire entités (personne, lieu, date…) |
| Code Python | `code` | `Écris du code Python pour faire ça :` | Générer une fonction propre et documentée |
| Adresse | `address` | `Reformate cette adresse.` | Normaliser au format standard |
| Langue | `language` | `Quelle est la langue de ce texte ?` | Identifier et retourner le code ISO 639-1 |
| Ticket support | `ticket` | `Donne la priorité de ce ticket.` | Classer critique / haute / moyenne / basse |
| Regex | `regex` | `Écris une regex Python pour ce format.` | Générer une regex compilable et précise |
| Feedback RH | `feedback` | `Reformule ce feedback pour qu'il soit constructif.` | Transformer un retour brut en feedback actionnable |
| Audit sécu | `security` | `Analyse ce code pour trouver des problèmes de sécurité.` | Détecter vulnérabilités et proposer des corrections |

---

## 💰 Suivi des coûts API

PromptForge affiche un tableau récapitulatif à la fin de chaque run :

```
┌─────────────────────────────────────────────┐
│           Coûts API — session               │
├──────────────────────┬──────────────────────┤
│ Appels API           │                   87 │
│ Tokens entrée        │               42,310 │
│ Tokens sortie        │                8,420 │
│ Cache read           │           38,100 (90%)│
│ Cache write          │                4,210 │
│ Coût estimé          │              ~$0.012 │
└──────────────────────┴──────────────────────┘
```

Configurez `MONTHLY_BUDGET_USD` dans `.env` pour recevoir une alerte si 80 % du budget journalier est consommé en une session.

---

## ⚙️ Options CLI

```
--dataset      Dataset intégré : summarization | sentiment | extraction | code |
               address | language | ticket | regex | feedback | security
--prompt       Prompt initial personnalisé
--task         Description de la tâche (avec --prompt)
--iterations   Nombre d'itérations (default: 4)
--variants     Variantes générées par itération (default: 5)
--top-k        Meilleurs prompts conservés entre itérations (default: 2)
--model        Modèle Claude (default: claude-haiku-4-5-20251001)
--execute      Exécute réellement chaque prompt sur les exemples avant scoring
               (plus fiable, ~3× plus d'appels API)
--resume       Reprend un run depuis un fichier checkpoint JSON
```

---

## 🧪 Tests

```bash
# Lancer toute la suite
pytest

# Un module en détail
pytest tests/unit/test_evaluator.py -v

# Avec couverture
pytest --tb=short -q
```

Les tests couvrent `evaluator._parse_score`, `CostTracker` (accumulation, alertes budget, `summary`), `parse_json_response` et `deduplicate`. Aucun appel API réel : le client Anthropic est intégralement mocké via `unittest.mock`.

---

## 📓 Utilisation dans Jupyter / notebooks

Le notebook `notebooks/demo.ipynb` couvre l'ensemble du workflow avec visualisations Plotly.

Il fonctionne **sans clé API** grâce à un mode mock activé par défaut :

```python
USE_MOCK = True   # ← charge les résultats pré-calculés depuis datasets/fixtures/
                  #   passer à False pour des appels API réels
```

Les résultats de référence sont stockés dans `datasets/fixtures/demo_code_run.json` (run sur le dataset `code`, 3 itérations, meilleur score 8.50/10). Changer `USE_MOCK = False` nécessite une `ANTHROPIC_API_KEY` configurée dans `.env`.

Jupyter fait déjà tourner une boucle asyncio — utilisez `arun()` avec `await` plutôt que `run()` pour éviter les conflits de contexte (Python 3.12+) :

```python
from core.loop import PromptForge
from datasets.examples import DATASETS

forge = PromptForge(model="claude-haiku-4-5-20251001", n_iterations=4, n_variants=5)
dataset = DATASETS["summarization"]

# Dans une cellule Jupyter — await obligatoire
run = await forge.arun(
    initial_prompt=dataset["initial_prompt"],
    examples=dataset["examples"],
    task_description=dataset["task"],
    verbose=True,
)
print(run.best_prompt)
```

`run()` (synchrone) reste disponible pour les scripts Python classiques.

---

## 📂 Structure du projet

```
promptforge/
│
├── core/
│   ├── generator.py        ← Génération de variantes via Claude
│   ├── evaluator.py        ← LLM-as-judge (scoring + feedback, 2 modes)
│   ├── optimizer.py        ← Mutation / croisement des meilleurs prompts
│   ├── loop.py             ← Orchestration async, checkpoint, sélection
│   ├── cost_tracker.py     ← Suivi des coûts API + alertes budget (rich)
│   ├── models.py           ← Dataclasses (OptimizationRun, IterationResult)
│   └── utils.py            ← parse_json_response, deduplicate, CostStats
│
├── datasets/
│   ├── examples.py         ← 10 datasets de test
│   └── fixtures/
│       └── demo_code_run.json  ← Résultats pré-calculés pour le notebook (mode mock)
│
├── notebooks/
│   └── demo.ipynb          ← Démo complète avec visualisations Plotly (USE_MOCK=True par défaut)
│
├── tests/
│   └── unit/
│       ├── test_evaluator.py   ← Tests de _parse_score et evaluate (client mocké)
│       ├── test_cost_tracker.py← Tests d'accumulation, alertes budget, summary
│       └── test_utils.py       ← Tests de parse_json_response et deduplicate
│
├── results/                ← Runs JSON + checkpoints (gitignorés)
│
├── .env.example            ← Variables d'environnement à configurer
├── .pre-commit-config.yaml ← Hooks ruff (lint + format)
├── pyproject.toml          ← Dépendances et config outils
├── main.py                 ← Point d'entrée CLI
└── README.md
```

---

## 🧠 Inspirations & références

- **DSPy** (Stanford, 2023) — Automatic prompting as a compilation problem
- **OPRO** (Google DeepMind, 2023) — *Large Language Models as Optimizers*
- **APE** (Zhou et al., 2022) — *Large Language Models Are Human-Level Prompt Engineers*

---

## 💡 Idées d'extensions

- [ ] Support multi-LLM (OpenAI, Mistral) pour comparer les judges
- [ ] Interface Streamlit pour une utilisation no-code
- [ ] Export du prompt optimisé vers un fichier `.txt` / `.yaml`
- [ ] Métriques automatiques complémentaires (BLEU, ROUGE pour le résumé)
- [ ] Support de prompts système vs prompts utilisateur

---

## 📄 Licence

MIT — Baptiste Mistral
