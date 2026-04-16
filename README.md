# PromptForge 🔨

> **Auto-optimisation de prompts par boucle agentique — inspiré de DSPy et OPRO**

![Tests](https://github.com/bmistral/promptforge/actions/workflows/tests.yml/badge.svg)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![Anthropic](https://img.shields.io/badge/Claude-API-orange)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Un prompt mal écrit peut diviser par 3 la qualité d'un LLM. **PromptForge** prend un prompt naïf, génère des variantes, les évalue automatiquement via un LLM-as-judge couplé à des métriques déterministes, et itère jusqu'à trouver le prompt optimal — sans intervention humaine.

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
│  ÉVALUATION HYBRIDE (parallèle)         │
│  LLM-as-Judge :                         │
│  • Clarté des instructions    (0-3)     │
│  • Qualité des sorties        (0-4)     │
│  • Robustesse / généralisation (0-3)    │
│  Métriques déterministes :              │
│  • exact_match / ROUGE-L / F1 / …      │
│  Score final = α × LLM + (1-α) × Det   │
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
- **Score hybride** : combine le LLM-judge avec des métriques déterministes/sémantiques par dataset (exact match, ROUGE-L, F1 token, similarité sémantique…) via un coefficient α configurable
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
pip install -e ".[metrics]"   # métriques déterministes (ROUGE-L, sentence-transformers)
pip install -e ".[notebook]"  # pour les notebooks Jupyter
pip install -e ".[dev]"       # pour pytest / ruff / mypy + toutes les dépendances
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

| Dataset | `--dataset` | Prompt initial naïf | Tâche | Métrique déterministe |
|---|---|---|---|---|
| Résumé | `summarization` | `Résume ce texte.` | Résumer en une phrase concise | ROUGE-L |
| Sentiment | `sentiment` | `Quel est le sentiment de ce texte ?` | Classifier positif / négatif / neutre | Exact match |
| Extraction | `extraction` | `Extrais les informations importantes.` | Extraire entités (personne, lieu, date…) | F1 token |
| Code Python | `code` | `Écris du code Python pour faire ça :` | Générer une fonction propre et documentée | Exécution subprocess |
| Adresse | `address` | `Reformate cette adresse.` | Normaliser au format standard | LLM-judge seul |
| Langue | `language` | `Quelle est la langue de ce texte ?` | Identifier et retourner le code ISO 639-1 | Exact match |
| Ticket support | `ticket` | `Donne la priorité de ce ticket.` | Classer critique / haute / moyenne / basse | Exact match |
| Regex | `regex` | `Écris une regex Python pour ce format.` | Générer une regex compilable et précise | Validité + cas de test |
| Feedback RH | `feedback` | `Reformule ce feedback pour qu'il soit constructif.` | Transformer un retour brut en feedback actionnable | Similarité sémantique (70 %) + ratio longueur (30 %) |
| Audit sécu | `security` | `Analyse ce code pour trouver des problèmes de sécurité.` | Détecter vulnérabilités et proposer des corrections | LLM-judge seul |

---

## ⚖️ Score hybride

Le score final est un mélange pondéré du LLM-judge et des métriques déterministes :

```
score_final = α × score_llm_judge + (1 − α) × score_déterministe × 10
```

`α` est configurable dans `config.yml` (valeur globale et surcharges par dataset) et passé à `PromptEvaluator(alpha=...)` lors de l'initialisation. Si aucune métrique déterministe n'est disponible pour le dataset, `α` est automatiquement forcé à 1.0 (score LLM pur).

**Recommandations par dataset :**

| Dataset | α recommandé | Raison |
|---|---|---|
| `sentiment`, `language`, `ticket` | 0.0 | Exact match fiable à 100 % — le LLM-judge est redondant |
| `summarization`, `extraction`, `feedback` | 0.3–0.5 | La métrique objective ancre le score, le LLM nuance |
| `security`, `address` | 1.0 | Pas de métrique déterministe définie |

Ces valeurs sont documentées dans `config.yml`.

### Métriques disponibles

| Fonction | Librairie | Datasets |
|---|---|---|
| `exact_match` | stdlib | sentiment, language, ticket |
| `rouge_l` | `rouge-score` | summarization |
| `f1_token` | stdlib | extraction |
| `regex_validity` | `re` (stdlib) | regex |
| `code_execution` | `subprocess` (stdlib) | code |
| `semantic_similarity` | `sentence-transformers` | feedback |
| `length_ratio_score` | stdlib | feedback (complément) |

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
               (plus fiable, ~3× plus d'appels API ; active le score hybride)
--resume       Reprend un run depuis un fichier checkpoint JSON
```

---

## 🧪 Tests

```bash
# Lancer toute la suite (111 tests)
pytest

# Un module en détail
pytest tests/unit/test_evaluator.py -v
pytest tests/unit/test_metrics.py -v

# Avec couverture
pytest --tb=short -q
```

La suite couvre :

| Fichier | Ce qui est testé |
|---|---|
| `test_evaluator.py` | `_parse_score` (JSON, clamping, markdown fences) + `evaluate` simulé avec `AsyncMock` |
| `test_cost_tracker.py` | Accumulation, alertes budget, `track_from_usage`, `summary` |
| `test_utils.py` | `parse_json_response` + `deduplicate` |
| `test_metrics.py` | 50 tests : toutes les métriques, lazy-loading `SentenceTransformer` (mocké), timeout subprocess boucle infinie |

Aucun appel API réel : `anthropic.AsyncAnthropic` et `SentenceTransformer` sont intégralement mockés.

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
│   ├── evaluator.py        ← Score hybride : LLM-judge + métriques déterministes (alpha)
│   ├── optimizer.py        ← Mutation / croisement des meilleurs prompts
│   ├── loop.py             ← Orchestration async, checkpoint, sélection
│   ├── metrics.py          ← 7 métriques déterministes/sémantiques + DATASET_METRICS
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
│       ├── test_utils.py       ← Tests de parse_json_response et deduplicate
│       └── test_metrics.py     ← 50 tests des métriques déterministes (sans réseau)
│
├── results/                ← Runs JSON + checkpoints (gitignorés)
│
├── config.yml              ← Configuration du score hybride (alpha global + overrides)
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
- [x] Métriques automatiques complémentaires (ROUGE-L, F1 token, similarité sémantique)
- [ ] Test cases intégrés par dataset pour les métriques `regex` et `code`
- [ ] Chargement automatique des overrides alpha depuis `config.yml` au démarrage
- [ ] Support de prompts système vs prompts utilisateur

---

## 📄 Licence

MIT — Baptiste Mistral
