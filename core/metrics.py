"""
Métriques déterministes et sémantiques pour l'évaluation de prompts PromptForge.

Chaque métrique prend des arguments spécifiques et retourne un float entre 0 et 1.
La fonction publique compute_deterministic_score agrège les métriques par dataset.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

# ─── Lazy-loading Sentence Transformers ──────────────────────────────────────

_sentence_model = None
SEMANTIC_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]


# ─── Métriques individuelles ──────────────────────────────────────────────────


def exact_match(generated: str, reference: str) -> float:
    """Normalise casse et espaces avant comparaison."""
    return 1.0 if generated.strip().lower() == reference.strip().lower() else 0.0


def regex_validity(generated: str, test_cases: list[dict]) -> float:
    """
    1. Tente re.compile(generated) — si échec, retourne 0.0
    2. Teste la regex sur chaque test_case : {"input": str, "should_match": bool}
    3. Retourne le taux de succès sur les test_cases.
    Si test_cases est vide et la regex compile, retourne 1.0.
    """
    generated = generated.strip()
    try:
        pattern = re.compile(generated)
    except re.error:
        return 0.0

    if not test_cases:
        return 1.0

    passed = 0
    for tc in test_cases:
        try:
            matched = bool(pattern.search(tc["input"]))
            if matched == tc["should_match"]:
                passed += 1
        except Exception:
            pass

    return passed / len(test_cases)


def code_execution(generated: str, test_cases: list[dict]) -> float:
    """
    Exécute le code généré dans un subprocess isolé avec timeout=5s.
    Chaque test_case : {"input": any, "expected_output": any}
    Retourne le taux de tests passés.
    Capture toutes les exceptions sans jamais crasher le process principal.
    """
    if not test_cases or not generated.strip():
        return 0.0

    harness = f"""\
import json as _json
import sys as _sys
import types as _types

_test_cases = {repr(test_cases)}
_code = {repr(generated)}

_namespace = {{}}
try:
    exec(_code, _namespace)
except Exception as _e:
    _sys.stdout.write(_json.dumps({{'passed': 0, 'total': len(_test_cases), 'error': str(_e)}}) + '\\n')
    _sys.exit(0)

_funcs = [_o for _n, _o in _namespace.items() if isinstance(_o, _types.FunctionType)]
if not _funcs:
    _sys.stdout.write(_json.dumps({{'passed': 0, 'total': len(_test_cases), 'error': 'no function found'}}) + '\\n')
    _sys.exit(0)

_func = _funcs[0]
_passed = 0
for _tc in _test_cases:
    try:
        _inp = _tc['input']
        _exp = _tc['expected_output']
        if isinstance(_inp, list):
            _result = _func(*_inp)
        elif _inp is None:
            _result = _func()
        else:
            _result = _func(_inp)
        if _result == _exp:
            _passed += 1
    except Exception:
        pass

_sys.stdout.write(_json.dumps({{'passed': _passed, 'total': len(_test_cases)}}) + '\\n')
"""

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(harness)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        stdout = result.stdout.strip()
        if not stdout:
            return 0.0
        data = json.loads(stdout)
        total = data.get("total", len(test_cases))
        passed = data.get("passed", 0)
        return float(passed) / total if total > 0 else 0.0

    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def rouge_l(generated: str, reference: str) -> float:
    """
    Utilise rouge_score.RougeScorer avec use_stemmer=True.
    Retourne rougeL.fmeasure.
    """
    if not generated.strip() or not reference.strip():
        return 0.0

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return float(scores["rougeL"].fmeasure)


def f1_token(generated: str, reference: str) -> float:
    """
    Tokenise par split() après normalisation casse.
    Calcule précision, recall, F1 sur l'intersection des tokens.
    Retourne F1 entre 0 et 1.
    """
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())

    if not gen_tokens and not ref_tokens:
        return 1.0
    if not gen_tokens or not ref_tokens:
        return 0.0

    intersection = gen_tokens & ref_tokens
    precision = len(intersection) / len(gen_tokens)
    recall = len(intersection) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def semantic_similarity(generated: str, reference: str) -> float:
    """
    Utilise paraphrase-multilingual-MiniLM-L12-v2.
    Lazy loading du modèle (chargé une seule fois à la première utilisation).
    Retourne cosine similarity entre 0 et 1.
    """
    global _sentence_model

    if SentenceTransformer is None:
        return 0.0

    try:
        import numpy as np

        if _sentence_model is None:
            _sentence_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

        embeddings = _sentence_model.encode([generated, reference])
        a = np.array(embeddings[0])
        b = np.array(embeddings[1])
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, min(1.0, similarity))
    except Exception:
        return 0.0


def length_ratio_score(generated: str, reference: str) -> float:
    """
    Retourne 1.0 si 1.2 <= ratio <= 3.0, sinon 0.5.
    ratio = len(generated.split()) / len(reference.split())
    Protège contre division par zéro.
    """
    ref_words = reference.split()
    if not ref_words:
        return 0.5
    ratio = len(generated.split()) / len(ref_words)
    return 1.0 if 1.2 <= ratio <= 3.0 else 0.5


# ─── Dataset → métriques mapping ─────────────────────────────────────────────

DATASET_METRICS: dict[str, list[dict]] = {
    "sentiment": [{"fn": exact_match, "weight": 1.0, "needs_test_cases": False}],
    "language": [{"fn": exact_match, "weight": 1.0, "needs_test_cases": False}],
    "ticket": [{"fn": exact_match, "weight": 1.0, "needs_test_cases": False}],
    "regex": [{"fn": regex_validity, "weight": 1.0, "needs_test_cases": True}],
    "code": [{"fn": code_execution, "weight": 1.0, "needs_test_cases": True}],
    "summarization": [{"fn": rouge_l, "weight": 1.0, "needs_test_cases": False}],
    "extraction": [{"fn": f1_token, "weight": 1.0, "needs_test_cases": False}],
    "feedback": [
        {"fn": semantic_similarity, "weight": 0.7, "needs_test_cases": False},
        {"fn": length_ratio_score, "weight": 0.3, "needs_test_cases": False},
    ],
    "address": [],   # LLM-judge seul pour l'instant
    "security": [],  # LLM-judge seul pour l'instant
}


def compute_deterministic_score(
    dataset_name: str,
    generated: str,
    reference: str,
    test_cases: list[dict] | None = None,
) -> float | None:
    """
    Retourne le score déterministe agrégé pour un dataset donné.
    
    Retourne None si le dataset n'a pas de métriques déterministes,
    ou si toutes les métriques applicables requièrent des test_cases non fournis.
    Agrège par moyenne pondérée des métriques actives du dataset.
    """
    metrics = DATASET_METRICS.get(dataset_name)
    if not metrics:
        return None

    total_weight = 0.0
    score = 0.0

    for m in metrics:
        fn = m["fn"]
        weight = m["weight"]
        needs_tc = m.get("needs_test_cases", False)

        if needs_tc and test_cases is None:
            continue  # Pas de test_cases → on ignore cette métrique

        try:
            s = fn(generated, test_cases) if needs_tc else fn(generated, reference)
            score += weight * s
            total_weight += weight
        except Exception:
            pass

    if total_weight == 0:
        return None

    return score / total_weight