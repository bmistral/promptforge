#!/usr/bin/env python3
"""
Hook PostToolUse — met à jour les docstrings Python après chaque Edit/Write.
Reçoit l'input de l'outil via stdin (JSON), modifie le fichier en place si nécessaire.
"""

import json
import os
import re
import sys

import anthropic

# Répertoires surveillés (relatifs à la racine du projet)
WATCHED_DIRS = ("core/", "datasets/")

SYSTEM_PROMPT = """Tu es un expert Python spécialisé dans la documentation de code.
Ton rôle est de mettre à jour les docstrings d'un fichier Python qui vient d'être modifié
pour qu'ils reflètent fidèlement le code actuel.

Règles strictes :
- Modifie UNIQUEMENT les docstrings (module, classes, méthodes, fonctions)
- N'ajoute pas de docstrings là où il n'y en a pas déjà, sauf si une fonction publique n'en a aucune
- Ne touche pas à la logique, aux commentaires inline (#), ni aux noms de variables
- Conserve exactement la même structure et indentation
- Si le code est déjà bien documenté, retourne-le identique
- Réponds UNIQUEMENT avec le contenu complet du fichier Python, sans bloc markdown ni explication
"""


def should_watch(file_path: str) -> bool:
    """Retourne True si le fichier doit être surveillé."""
    try:
        rel = os.path.relpath(file_path)
    except ValueError:
        return False
    return rel.endswith(".py") and any(rel.startswith(d) for d in WATCHED_DIRS)


def update_docs(file_path: str) -> None:
    """Lit le fichier, appelle Claude pour mettre à jour les docstrings, réécrit si besoin."""
    with open(file_path, encoding="utf-8") as f:
        original = f.read()

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Voici le fichier `{os.path.basename(file_path)}` qui vient d'être modifié.\n"
                f"Mets à jour ses docstrings si nécessaire :\n\n```python\n{original}\n```"
            ),
        }],
    )

    updated = response.content[0].text.strip()

    # Nettoyer les backticks si Claude en a quand même ajouté
    if updated.startswith("```"):
        updated = re.sub(r"^```(?:python)?\n?", "", updated)
        updated = re.sub(r"\n?```$", "", updated).strip()

    if updated and updated != original:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"[doc-sync] ✓ Docstrings mis à jour : {file_path}", file=sys.stderr)
    else:
        print(f"[doc-sync] — Aucun changement nécessaire : {file_path}", file=sys.stderr)


def main() -> None:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        sys.exit(0)

    # Compatibilité : l'input peut être le tool_input directement ou wrappé
    tool_input = data.get("tool_input", data)
    file_path = tool_input.get("file_path") or tool_input.get("path", "")

    if not file_path or not should_watch(file_path):
        sys.exit(0)

    if not os.path.isfile(file_path):
        sys.exit(0)

    update_docs(file_path)


if __name__ == "__main__":
    main()
