Analyse tous les fichiers Python dans `core/` et `datasets/` et mets à jour leurs docstrings pour qu'ils reflètent fidèlement le code actuel.

Pour chaque fichier :
1. Lis son contenu
2. Vérifie que les docstrings de module, classes et méthodes/fonctions correspondent bien à ce que le code fait réellement
3. Mets à jour les docstrings obsolètes ou incomplets — en particulier les Args, Returns et les descriptions de comportement
4. N'ajoute pas de docstrings là où il n'y en a pas, sauf pour les fonctions publiques sans aucune documentation
5. Ne modifie pas la logique ni les commentaires inline

Commence par `core/models.py`, puis `core/generator.py`, `core/evaluator.py`, `core/optimizer.py`, `core/loop.py`, et enfin `datasets/examples.py`.

Après chaque fichier modifié, indique brièvement ce qui a changé.
