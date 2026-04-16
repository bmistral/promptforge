"""
Datasets d'exemples pour tester PromptForge sur différentes tâches.
"""

# ─── Résumé de texte ────────────────────────────────────────────────────────
SUMMARIZATION_TASK = "Résumer un texte en une phrase concise"

SUMMARIZATION_INITIAL_PROMPT = "Résume ce texte."

SUMMARIZATION_EXAMPLES = [
    {
        "input": "La photosynthèse est le processus par lequel les plantes convertissent la lumière solaire, l'eau et le dioxyde de carbone en glucose et en oxygène. Ce mécanisme fondamental permet aux végétaux de produire leur propre énergie et constitue la base de presque toutes les chaînes alimentaires sur Terre.",
        "expected_output": "La photosynthèse permet aux plantes de transformer lumière, eau et CO2 en énergie (glucose) et oxygène, fondant ainsi la quasi-totalité des chaînes alimentaires terrestres.",
    },
    {
        "input": "Le machine learning supervisé est une approche où un algorithme apprend à partir de données étiquetées. Le modèle identifie des patterns dans les données d'entraînement pour ensuite effectuer des prédictions sur de nouvelles données. Les exemples courants incluent la classification d'emails en spam/non-spam et la prédiction de prix immobiliers.",
        "expected_output": "Le ML supervisé entraîne des algorithmes sur des données étiquetées pour leur permettre de faire des prédictions sur de nouvelles données.",
    },
    {
        "input": "Les microplastiques sont des fragments de plastique inférieurs à 5mm qui contaminent désormais tous les écosystèmes de la planète, des profondeurs océaniques aux sommets montagneux. Leurs effets sur la santé humaine et la biodiversité font l'objet de recherches intensives.",
        "expected_output": "Les microplastiques, fragments plastiques < 5mm, ont envahi tous les écosystèmes mondiaux et leurs impacts sanitaires et écologiques sont activement étudiés.",
    },
]


# ─── Classification de sentiment ────────────────────────────────────────────
SENTIMENT_TASK = "Classifier le sentiment d'un avis client (positif / négatif / neutre)"

SENTIMENT_INITIAL_PROMPT = "Quel est le sentiment de ce texte ? Réponds par positif, négatif ou neutre."

SENTIMENT_EXAMPLES = [
    {
        "input": "Ce produit est absolument fantastique, je le recommande vivement à tout le monde !",
        "expected_output": "positif",
    },
    {
        "input": "Livraison correcte mais l'emballage était abîmé. Le produit fonctionne bien pour l'instant.",
        "expected_output": "neutre",
    },
    {
        "input": "Très déçu, le produit ne correspond pas du tout à la description. Je demande un remboursement.",
        "expected_output": "négatif",
    },
    {
        "input": "Service client réactif et professionnel, problème résolu en moins d'une heure.",
        "expected_output": "positif",
    },
]


# ─── Extraction d'informations ───────────────────────────────────────────────
EXTRACTION_TASK = "Extraire les entités clés (personne, organisation, lieu, date) d'un texte"

EXTRACTION_INITIAL_PROMPT = "Extrais les informations importantes de ce texte."

EXTRACTION_EXAMPLES = [
    {
        "input": "Elon Musk a annoncé le 15 mars 2024 que Tesla ouvrirait une nouvelle gigafactory à Berlin.",
        "expected_output": '{"personne": "Elon Musk", "organisation": "Tesla", "lieu": "Berlin", "date": "15 mars 2024"}',
    },
    {
        "input": "La BCE, dirigée par Christine Lagarde, a maintenu ses taux directeurs lors de sa réunion de janvier 2024 à Francfort.",
        "expected_output": '{"personne": "Christine Lagarde", "organisation": "BCE", "lieu": "Francfort", "date": "janvier 2024"}',
    },
]


# ─── Génération de code ──────────────────────────────────────────────────────
CODE_TASK = "Générer une fonction Python propre et documentée à partir d'une description"

CODE_INITIAL_PROMPT = "Écris du code Python pour faire ça :"

CODE_EXAMPLES = [
    {
        "input": "Une fonction qui calcule la moyenne d'une liste de nombres en ignorant les valeurs None",
        "expected_output": 'def calculate_mean(values: list) -> float:\n    """Calcule la moyenne en ignorant les None."""\n    clean = [v for v in values if v is not None]\n    return sum(clean) / len(clean) if clean else 0.0',
    },
    {
        "input": "Une fonction qui vérifie si une chaîne est un palindrome (insensible à la casse et aux espaces)",
        "expected_output": 'def is_palindrome(s: str) -> bool:\n    """Vérifie si s est un palindrome."""\n    cleaned = s.lower().replace(" ", "")\n    return cleaned == cleaned[::-1]',
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# NOUVEAUX DATASETS — classés par complexité croissante
# ═══════════════════════════════════════════════════════════════════════════════

# ─── [SIMPLE] Normalisation d'adresses ──────────────────────────────────────
ADDRESS_TASK = "Normaliser une adresse en format standard : Numéro Voie, Code_postal Ville, Pays"

ADDRESS_INITIAL_PROMPT = "Reformate cette adresse."

ADDRESS_EXAMPLES = [
    {
        "input": "15 rue de la paix, paris 75002 france",
        "expected_output": "15 rue de la Paix, 75002 Paris, France",
    },
    {
        "input": "bd haussmann 42 - 75009 PARIS",
        "expected_output": "42 boulevard Haussmann, 75009 Paris, France",
    },
    {
        "input": "8 AV. DES CHAMPS ELYSEES 75008 paris FRANCE",
        "expected_output": "8 avenue des Champs-Élysées, 75008 Paris, France",
    },
    {
        "input": "place du capitole 1, toulouse, 31000",
        "expected_output": "1 place du Capitole, 31000 Toulouse, France",
    },
]


# ─── [SIMPLE] Détection de langue ───────────────────────────────────────────
LANGUAGE_TASK = "Identifier la langue d'un texte et répondre uniquement avec le code ISO 639-1 en minuscules (fr, en, de, es, it…)"

LANGUAGE_INITIAL_PROMPT = "Quelle est la langue de ce texte ?"

LANGUAGE_EXAMPLES = [
    {
        "input": "The quick brown fox jumps over the lazy dog.",
        "expected_output": "en",
    },
    {
        "input": "Die Katze sitzt auf der Matte und schläft.",
        "expected_output": "de",
    },
    {
        "input": "Le soleil brille et les oiseaux chantent.",
        "expected_output": "fr",
    },
    {
        "input": "El cielo está despejado y hace mucho calor hoy.",
        "expected_output": "es",
    },
    {
        "input": "Il treno è in ritardo di venti minuti.",
        "expected_output": "it",
    },
]


# ─── [SIMPLE] Priorité de tickets support ───────────────────────────────────
TICKET_TASK = "Classer un ticket support en exactement un niveau : critique / haute / moyenne / basse"

TICKET_INITIAL_PROMPT = "Donne la priorité de ce ticket."

TICKET_EXAMPLES = [
    {
        "input": "Le site est complètement down, plus aucun client ne peut passer commande depuis 30 minutes.",
        "expected_output": "critique",
    },
    {
        "input": "L'API de paiement retourne des erreurs 500 de façon intermittente (environ 1 requête sur 10).",
        "expected_output": "haute",
    },
    {
        "input": "Le bouton 'Exporter en PDF' ne fonctionne pas sur Firefox, les autres navigateurs sont OK.",
        "expected_output": "moyenne",
    },
    {
        "input": "La couleur du bouton 'Annuler' ne correspond pas à la charte graphique (gris au lieu de rouge).",
        "expected_output": "basse",
    },
    {
        "input": "Impossible de se connecter pour l'ensemble des utilisateurs du compte entreprise Acme Corp.",
        "expected_output": "critique",
    },
]


# ─── [MEDIUM] Génération de regex ───────────────────────────────────────────
REGEX_TASK = "Générer une regex Python compilable (sans flags) qui valide exactement le format décrit, sans texte autour"

REGEX_INITIAL_PROMPT = "Écris une regex Python pour ce format."

REGEX_EXAMPLES = [
    {
        "input": "Un numéro de téléphone français mobile : commence par 06 ou 07, suivi de 8 chiffres",
        "expected_output": r"^0[67]\d{8}$",
    },
    {
        "input": "Un code couleur hexadécimal CSS, avec ou sans # initial, 6 chiffres hex",
        "expected_output": r"^#?[0-9a-fA-F]{6}$",
    },
    {
        "input": "Une adresse email basique : caractères alphanumériques et points avant @, domaine avec au moins un point",
        "expected_output": r"^[\w.+-]+@[\w-]+\.[\w.-]+$",
    },
    {
        "input": "Un code postal français : exactement 5 chiffres",
        "expected_output": r"^\d{5}$",
    },
    {
        "input": "Une date au format JJ/MM/AAAA",
        "expected_output": r"^\d{2}/\d{2}/\d{4}$",
    },
]


# ─── [MEDIUM] Reformulation de feedback RH (méthode SBI) ────────────────────
FEEDBACK_TASK = (
    "Transformer un feedback brut en feedback constructif selon la méthode SBI : "
    "Situation (contexte précis), Behavior (comportement observé), Impact (effet concret). "
    "Format : une seule phrase structurée SBI, ton professionnel et bienveillant."
)

FEEDBACK_INITIAL_PROMPT = "Reformule ce feedback pour qu'il soit constructif."

FEEDBACK_EXAMPLES = [
    {
        "input": "Tu es toujours en retard et ça énerve tout le monde.",
        "expected_output": "En réunion d'équipe du lundi matin (S), tes arrivées après l'heure de début perturbent le démarrage (B) et créent une frustration visible chez les autres participants (I).",
    },
    {
        "input": "Tes présentations sont nulles, personne ne comprend ce que tu veux dire.",
        "expected_output": "Lors de la revue trimestrielle de la semaine dernière (S), les slides contenaient beaucoup de données sans synthèse claire (B), ce qui a rendu difficile la prise de décision pour l'équipe (I).",
    },
    {
        "input": "Tu ne communiques jamais avec le reste de l'équipe sur l'avancement de tes tâches.",
        "expected_output": "Sur le sprint en cours (S), les mises à jour de statut dans Jira n'ont pas été renseignées (B), ce qui a bloqué deux collègues qui attendaient tes livrables pour avancer (I).",
    },
    {
        "input": "Tu prends toujours les décisions tout seul sans consulter les autres.",
        "expected_output": "Lors du choix de l'architecture de la nouvelle feature la semaine passée (S), la décision a été prise et implémentée sans consultation de l'équipe (B), ce qui a entraîné deux jours de refactoring pour aligner les approches (I).",
    },
]


# ─── [COMPLEXE] Audit de sécurité de code ───────────────────────────────────
SECURITY_TASK = (
    "Analyser un extrait de code pour identifier les vulnérabilités de sécurité. "
    "Répondre en JSON avec : faille (type OWASP), ligne (numéro), sévérité (1-10), "
    "remédiation (correction concrète en code). Si plusieurs failles, retourner un tableau JSON."
)

SECURITY_INITIAL_PROMPT = "Analyse ce code pour trouver des problèmes de sécurité."

SECURITY_EXAMPLES = [
    {
        "input": (
            "def login(username, password):\n"
            "    query = f\"SELECT * FROM users WHERE name='{username}' AND pwd='{password}'\"\n"
            "    result = db.execute(query)\n"
            "    return result.fetchone() is not None"
        ),
        "expected_output": (
            '{"faille": "SQL Injection (OWASP A03)", "ligne": 2, "sévérité": 9, '
            '"remédiation": "db.execute(\'SELECT * FROM users WHERE name=? AND pwd=?\', (username, password))"}'
        ),
    },
    {
        "input": (
            "import subprocess\n"
            "\n"
            "def convert_file(filename):\n"
            "    cmd = 'convert ' + filename + ' output.pdf'\n"
            "    subprocess.run(cmd, shell=True)\n"
            "    return 'output.pdf'"
        ),
        "expected_output": (
            '{"faille": "Command Injection (OWASP A03)", "ligne": 4, "sévérité": 9, '
            '"remédiation": "subprocess.run([\'convert\', filename, \'output.pdf\'], shell=False)"}'
        ),
    },
    {
        "input": (
            "SECRET_KEY = 'abc123supersecret'\n"
            "DEBUG = True\n"
            "\n"
            "def get_user(user_id):\n"
            "    return db.query(f'SELECT * FROM users WHERE id={user_id}')"
        ),
        "expected_output": (
            '[{"faille": "Hardcoded Secret (OWASP A02)", "ligne": 1, "sévérité": 8, '
            '"remédiation": "SECRET_KEY = os.environ[\'SECRET_KEY\']"}, '
            '{"faille": "SQL Injection (OWASP A03)", "ligne": 5, "sévérité": 9, '
            '"remédiation": "db.query(\'SELECT * FROM users WHERE id=?\', (user_id,))"}]'
        ),
    },
]


# ─── Registre des datasets ───────────────────────────────────────────────────
DATASETS = {
    # Datasets originaux
    "summarization": {
        "task": SUMMARIZATION_TASK,
        "initial_prompt": SUMMARIZATION_INITIAL_PROMPT,
        "examples": SUMMARIZATION_EXAMPLES,
    },
    "sentiment": {
        "task": SENTIMENT_TASK,
        "initial_prompt": SENTIMENT_INITIAL_PROMPT,
        "examples": SENTIMENT_EXAMPLES,
    },
    "extraction": {
        "task": EXTRACTION_TASK,
        "initial_prompt": EXTRACTION_INITIAL_PROMPT,
        "examples": EXTRACTION_EXAMPLES,
    },
    "code": {
        "task": CODE_TASK,
        "initial_prompt": CODE_INITIAL_PROMPT,
        "examples": CODE_EXAMPLES,
    },
    # Nouveaux datasets — simples
    "address": {
        "task": ADDRESS_TASK,
        "initial_prompt": ADDRESS_INITIAL_PROMPT,
        "examples": ADDRESS_EXAMPLES,
    },
    "language": {
        "task": LANGUAGE_TASK,
        "initial_prompt": LANGUAGE_INITIAL_PROMPT,
        "examples": LANGUAGE_EXAMPLES,
    },
    "ticket": {
        "task": TICKET_TASK,
        "initial_prompt": TICKET_INITIAL_PROMPT,
        "examples": TICKET_EXAMPLES,
    },
    # Nouveaux datasets — medium
    "regex": {
        "task": REGEX_TASK,
        "initial_prompt": REGEX_INITIAL_PROMPT,
        "examples": REGEX_EXAMPLES,
    },
    "feedback": {
        "task": FEEDBACK_TASK,
        "initial_prompt": FEEDBACK_INITIAL_PROMPT,
        "examples": FEEDBACK_EXAMPLES,
    },
    # Nouveau dataset — complexe
    "security": {
        "task": SECURITY_TASK,
        "initial_prompt": SECURITY_INITIAL_PROMPT,
        "examples": SECURITY_EXAMPLES,
    },
}
