import pandas as pd
import numpy as np
import joblib
import requests

OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL = "qwen2.5:3b"

# ============================================================
# 1. CHARGEMENT DES DOSSIERS DE PREUVES (sans vérité terrain)
# ============================================================
evidence = pd.read_csv("data/incident_evidence_records.csv")
print(f"Dossiers de preuves chargés : {len(evidence)}")

# ============================================================
# 2. CHARGEMENT DU CLASSIFICATEUR DE TYPE D'ATTAQUE
# ============================================================
clf_model = joblib.load("data/attack_classifier_model.joblib")
clf_scaler = joblib.load("data/attack_classifier_scaler.joblib")
clf_features = joblib.load("data/attack_classifier_features.joblib")

# Mapping simple type d'attaque -> technique MITRE ATT&CK (bonus rapide,
# à enrichir plus tard avec une vraie base RAG).
MITRE_MAPPING = {
    "PortScan": "T1046 - Network Service Discovery",
    "DDoS": "T1498 - Network Denial of Service",
    "DoS Hulk": "T1498 - Network Denial of Service",
    "DoS GoldenEye": "T1498 - Network Denial of Service",
    "DoS slowloris": "T1499 - Endpoint Denial of Service",
    "DoS Slowhttptest": "T1499 - Endpoint Denial of Service",
    "FTP-Patator": "T1110 - Brute Force",
    "SSH-Patator": "T1110 - Brute Force",
    "Bot": "T1071 - Application Layer Protocol (C2)",
    "Web Attack  Brute Force": "T1110.001 - Password Guessing",
    "Web Attack  XSS": "T1059 - Command and Scripting Interpreter",
    "Web Attack  Sql Injection": "T1190 - Exploit Public-Facing Application",
    "Infiltration": "T1190 - Exploit Public-Facing Application",
    "Heartbleed": "T1190 - Exploit Public-Facing Application",
    "BENIGN": "N/A",
}

# Impact ET actions ANCRÉS (déterministes), pas inventés par le LLM à
# chaque appel -- le LLM les cite et les contextualise, il ne les invente
# jamais. Cohérent avec le principe de repli déterministe du projet.
IMPACT_MAPPING = {
    "PortScan": "Reconnaissance réseau -- prépare souvent une attaque ultérieure ciblée.",
    "DDoS": "Indisponibilité potentielle du service ciblé (déni de service).",
    "DoS Hulk": "Indisponibilité potentielle du service ciblé (déni de service applicatif).",
    "DoS GoldenEye": "Indisponibilité potentielle du service ciblé (déni de service applicatif).",
    "DoS slowloris": "Épuisement des connexions serveur, indisponibilité progressive.",
    "DoS Slowhttptest": "Épuisement des connexions serveur, indisponibilité progressive.",
    "FTP-Patator": "Risque de compromission d'identifiants FTP si l'attaque réussit.",
    "SSH-Patator": "Risque de compromission d'accès SSH (accès root/administrateur possible).",
    "Bot": "Possible machine déjà compromise communiquant avec un serveur de commande (C2).",
    "Web Attack  Brute Force": "Risque de compromission de comptes applicatifs web.",
    "Web Attack  XSS": "Risque d'exécution de script côté client, vol de session possible.",
    "Infiltration": "Mouvement latéral ou exploitation active -- impact potentiellement élevé.",
    "Heartbleed": "Fuite possible de données sensibles en mémoire (clés, identifiants).",
    "BENIGN": "Aucun impact attendu -- probable faux positif.",
}

ACTION_PLAYBOOK = {
    "PortScan": ["Surveiller la source", "Vérifier si scan interne autorisé (audit/inventaire)"],
    "DDoS": ["Activer limitation de débit (rate limiting)", "Bloquer temporairement si volume confirmé"],
    "DoS Hulk": ["Activer limitation de débit (rate limiting)", "Bloquer temporairement si volume confirmé"],
    "DoS GoldenEye": ["Activer limitation de débit (rate limiting)", "Bloquer temporairement si volume confirmé"],
    "DoS slowloris": ["Vérifier configuration serveur (timeouts)", "Bloquer temporairement la source"],
    "DoS Slowhttptest": ["Vérifier configuration serveur (timeouts)", "Bloquer temporairement la source"],
    "FTP-Patator": ["Bloquer temporairement la source", "Forcer rotation des mots de passe FTP"],
    "SSH-Patator": ["Bloquer temporairement la source", "Vérifier logs d'authentification SSH réussie"],
    "Bot": ["Isoler la machine du réseau", "Analyser la machine pour malware"],
    "Web Attack  Brute Force": ["Bloquer temporairement la source", "Activer CAPTCHA/2FA si disponible"],
    "Web Attack  XSS": ["Investiguer manuellement le payload", "Vérifier logs applicatifs web"],
    "Infiltration": ["Isoler la machine immédiatement", "Escalade prioritaire vers un analyste senior"],
    "Heartbleed": ["Patcher immédiatement le service vulnérable", "Régénérer les certificats/clés"],
    "BENIGN": ["Aucune action -- surveiller si récurrence"],
}

# ============================================================
# 3. SÉLECTION DES INCIDENTS À INVESTIGUER
# ============================================================
candidates = evidence[
    (evidence["port_scan_triggered"] == True)
    | (evidence["brute_force_triggered"] == True)
    | (evidence["long_connection_triggered"] == True)
    | (evidence["large_flow_triggered"] == True)
    | (evidence["n_if_anomalies"] >= 10)
].copy()

print(f"Incidents candidats à investiguer (au moins un signal) : {len(candidates)}")

N_DEMO = 30
# Diversité : au plus 2 fenêtres par couple (Source IP, Destination IP),
# pour éviter que la démo ne montre que la même paire répétée -- plus
# représentatif de la variété réelle des incidents candidats.
diversified = (
    candidates.sort_values("n_if_anomalies", ascending=False)
    .groupby(["Source IP", "Destination IP"])
    .head(2)
)
sample = diversified.sort_values("n_if_anomalies", ascending=False).head(N_DEMO)

# ============================================================
# 4. PRÉDICTION DU TYPE D'ATTAQUE (classificateur, pas une heuristique)
# ============================================================
for col in clf_features:
    if sample[col].dtype == bool:
        sample[col] = sample[col].astype(int)

X_sample = sample[clf_features]
X_sample_scaled = clf_scaler.transform(X_sample)

predicted_types = clf_model.predict(X_sample_scaled)
predicted_probs = clf_model.predict_proba(X_sample_scaled)
class_names = clf_model.classes_

sample["predicted_attack_type"] = predicted_types
sample["predicted_confidence"] = [
    probs[list(class_names).index(pred)] for probs, pred in zip(predicted_probs, predicted_types)
]
sample["mitre_technique"] = sample["predicted_attack_type"].map(MITRE_MAPPING).fillna("Non cartographié")

print("\n===== PRÉDICTIONS DU CLASSIFICATEUR =====")
print(sample[["Source IP", "Destination IP", "predicted_attack_type", "predicted_confidence", "mitre_technique"]].to_string(index=False))

# ============================================================
# 5. AGENT D'INVESTIGATION -- CONSTRUCTION DU CONTEXTE STRUCTURÉ
# ============================================================
def build_evidence_text(row):
    """Traduit une ligne du dossier de preuves (+ prédiction) en texte lisible pour le LLM."""
    anomaly_ratio = 100 * row["n_if_anomalies"] / row["n_flows"]
    pred_type = row["predicted_attack_type"]

    # -- Détection de DÉSACCORD entre les méthodes --
    # Important : ne jamais laisser un désaccord passer silencieusement
    # (ex. règles déclenchées MAIS classificateur dit BENIGN). On le
    # signale explicitement pour que l'agent le commente, plutôt que de
    # produire un rapport incohérent.
    any_rule_triggered = row['port_scan_triggered'] or row['brute_force_triggered'] or \
                          row['long_connection_triggered'] or row['large_flow_triggered']
    disagreement = None
    if pred_type == "BENIGN" and any_rule_triggered:
        disagreement = ("Le classificateur prédit BENIGN, mais au moins une règle s'est déclenchée. "
                         "Ce désaccord doit être signalé -- possible faux positif des règles, "
                         "ou attaque non représentée dans les données d'entraînement du classificateur.")
    elif pred_type != "BENIGN" and not any_rule_triggered and anomaly_ratio < 20:
        disagreement = ("Le classificateur prédit une attaque mais peu d'autres signaux (règles, "
                         "ratio d'anomalies IF) la confirment -- confiance à interpréter avec prudence.")

    impact = IMPACT_MAPPING.get(pred_type, "Impact non cartographié pour ce type.")
    actions = ACTION_PLAYBOOK.get(pred_type, ["Investiguer manuellement (aucun playbook prédéfini)"])
    actions_text = "\n".join(f"  - {a}" for a in actions)

    # -- Niveau de suspicion SUGGÉRÉ, ancré sur des règles simples --
    # Le LLM local (petit modèle) calibre mal ce niveau seul (ex. il a pu
    # dire "Moyen" pour un DoS à 100% de confiance et impact sévère --
    # incohérent). On lui donne donc une suggestion de départ, qu'il peut
    # ajuster UNIQUEMENT s'il a une preuve contraire forte à citer.
    if pred_type == "BENIGN":
        suggested_suspicion = "Faible"
    elif row["predicted_confidence"] >= 0.8 and pred_type in [
        "DDoS", "DoS Hulk", "DoS GoldenEye", "Infiltration", "Heartbleed", "Bot"
    ]:
        suggested_suspicion = "Élevé"
    elif row["predicted_confidence"] >= 0.6:
        suggested_suspicion = "Moyen à Élevé"
    else:
        suggested_suspicion = "Moyen"

    text = f"""
Fenêtre temporelle : {row['window_start']} -> {row['window_end']}
Source : {row['Source IP']}
Destination : {row['Destination IP']}

Prédiction du classificateur de type d'attaque (modèle entraîné, pas une supposition) :
- Type probable : {pred_type}
- Confiance du modèle : {row['predicted_confidence']*100:.0f}%
- Technique MITRE ATT&CK associée : {row['mitre_technique']}

Impact potentiel (référence connue pour ce type, pas une invention) :
{impact}

Actions standards associées à ce type d'attaque (playbook de référence) :
{actions_text}

Niveau de suspicion suggéré par défaut (ajustable UNIQUEMENT si tu as une
preuve contraire forte, sinon reprends-le tel quel) : {suggested_suspicion}

Preuves collectées :
- Nombre de flux dans cette fenêtre : {row['n_flows']}
- Flux jugés anormaux par Isolation Forest : {row['n_if_anomalies']}
- Pourcentage de flux anormaux : {anomaly_ratio:.1f} %
- Score d'anomalie maximal : {row['if_anomaly_max']:.3f}
- Ports de destination distincts contactés : {row['n_distinct_dst_ports']}
- Règle Scan de ports déclenchée : {"Oui" if row['port_scan_triggered'] else "Non"}
- Règle Brute force déclenchée : {"Oui" if row['brute_force_triggered'] else "Non"}
- Règle Connexion longue déclenchée : {"Oui" if row['long_connection_triggered'] else "Non"}
- Règle Volume anormal déclenchée : {"Oui" if row['large_flow_triggered'] else "Non"}

Contexte de la machine source :
- Déviation par rapport à son comportement habituel : {row['baseline_deviation_mean']:.2f} (max : {row['baseline_deviation_max']:.2f})
- Profil de l'hôte : {row['host_profile']}
- Nombre d'observations historiques : {row['host_n_observations']}
""".strip()

    if disagreement:
        text += f"\n\n[!] DÉSACCORD ENTRE MÉTHODES DÉTECTÉ :\n{disagreement}"

    return text


def call_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False,
              "options": {"num_predict": 300}},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"]


AGENT_SYSTEM_PROMPT = """
Tu es un analyste SOC expérimenté.

Tu reçois :
- un dossier de preuves réseau,
- le type d'attaque prédit par un classificateur supervisé,
- le niveau de confiance,
- le mapping MITRE ATT&CK.

Ta mission est d'analyser la cohérence entre toutes les preuves.

Réponds STRICTEMENT sous le format suivant :

Type d'attaque probable :
...

Justification :
(Explique pourquoi ce type d'attaque est cohérent avec les preuves observées : anomalies Isolation Forest, règles déclenchées, comportement de l'hôte, etc.)

Cohérence entre les méthodes :
(Le classificateur est-il cohérent avec les preuves ? Oui / Non et pourquoi.)

Impact potentiel :
...

Priorité :
Faible / Moyenne / Élevée / Critique

Décision SOC :
Surveiller /
Investiguer manuellement /
Bloquer temporairement

Conclusion :
(Résume en 2 ou 3 phrases l'incident.)

Actions recommandées :
- ...
- ...

Ne jamais inventer de nouvelles preuves.
Base-toi uniquement sur les informations fournies.
Réponse concise (maximum 180 mots).
"""

# ============================================================
# 6. EXÉCUTION DE L'AGENT
# ============================================================
results = []
for idx, row in sample.iterrows():
    evidence_text = build_evidence_text(row)
    full_prompt = f"{AGENT_SYSTEM_PROMPT}\n\n=== DOSSIER DE PREUVES ===\n{evidence_text}\n\n=== TON ANALYSE ==="

    print(f"\n{'='*60}")
    print(f"Investigation : {row['Source IP']} -> {row['Destination IP']} ({row['window_start']})")
    print(f"Type prédit : {row['predicted_attack_type']} ({row['predicted_confidence']*100:.0f}%)")
    print(f"{'='*60}")

    try:
        analysis = call_ollama(full_prompt)
        print(analysis)
    except Exception as e:
        analysis = f"[Erreur d'appel au LLM local : {e}]"
        print(analysis)

    results.append({
        "source_ip": row["Source IP"],
        "destination_ip": row["Destination IP"],
        "time_window": row["window_start"],
        "n_flows": row["n_flows"],
        "n_if_anomalies": row["n_if_anomalies"],
        "predicted_attack_type": row["predicted_attack_type"],
        "predicted_confidence": row["predicted_confidence"],
        "mitre_technique": row["mitre_technique"],
        "port_scan_triggered": row["port_scan_triggered"],
        "brute_force_triggered": row["brute_force_triggered"],
        "host_profile": row["host_profile"],
        "agent_analysis": analysis,
    })

# ============================================================
# 7. SAUVEGARDE POUR LE DASHBOARD
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv("data/investigated_incidents.csv", index=False)
print(f"\n\n{len(results)} incidents investigués sauvegardés dans data/investigated_incidents.csv")
