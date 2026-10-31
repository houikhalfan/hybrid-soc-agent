from fastapi import FastAPI
import pandas as pd
import joblib
import json
import os
from datetime import datetime, timedelta
import subprocess


def _iptables_block(ip: str):
    if not ip:
        return False, "no IP provided"
    result = subprocess.run(
        ["iptables", "-C", "INPUT", "-s", ip, "-j", "DROP"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return True, "already blocked"  # rule already exists, avoid duplicates
    result = subprocess.run(
        ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"],
        capture_output=True, text=True
    )
    return result.returncode == 0, result.stderr


def _iptables_unblock(ip: str):
    if not ip:
        return False, "no IP provided"
    result = subprocess.run(
        ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"],
        capture_output=True, text=True
    )
    return result.returncode == 0, result.stderr


app = FastAPI(title="SOC Detection API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
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


@app.get("/")
def health_check():
    return {"status": "ok", "message": "SOC API is running"}


@app.get("/candidates")
def get_candidates(limit: int = 20, only_attacks: bool = True):
    evidence = pd.read_csv("data/incident_evidence_records.csv")

    # --- filtre blocklist, avant tout traitement ---
    # (défense en profondeur : correlation_engine.py filtre déjà en amont
    # avant Isolation Forest ; ceci rattrape une IP bloquée APRÈS la
    # dernière génération du CSV, entre deux exécutions du pipeline)
    blocked = set(get_blocked_ips()["blocked_ips"])
    evidence = evidence[~evidence["Source IP"].isin(blocked)]

    candidates = evidence[
    (evidence["port_scan_triggered"] == True)
    | (evidence["brute_force_triggered"] == True)
    | (evidence["long_connection_triggered"] == True)
    | (evidence["large_flow_triggered"] == True)
    | (evidence["if_severity_triggered"] == True)
    | (evidence["baseline_severity_triggered"] == True)
    | (evidence["beaconing_triggered"] == True)
].copy()

    clf_model = joblib.load("data/attack_classifier_model.joblib")
    clf_scaler = joblib.load("data/attack_classifier_scaler.joblib")
    clf_features = joblib.load("data/attack_classifier_features.joblib")

    for col in clf_features:
        if candidates[col].dtype == bool:
            candidates[col] = candidates[col].astype(int)

    X = candidates[clf_features]
    X_scaled = clf_scaler.transform(X)
    preds = clf_model.predict(X_scaled)
    probs = clf_model.predict_proba(X_scaled)
    class_names = list(clf_model.classes_)

    candidates["predicted_attack_type"] = preds
    candidates["predicted_confidence"] = [
        p[class_names.index(pred)] for p, pred in zip(probs, preds)
    ]
    candidates["predicted_attack_type"] = candidates["predicted_attack_type"].str.replace("\x96", "", regex=False)

    candidates["mitre_technique"] = candidates["predicted_attack_type"].map(MITRE_MAPPING).fillna("Non cartographié")

    if only_attacks:
        candidates = candidates[candidates["predicted_attack_type"] != "BENIGN"]

    candidates = candidates.sort_values("window_start", ascending=False).head(limit)

    return candidates.to_dict(orient="records")


@app.get("/candidates/{index}")
def get_candidate_by_index(index: int, only_attacks: bool = True):
    """Renvoie un seul incident (utile pour un traitement un par un dans n8n)."""
    all_candidates = get_candidates(limit=100000, only_attacks=only_attacks)
    if index < 0 or index >= len(all_candidates):
        return {"error": "index hors limites"}
    return all_candidates[index]


ACTIONS_LOG_PATH = "data/actions_log.json"

# Garde-fous du confinement automatique (Module 9.2 du plan) :
MAX_AUTO_BLOCKS_PER_HOUR = 3   # circuit breaker
AUTO_BLOCK_DURATION_HOURS = 2  # réversibilité bornée (2 à 8h dans le plan)


def _load_actions_log():
    if not os.path.exists(ACTIONS_LOG_PATH):
        return []
    with open(ACTIONS_LOG_PATH, "r") as f:
        return json.load(f)


def _save_actions_log(log):
    os.makedirs(os.path.dirname(ACTIONS_LOG_PATH), exist_ok=True)
    with open(ACTIONS_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)


@app.get("/blocked-ips")
def get_blocked_ips():
    """IP actuellement bloquées : actives et non expirées."""
    log = _load_actions_log()
    now = datetime.utcnow()
    active_ips = set()
    for entry in log:
        if entry.get("active"):
            expiry = entry.get("expiry")
            if expiry is None or datetime.fromisoformat(expiry) > now:
                active_ips.add(entry["source_ip"])
    return {"blocked_ips": sorted(active_ips)}


def _is_blocked(ip: str) -> bool:
    """Petit utilitaire partagé : vrai si cette IP est actuellement bloquée."""
    if not ip:
        return False
    return ip in get_blocked_ips()["blocked_ips"]


@app.get("/action/log")
def get_actions_log():
    """Renvoie l'historique complet des actions (pour le dashboard)."""
    return _load_actions_log()


@app.post("/action/mark-reminded")
def mark_reminded(payload: dict):
    """Marque un blocage comme ayant reçu son rappel 30 min avant expiration."""
    source_ip = payload.get("source_ip")
    log = _load_actions_log()
    for entry in log:
        if entry.get("source_ip") == source_ip and entry.get("active") and entry.get("trigger") == "automatique":
            entry["reminder_sent"] = True
    _save_actions_log(log)
    return {"status": "rappel marqué", "source_ip": source_ip}


@app.post("/action/execute")
def execute_validated_action(payload: dict):
    source_ip = payload.get("source_ip")
    action = payload.get("action")

    if action in ("bloquer", "isoler"):
        success, detail = _iptables_block(source_ip)
        if not success:
            return {"status": "échec", "reason": f"iptables error: {detail}"}

    log = _load_actions_log()
    entry = {
        "source_ip": source_ip,
        "action": action,
        "incident_type": payload.get("incident_type"),
        "trigger": "validation_humaine",
        "validated_by": payload.get("validated_by", "inconnu"),
        "timestamp": datetime.utcnow().isoformat(),
        "expiry": None,
        "active": action in ("bloquer", "isoler"),
    }
    log.append(entry)
    _save_actions_log(log)
    return {"status": "exécuté", "entry": entry}


@app.post("/action/auto-block")
def auto_block(payload: dict):
    """
    Confinement automatique déclenché en l'absence d'analyste (Module 9.2
    du plan). Toujours borné dans le temps et journalisé, jamais permanent.
    """
    log = _load_actions_log()

    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    recent_auto_blocks = [
        e for e in log
        if e.get("trigger") == "automatique"
        and datetime.fromisoformat(e["timestamp"]) >= one_hour_ago
    ]
    if len(recent_auto_blocks) >= MAX_AUTO_BLOCKS_PER_HOUR:
        return {
            "status": "refusé",
            "reason": f"Circuit breaker déclenché : {len(recent_auto_blocks)} confinements "
                      f"automatiques déjà effectués dans la dernière heure "
                      f"(limite : {MAX_AUTO_BLOCKS_PER_HOUR}). Bascule en alerte simple.",
        }

    source_ip = payload.get("source_ip")
    success, detail = _iptables_block(source_ip)
    if not success:
        return {"status": "échec", "reason": f"iptables error: {detail}"}

    duration_hours = payload.get("duration_hours", AUTO_BLOCK_DURATION_HOURS)
    expiry = datetime.utcnow() + timedelta(hours=duration_hours)
    entry = {
        "source_ip": source_ip,
        "action": "bloquer_temporaire",
        "incident_type": payload.get("incident_type"),
        "reason": payload.get("reason", "Aucun analyste disponible"),
        "trigger": "automatique",
        "timestamp": datetime.utcnow().isoformat(),
        "expiry": expiry.isoformat(),
        "active": True,
        "reminder_sent": False,
    }
    log.append(entry)
    _save_actions_log(log)
    return {"status": "confinement automatique activé", "entry": entry}


@app.post("/action/unblock")
def manual_unblock(payload: dict):
    """
    Lève un blocage manuellement, qu'il soit automatique (temporaire) ou
    permanent (humain). Utilisé par les boutons "Lever le blocage" /
    "Débloquer" du dashboard.
    """
    source_ip = payload.get("source_ip")
    validated_by = payload.get("validated_by", "inconnu")
    if not source_ip:
        return {"status": "échec", "reason": "source_ip manquant"}

    success, detail = _iptables_unblock(source_ip)
    if not success:
        return {"status": "échec", "reason": f"iptables error: {detail}"}

    log = _load_actions_log()
    found = False
    for entry in log:
        if entry.get("source_ip") == source_ip and entry.get("active"):
            entry["active"] = False
            entry["rolled_back_at"] = datetime.utcnow().isoformat()
            entry["rolled_back_by"] = validated_by
            found = True
    if not found:
        return {"status": "échec", "reason": f"aucun blocage actif trouvé pour {source_ip}"}

    _save_actions_log(log)
    return {"status": "débloqué", "source_ip": source_ip, "validated_by": validated_by}


@app.post("/action/rollback-expired")
def rollback_expired():
    """
    À appeler périodiquement (ex. toutes les 15 min via un Cron n8n).
    Désactive automatiquement tout confinement automatique expiré.
    """
    log = _load_actions_log()
    now = datetime.utcnow()
    rolled_back = []

    for entry in log:
        if (entry.get("active") and entry.get("expiry")
                and datetime.fromisoformat(entry["expiry"]) <= now):
            _iptables_unblock(entry["source_ip"])
            entry["active"] = False
            entry["rolled_back_at"] = now.isoformat()
            rolled_back.append(entry["source_ip"])

    _save_actions_log(log)
    return {"rolled_back": rolled_back, "count": len(rolled_back)}


INCIDENT_ANALYSIS_LOG_PATH = "data/incident_analysis_log.json"

def _load_incident_analysis():
    if not os.path.exists(INCIDENT_ANALYSIS_LOG_PATH):
        return []
    with open(INCIDENT_ANALYSIS_LOG_PATH, "r") as f:
        return json.load(f)

def _save_incident_analysis(log):
    os.makedirs(os.path.dirname(INCIDENT_ANALYSIS_LOG_PATH), exist_ok=True)
    with open(INCIDENT_ANALYSIS_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)

@app.post("/incidents/analysis")
def save_incident_analysis(payload: dict):
    """
    Enregistre la sortie du pipeline d'agents n8n. Si l'IP source est déjà
    bloquée au moment de l'enregistrement, on journalise quand même (pour
    l'audit — on ne perd jamais de trace), mais on marque l'entrée
    'superseded_by_block' pour que GET /incidents/analysis puisse la
    masquer par défaut sans la supprimer.
    """
    payload["saved_at"] = datetime.utcnow().isoformat()
    payload["superseded_by_block"] = _is_blocked(payload.get("source_ip"))
    log = _load_incident_analysis()
    log.append(payload)
    _save_incident_analysis(log)
    return {"status": "enregistré", "superseded_by_block": payload["superseded_by_block"]}

@app.get("/incidents/analysis")
def get_incident_analysis(include_blocked: bool = False):
    """
    Par défaut, masque les analyses dont la source est ACTUELLEMENT
    bloquée — qu'elle l'était déjà au moment de l'enregistrement, ou
    qu'elle l'est devenue depuis (ex. bloquée manuellement après coup).
    Passer include_blocked=true pour tout voir (utile en audit).
    """
    log = _load_incident_analysis()
    if include_blocked:
        return log
    blocked = set(get_blocked_ips()["blocked_ips"])
    return [entry for entry in log if entry.get("source_ip") not in blocked]
