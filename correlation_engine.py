import pandas as pd
import numpy as np
import joblib
import os
import json
import time
from datetime import datetime

_t0 = time.time()
def checkpoint(label):
    print(f"[{time.time() - _t0:6.1f}s] {label}", flush=True)

checkpoint("Script started")

# ============================================================
# 1. CHARGEMENT DES DONNÉES ET DES MODÈLES
# ============================================================
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()
y_test_multi = pd.read_csv("data/y_test_multiclass.csv").squeeze()
meta_test = pd.read_csv("data/meta_test.csv")

test_df = pd.concat([X_test.reset_index(drop=True), meta_test.reset_index(drop=True)], axis=1)
test_df = test_df.loc[:, ~test_df.columns.duplicated()]
test_df["is_attack"] = y_test.values          # vérité terrain -- évaluation interne seulement
test_df["attack_type"] = y_test_multi.values  # idem

print(f"Test : {test_df.shape}")
checkpoint("Data loaded")

# ============================================================
# 1bis. FILTRAGE BLOCKLIST — avant tout calcul Isolation Forest
# ============================================================

ACTIONS_LOG_PATH = "data/actions_log.json"

def get_active_blocked_ips():
    if not os.path.exists(ACTIONS_LOG_PATH):
        return set()
    with open(ACTIONS_LOG_PATH, "r") as f:
        log = json.load(f)
    now = datetime.utcnow()
    blocked = set()
    for entry in log:
        if entry.get("active"):
            expiry = entry.get("expiry")
            if expiry is None or datetime.fromisoformat(expiry) > now:
                blocked.add(entry["source_ip"])
    return blocked

blocked_ips = get_active_blocked_ips()
mask_not_blocked = ~test_df["Source IP"].isin(blocked_ips)

n_before = len(test_df)
test_df = test_df[mask_not_blocked].reset_index(drop=True)
X_test = X_test[mask_not_blocked.values].reset_index(drop=True)

print(f"Filtrage blocklist : {n_before - len(test_df)} flux rejetés "
      f"({len(blocked_ips)} IP actuellement bloquées) — non transmis à Isolation Forest.")
checkpoint("Blocklist filter done")


# ============================================================
# 2. PREUVE 1 -- SCORE D'ANOMALIE PAR FLUX (Isolation Forest)
# ============================================================
if_model = joblib.load("data/isolation_forest_model.joblib")
if_scaler = joblib.load("data/scaler.joblib")
if_threshold = joblib.load("data/isolation_forest_threshold.joblib")

X_test_scaled = if_scaler.transform(X_test)
test_df["if_anomaly_score"] = -if_model.decision_function(X_test_scaled)
test_df["if_is_anomalous_flow"] = (test_df["if_anomaly_score"] >= if_threshold).astype(int)
checkpoint("Isolation Forest scoring done")

# ============================================================
# 3. GROUPEMENT -- FENÊTRE DYNAMIQUE NON CHEVAUCHANTE PAR (Source IP, Destination IP)
# ============================================================
# [!] Terminologie précise : ceci n'est PAS une vraie "sliding window" au
# sens classique (qui glisserait en continu, flux par flux, en générant
# des fenêtres qui se chevauchent). C'est une fenêtre DYNAMIQUE ANCRÉE :
# le premier flux non encore traité d'un couple (source, destination)
# ouvre une fenêtre de 60 secondes ; tous les flux qui tombent dedans y
# sont rattachés ; la fenêtre suivante démarre au premier flux situé
# après la fin de la précédente (pas de chevauchement).
#
# [PERF] Implémentation en UNE seule passe sur le tableau trié, sans
# passer par un groupby().apply() par paire (Source IP, Destination IP).
# À pleine échelle (blocklist vide), le nombre de paires distinctes peut
# dépasser 100 000 (trafic web normal = beaucoup de destinations externes
# différentes) ; un groupby Python par paire devient alors le vrai goulot
# d'étranglement à cause du coût fixe par groupe de pandas, pas de la
# taille des données elle-même. Ici, une seule boucle sur le tableau
# trié (numpy pur, pas de sous-DataFrame créée par groupe) traite les
# ~500k lignes en quelques secondes.
rule_config = joblib.load("data/rule_engine_config.joblib")

test_df["Timestamp_parsed"] = pd.to_datetime(test_df["Timestamp"], errors="coerce")
test_df = test_df.dropna(subset=["Timestamp_parsed"])
test_df = test_df.sort_values(["Source IP", "Destination IP", "Timestamp_parsed"]).reset_index(drop=False)
checkpoint(f"Sorted {len(test_df)} rows, starting single-pass windowing")

WINDOW = pd.Timedelta(seconds=60)
window_td64 = np.timedelta64(60, "s")

src_arr = test_df["Source IP"].to_numpy()
dst_arr = test_df["Destination IP"].to_numpy()
ts_arr = test_df["Timestamp_parsed"].to_numpy()
n = len(test_df)

window_start_arr = np.empty(n, dtype=ts_arr.dtype)
current_start = ts_arr[0]
window_start_arr[0] = current_start

for i in range(1, n):
    new_pair = (src_arr[i] != src_arr[i - 1]) or (dst_arr[i] != dst_arr[i - 1])
    if new_pair or ts_arr[i] >= current_start + window_td64:
        current_start = ts_arr[i]
    window_start_arr[i] = current_start
    if i % 100000 == 0:
        checkpoint(f"  windowing progress: {i}/{n} rows")

test_df["time_bin"] = window_start_arr
test_df = test_df.set_index("index")
test_df.index.name = None
checkpoint("Windowing complete")

GROUP_KEYS = ["Source IP", "Destination IP", "time_bin"]

# ============================================================
# 3bis. PREUVE CONTEXTUELLE -- BASELINE PAR MACHINE (Source IP)
# ============================================================
baseline_profile = joblib.load("data/baseline_profile.joblib")
global_mean, global_std = joblib.load("data/baseline_global_stats.joblib")
baseline_features = joblib.load("data/baseline_features.joblib")

X_train_for_baseline = pd.read_csv("data/X_train.csv")
y_train_for_baseline = pd.read_csv("data/y_train.csv").squeeze()
meta_train_for_baseline = pd.read_csv("data/meta_train.csv")

train_normal = pd.concat(
    [X_train_for_baseline.reset_index(drop=True), meta_train_for_baseline.reset_index(drop=True)], axis=1
)
train_normal = train_normal.loc[:, ~train_normal.columns.duplicated()]
train_normal = train_normal[y_train_for_baseline.values == 0]

MATURITY_THRESHOLD = 50
n_observations_per_ip = train_normal.groupby("Source IP").size()
confidence_per_ip = (n_observations_per_ip / MATURITY_THRESHOLD).clip(upper=1.0)

baseline_test = X_test.loc[test_df.index, baseline_features].copy()
baseline_test["Source IP"] = test_df["Source IP"].values
profile_reset = baseline_profile.reset_index()
merged_baseline = baseline_test.merge(profile_reset, on="Source IP", how="left")

mean_cols = [f"{f}_mean" for f in baseline_features]
std_cols = [f"{f}_std" for f in baseline_features]
for f in baseline_features:
    merged_baseline[f"{f}_mean"] = merged_baseline[f"{f}_mean"].fillna(global_mean[f])
    merged_baseline[f"{f}_std"] = merged_baseline[f"{f}_std"].fillna(global_std[f])

values = merged_baseline[baseline_features].to_numpy()
means = merged_baseline[mean_cols].to_numpy()
stds = merged_baseline[std_cols].to_numpy()
stds_safe = np.where(stds == 0, 1e-6, stds)
z_matrix = np.abs((values - means) / stds_safe)

test_df["baseline_deviation"] = z_matrix.max(axis=1)
test_df["host_confidence"] = test_df["Source IP"].map(confidence_per_ip).fillna(0.0)
test_df["host_n_observations"] = test_df["Source IP"].map(n_observations_per_ip).fillna(0).astype(int)
checkpoint("Baseline deviation computed")

# ============================================================
# 4. RÈGLES CALCULÉES AU NIVEAU DU GROUPE (pas du flux individuel)
# ============================================================
group_stats = test_df.groupby(GROUP_KEYS).agg(
    n_flows=("if_anomaly_score", "count"),
    n_if_anomalies=("if_is_anomalous_flow", "sum"),
    if_anomaly_mean=("if_anomaly_score", "mean"),
    if_anomaly_max=("if_anomaly_score", "max"),
    n_distinct_dst_ports=("Destination Port", "nunique"),
    long_connection_triggered=("Flow Duration", lambda s: bool((s >= rule_config["DURATION_THRESHOLD"]).max())),
    large_flow_triggered=("Total Length of Fwd Packets", lambda s: bool((s >= rule_config["VOLUME_THRESHOLD"]).max())),
    baseline_deviation_mean=("baseline_deviation", "mean"),
    baseline_deviation_max=("baseline_deviation", "max"),
    host_confidence=("host_confidence", "first"),
    host_n_observations=("host_n_observations", "first"),
    n_true_attacks=("is_attack", "sum"),
    attack_types_present=("attack_type", lambda s: sorted(set(s[s != "BENIGN"]))),
    dominant_attack_type=("attack_type", lambda s: (
        s[s != "BENIGN"].mode().iloc[0] if not s[s != "BENIGN"].empty else "BENIGN"
    )),
).reset_index()
checkpoint(f"group_stats built: {len(group_stats)} groups")

group_stats["port_scan_triggered"] = (group_stats["n_distinct_dst_ports"] >= rule_config["PORT_SCAN_THRESHOLD"])

brute_group = test_df[test_df["Destination Port"].isin(rule_config["SENSITIVE_PORTS"])].groupby(
    ["Source IP", "Destination IP", "Destination Port", "time_bin"]
).size().reset_index(name="n_attempts")
brute_group["brute_force_triggered"] = brute_group["n_attempts"] >= rule_config["BRUTE_FORCE_THRESHOLD"]
brute_flag_per_group = brute_group.groupby(["Source IP", "Destination IP", "time_bin"])["brute_force_triggered"].any().reset_index()

group_stats = group_stats.merge(brute_flag_per_group, on=["Source IP", "Destination IP", "time_bin"], how="left")
group_stats["brute_force_triggered"] = group_stats["brute_force_triggered"].fillna(False)
checkpoint("Brute-force flag merged")

# ============================================================
# 4ter. DÉCLENCHEURS DE SÉVÉRITÉ INDÉPENDANTS DE LA TAILLE DU GROUPE
# ============================================================
# Un groupe à 1 ou 2 flux ne peut jamais atteindre "n_if_anomalies >= 10".
# Ces déclencheurs utilisent le score MAX du groupe -> fonctionnent même
# pour une fenêtre à un seul flux, sans ignorer un signal fort d'IF ou
# de la baseline juste parce que la fenêtre est petite.
IF_SEVERITY_THRESHOLD = if_threshold          # même seuil calibré (F1) qu'Isolation Forest
BASELINE_SEVERITY_THRESHOLD = 3.0             # 3 écarts-types = anomalie statistique forte
MIN_HOST_CONFIDENCE_FOR_BASELINE = 0.3        # évite les faux positifs "IP jamais vue"

group_stats["if_severity_triggered"] = group_stats["if_anomaly_max"] >= IF_SEVERITY_THRESHOLD
group_stats["baseline_severity_triggered"] = (
    (group_stats["baseline_deviation_max"] >= BASELINE_SEVERITY_THRESHOLD)
    & (group_stats["host_confidence"] >= MIN_HOST_CONFIDENCE_FOR_BASELINE)
)

# ============================================================
# 4quinquies. DÉTECTION DE BEACONING (C2 discret : faible volume, périodique)
# — vectorisé via groupby().agg() / .shift(), pas de boucle par groupe —
# ============================================================
BEACON_MIN_WINDOWS = 4          # au moins 4 fenêtres distinctes vers la même destination
BEACON_MAX_AVG_FLOWS = 3        # peu de flux par fenêtre (comportement discret)
BEACON_MAX_INTERVAL_CV = 0.5    # intervalles entre fenêtres réguliers (faible variance)

gs_sorted = group_stats.sort_values(["Source IP", "Destination IP", "time_bin"]).copy()
gs_sorted["prev_time"] = gs_sorted.groupby(["Source IP", "Destination IP"])["time_bin"].shift(1)
gs_sorted["gap_seconds"] = (gs_sorted["time_bin"] - gs_sorted["prev_time"]).dt.total_seconds()

pair_stats = gs_sorted.groupby(["Source IP", "Destination IP"]).agg(
    n_windows=("time_bin", "count"),
    avg_flows_per_window=("n_flows", "mean"),
    gap_mean=("gap_seconds", "mean"),
    gap_std=("gap_seconds", "std"),
).reset_index()

pair_stats["gap_mean"] = pair_stats["gap_mean"].fillna(0)
pair_stats["gap_std"] = pair_stats["gap_std"].fillna(0)
pair_stats["interval_cv"] = np.where(
    pair_stats["gap_mean"] > 0,
    pair_stats["gap_std"] / pair_stats["gap_mean"],
    np.inf
)

pair_stats["beaconing_triggered"] = (
    (pair_stats["n_windows"] >= BEACON_MIN_WINDOWS)
    & (pair_stats["avg_flows_per_window"] <= BEACON_MAX_AVG_FLOWS)
    & (pair_stats["interval_cv"] <= BEACON_MAX_INTERVAL_CV)
)

group_stats = group_stats.merge(
    pair_stats[["Source IP", "Destination IP", "beaconing_triggered"]],
    on=["Source IP", "Destination IP"], how="left"
)
group_stats["beaconing_triggered"] = group_stats["beaconing_triggered"].fillna(False)
checkpoint("Beaconing detection done")

# ============================================================
# 4quater. SCORE DE RISQUE COMPOSITE (Module 5 du plan) — niveau groupe
# ============================================================
# Reprend la même logique que risk_score_engine.py (poids IF/règles/
# baseline, confiance appliquée à la baseline), mais calculée ici au
# niveau du GROUPE (incident) plutôt qu'au niveau du flux individuel,
# pour que le score arrive directement dans incident_evidence_records.csv
# et donc dans /candidates, sans script séparé à appeler.
# [!] Placé APRÈS le calcul de brute_force_triggered ci-dessus, car
# rules_component en a besoin -- sinon KeyError.
X_train_for_norm = pd.read_csv("data/X_train.csv")
X_train_scaled_for_norm = if_scaler.transform(X_train_for_norm)
if_train_scores = -if_model.decision_function(X_train_scaled_for_norm)
if_p1, if_p99 = np.percentile(if_train_scores, [1, 99])
checkpoint("Training-set percentiles computed")

if_component = np.clip(
    (group_stats["if_anomaly_mean"] - if_p1) / (if_p99 - if_p1 + 1e-9), 0, 1
)
rules_component = (
    group_stats["port_scan_triggered"].astype(int)
    + group_stats["brute_force_triggered"].astype(int)
    + group_stats["long_connection_triggered"].astype(int)
    + group_stats["large_flow_triggered"].astype(int)
    + group_stats["beaconing_triggered"].astype(int)
) / 5.0
baseline_deviation_norm = np.clip(group_stats["baseline_deviation_mean"] / 10.0, 0, 1)

W_IF, W_RULES, W_BASELINE = 0.40, 0.30, 0.30
effective_w_baseline = W_BASELINE * group_stats["host_confidence"]
weight_sum = W_IF + W_RULES + effective_w_baseline

group_stats["risk_score"] = (
    W_IF * if_component + W_RULES * rules_component + effective_w_baseline * baseline_deviation_norm
) / weight_sum * 100
checkpoint("Risk score computed")

def _risk_class(score):
    if score < 30:
        return "faible"
    elif score < 60:
        return "moyen"
    elif score < 80:
        return "élevé"
    else:
        return "critique"

group_stats["risk_class"] = group_stats["risk_score"].apply(_risk_class)

# ============================================================
# 4bis. PROFIL D'HÔTE -- LECTURE HUMAINE DE LA CONFIANCE BASELINE
# ============================================================
def host_profile_label(confidence, n_obs):
    if confidence >= 0.8:
        return "Connu et stable"
    elif confidence >= 0.3:
        return "Partiellement connu"
    elif n_obs > 0:
        return "Peu observé"
    else:
        return "Jamais vu en entraînement"

group_stats["host_profile"] = group_stats.apply(
    lambda r: host_profile_label(r["host_confidence"], r["host_n_observations"]), axis=1
)
checkpoint("Host profile labels assigned")

group_stats["true_incident"] = (group_stats["n_true_attacks"] > 0).astype(int)
group_stats = group_stats.rename(columns={"time_bin": "window_start"})
group_stats["window_end"] = group_stats["window_start"] + WINDOW

print(f"\nNombre de dossiers d'incident candidats (fenêtres dynamiques ancrées, Source IP x Destination IP) : {len(group_stats)}")

# ============================================================
# 5. APERÇU -- SANS SCORE COMPOSITE, JUSTE LE TABLEAU DE PREUVES BRUT
# ============================================================
print("\n===== EXEMPLE DE DOSSIERS (triés par nombre d'anomalies IF, top 5) =====")
top5 = group_stats.sort_values("n_if_anomalies", ascending=False).head(5)
print(top5[["Source IP", "Destination IP", "window_start", "window_end", "n_flows", "n_if_anomalies",
            "n_distinct_dst_ports", "port_scan_triggered", "brute_force_triggered",
            "host_confidence", "host_profile", "true_incident", "attack_types_present"]].to_string(index=False))

# ============================================================
# 6. VALIDATION INTERNE (pas transmise à l'agent)
# ============================================================
correlation_if = group_stats["n_if_anomalies"].corr(group_stats["true_incident"])
correlation_scan = group_stats["port_scan_triggered"].astype(int).corr(group_stats["true_incident"])
print(f"\nCorrélation (nb anomalies IF vs vrai incident) : {correlation_if:.3f}")
print(f"Corrélation (scan déclenché vs vrai incident) : {correlation_scan:.3f}")

# ============================================================
# 7. SAUVEGARDE
# ============================================================
evidence_for_agent = group_stats.drop(columns=["n_true_attacks", "attack_types_present", "dominant_attack_type", "true_incident"])
evidence_for_agent.to_csv("data/incident_evidence_records.csv", index=False)
group_stats.to_csv("data/incident_evidence_records_with_ground_truth.csv", index=False)

print("\nDossiers de preuves sauvegardés :")
print("  - data/incident_evidence_records.csv (destiné à l'agent, sans vérité terrain)")
print("  - data/incident_evidence_records_with_ground_truth.csv (pour évaluation interne)")
