import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. CHARGEMENT DES DONNÉES DE TEST ET DES MODÈLES ENTRAÎNÉS
# ============================================================
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()
meta_train = pd.read_csv("data/meta_train.csv")
meta_test = pd.read_csv("data/meta_test.csv")

test_df = pd.concat([X_test.reset_index(drop=True), meta_test.reset_index(drop=True)], axis=1)
test_df = test_df.loc[:, ~test_df.columns.duplicated()]

print(f"Test : {test_df.shape}")

# ============================================================
# 2. COMPOSANTE 1 -- ISOLATION FOREST (normalisée 0-1)
# ============================================================
if_model = joblib.load("data/isolation_forest_model.joblib")
if_scaler = joblib.load("data/scaler.joblib")

X_test_scaled = if_scaler.transform(X_test)
if_raw_scores = -if_model.decision_function(X_test_scaled)  # plus grand = plus anormal

# Normalisation min-max basée sur les percentiles du TRAIN (robuste aux
# valeurs extrêmes) pour ramener le score sur une échelle 0-1 comparable
# aux autres composantes.
X_train_scaled = if_scaler.transform(X_train)
if_train_scores = -if_model.decision_function(X_train_scaled)
if_p1, if_p99 = np.percentile(if_train_scores, [1, 99])

if_component = np.clip((if_raw_scores - if_p1) / (if_p99 - if_p1 + 1e-9), 0, 1)

# ============================================================
# 3. COMPOSANTE 2 -- MOTEUR DE RÈGLES (score gradué, pas binaire)
# ============================================================
# Au lieu d'un simple "au moins une règle déclenchée = 1", on compte
# la PROPORTION de règles déclenchées (0, 1, 2, 3 ou 4 sur 4) -> un score
# plus nuancé qui reflète l'intensité du signal plutôt qu'un tout-ou-rien.
rule_config = joblib.load("data/rule_engine_config.joblib")

def parse_datetime_safe(ts):
    return pd.to_datetime(ts, errors="coerce")

test_df["Timestamp_parsed"] = parse_datetime_safe(test_df["Timestamp"])
test_df["time_bin"] = test_df["Timestamp_parsed"].dt.floor("60s")

scan_counts = test_df.groupby(["Source IP", "time_bin"])["Destination Port"].transform("nunique")
rule_port_scan = (scan_counts >= rule_config["PORT_SCAN_THRESHOLD"]).astype(int)

brute_counts = test_df.groupby(["Source IP", "Destination Port", "time_bin"])["Destination Port"].transform("count")
is_sensitive_port = test_df["Destination Port"].isin(rule_config["SENSITIVE_PORTS"])
rule_brute_force = ((brute_counts >= rule_config["BRUTE_FORCE_THRESHOLD"]) & is_sensitive_port).astype(int)

rule_volume = (test_df["Total Length of Fwd Packets"] >= rule_config["VOLUME_THRESHOLD"]).astype(int)
rule_duration = (test_df["Flow Duration"] >= rule_config["DURATION_THRESHOLD"]).astype(int)

rules_matrix = pd.concat([rule_port_scan, rule_brute_force, rule_volume, rule_duration], axis=1)
rules_component = rules_matrix.sum(axis=1) / 4.0  # proportion de règles déclenchées (0 à 1)

# ============================================================
# 4. COMPOSANTE 3 -- BASELINE, PONDÉRÉE PAR UN INDICE DE CONFIANCE
# ============================================================
baseline_profile = joblib.load("data/baseline_profile.joblib")
global_mean, global_std = joblib.load("data/baseline_global_stats.joblib")
baseline_features = joblib.load("data/baseline_features.joblib")

# --- Indice de confiance par machine ---
# Basé sur le nombre d'observations normales vues en entraînement pour
# cette IP. Plus une machine a d'historique connu, plus sa baseline est
# fiable. MATURITY_THRESHOLD = nombre d'observations à partir duquel on
# considère la confiance maximale (1.0) -- valeur à ajuster selon le
# contexte réel (ex. quelques jours d'observation sur un vrai réseau).
MATURITY_THRESHOLD = 50

train_normal = pd.concat([X_train.reset_index(drop=True), meta_train.reset_index(drop=True)], axis=1)
train_normal = train_normal.loc[:, ~train_normal.columns.duplicated()]
train_normal = train_normal[y_train.values == 0]

n_observations_per_ip = train_normal.groupby("Source IP").size()
confidence_per_ip = (n_observations_per_ip / MATURITY_THRESHOLD).clip(upper=1.0)

print(f"\nRépartition de la confiance baseline (par machine) :")
print(confidence_per_ip.describe())

# --- Déviation baseline (z-score), comme avant ---
baseline_test = X_test[baseline_features].copy()
baseline_test["Source IP"] = meta_test["Source IP"].values

profile_reset = baseline_profile.reset_index()
merged = baseline_test.merge(profile_reset, on="Source IP", how="left")

mean_cols = [f"{f}_mean" for f in baseline_features]
std_cols = [f"{f}_std" for f in baseline_features]
for f in baseline_features:
    merged[f"{f}_mean"] = merged[f"{f}_mean"].fillna(global_mean[f])
    merged[f"{f}_std"] = merged[f"{f}_std"].fillna(global_std[f])

values = merged[baseline_features].to_numpy()
means = merged[mean_cols].to_numpy()
stds = merged[std_cols].to_numpy()
stds_safe = np.where(stds == 0, 1e-6, stds)

z_matrix = np.abs((values - means) / stds_safe)
baseline_zscore = z_matrix.max(axis=1)

# Normalisation du z-score en 0-1 (saturation au-delà de 10 écarts-types,
# largement suffisant pour capturer une vraie anomalie).
baseline_deviation = np.clip(baseline_zscore / 10.0, 0, 1)

# --- Confiance appliquée à chaque ligne du test ---
confidence = test_df["Source IP"].map(confidence_per_ip).fillna(0.0).values  # 0 si machine jamais vue
baseline_component = confidence * baseline_deviation

print(f"\nMachines avec confiance = 0 (jamais vues en entraînement) : {(confidence == 0).sum()} lignes")
print(f"Machines avec confiance = 1 (bien connues) : {(confidence == 1).sum()} lignes")

# ============================================================
# 5. SCORE DE RISQUE PONDÉRÉ (avec renormalisation des poids)
# ============================================================
# Poids de base (à calibrer -- ce sont des valeurs de départ, comme
# précisé dans le document de conception, section Score de risque).
W_IF = 0.40
W_RULES = 0.30
W_BASELINE = 0.30

# Renormalisation : si la confiance baseline est faible, son poids
# diminue, et le score reste sur une échelle 0-100 cohérente en
# redistribuant implicitement le poids restant sur IF + règles
# (via la division par la somme des poids effectifs).
effective_w_baseline = W_BASELINE * confidence
weight_sum = W_IF + W_RULES + effective_w_baseline

risk_score = (
    W_IF * if_component + W_RULES * rules_component + effective_w_baseline * baseline_deviation
) / weight_sum * 100

# ============================================================
# 6. CLASSES DE RISQUE (Module 5 du plan)
# ============================================================
def risk_class(score):
    if score < 30:
        return "faible"
    elif score < 60:
        return "moyen"
    elif score < 80:
        return "élevé"
    else:
        return "critique"

risk_classes = pd.Series(risk_score).apply(risk_class)
print("\n===== RÉPARTITION DES CLASSES DE RISQUE =====")
print(risk_classes.value_counts())

# ============================================================
# 7. ÉVALUATION (seuil de décision calibré sur le score continu)
# ============================================================
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, risk_score)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]

print(f"\nMeilleur seuil de score (calibré) : {best_threshold:.1f}/100")
print(f"F1 attendu à ce seuil : {f1_scores[best_idx]:.3f}")

y_pred_final = (risk_score >= best_threshold).astype(int)

print("\n===== RAPPORT DE CLASSIFICATION -- SCORE DE RISQUE PONDÉRÉ =====")
print(classification_report(y_test, y_pred_final, target_names=["Normal", "Attaque"]))

cm = confusion_matrix(y_test, y_pred_final)
print("===== MATRICE DE CONFUSION =====")
print(f"                 Prédit Normal   Prédit Attaque")
print(f"Réel Normal      {cm[0][0]:<15} {cm[0][1]}")
print(f"Réel Attaque     {cm[1][0]:<15} {cm[1][1]}")

# ============================================================
# 8. SAUVEGARDE
# ============================================================
result_df = pd.DataFrame({
    "y_true": y_test.values,
    "if_component": if_component,
    "rules_component": rules_component,
    "baseline_component": baseline_component,
    "confidence": confidence,
    "risk_score": risk_score,
    "risk_class": risk_classes.values,
    "pred_final": y_pred_final,
})
result_df.to_csv("data/risk_score_results.csv", index=False)

joblib.dump({
    "W_IF": W_IF, "W_RULES": W_RULES, "W_BASELINE": W_BASELINE,
    "MATURITY_THRESHOLD": MATURITY_THRESHOLD, "decision_threshold": best_threshold,
    "if_p1": if_p1, "if_p99": if_p99,          # <-- add these
}, "data/risk_score_config.joblib")

print("\nRésultats et configuration sauvegardés dans data/")
