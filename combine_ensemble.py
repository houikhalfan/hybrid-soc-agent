import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ============================================================
# 1. CHARGEMENT DES DONNÉES DE TEST
# ============================================================
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()
meta_test = pd.read_csv("data/meta_test.csv")

test_df = pd.concat([X_test.reset_index(drop=True), meta_test.reset_index(drop=True)], axis=1)
test_df = test_df.loc[:, ~test_df.columns.duplicated()]  # évite les doublons Src/Dst Port

print(f"Test : {test_df.shape}")

# ============================================================
# 2. PRÉDICTION -- MOTEUR DE RÈGLES
# ============================================================
rule_config = joblib.load("data/rule_engine_config.joblib")

def parse_minute_second(ts):
    try:
        minute_str, second_str = ts.split(":")
        return int(minute_str) * 60 + float(second_str)
    except (ValueError, AttributeError):
        return np.nan

test_df["time_in_seconds"] = test_df["Timestamp"].apply(parse_minute_second)
test_df["time_bin"] = (test_df["time_in_seconds"] // 60).astype("Int64")

scan_counts = test_df.groupby(["Src IP dec", "time_bin"])["Dst Port"].transform("nunique")
rule_port_scan = (scan_counts >= rule_config["PORT_SCAN_THRESHOLD"]).astype(int)

brute_counts = test_df.groupby(["Src IP dec", "Dst Port", "time_bin"])["Dst Port"].transform("count")
is_sensitive_port = test_df["Dst Port"].isin(rule_config["SENSITIVE_PORTS"])
rule_brute_force = ((brute_counts >= rule_config["BRUTE_FORCE_THRESHOLD"]) & is_sensitive_port).astype(int)

rule_volume = (test_df["Total Length of Fwd Packet"] >= rule_config["VOLUME_THRESHOLD"]).astype(int)
rule_duration = (test_df["Flow Duration"] >= rule_config["DURATION_THRESHOLD"]).astype(int)

y_pred_rules = pd.concat([rule_port_scan, rule_brute_force, rule_volume, rule_duration], axis=1).max(axis=1)

# ============================================================
# 3. PRÉDICTION -- ISOLATION FOREST
# ============================================================
if_model = joblib.load("data/isolation_forest_model.joblib")
if_scaler = joblib.load("data/scaler.joblib")
if_threshold = joblib.load("data/isolation_forest_threshold.joblib")

X_test_scaled = if_scaler.transform(X_test)
if_scores = -if_model.decision_function(X_test_scaled)  # plus grand = plus anormal
y_pred_if = (if_scores >= if_threshold).astype(int)

# ============================================================
# 4. PRÉDICTION -- BASELINE STATISTIQUE
# ============================================================
baseline_profile = joblib.load("data/baseline_profile.joblib")
global_mean, global_std = joblib.load("data/baseline_global_stats.joblib")
baseline_threshold = joblib.load("data/baseline_threshold.joblib")
baseline_features = joblib.load("data/baseline_features.joblib")

baseline_test = X_test[baseline_features].copy()
baseline_test["Src IP dec"] = meta_test["Src IP dec"].values

profile_reset = baseline_profile.reset_index()
merged = baseline_test.merge(profile_reset, on="Src IP dec", how="left")

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
baseline_scores = z_matrix.max(axis=1)
y_pred_baseline = (baseline_scores >= baseline_threshold).astype(int)

# ============================================================
# 5. COMBINAISON (Module 3 du plan)
# ============================================================
# Ensemble ML : anomalie retenue seulement si Isolation Forest ET
# baseline concordent (réduit les faux positifs de chaque méthode prise
# seule -- cf. document de conception).
y_pred_ml_ensemble = ((y_pred_if == 1) & (y_pred_baseline == 1)).astype(int)

# Décision finale : alerte si le moteur de règles se déclenche, OU si
# l'ensemble ML (IF+baseline) est d'accord. Les règles catchent des
# signatures connues avec haute précision ; l'ensemble ML catche des
# comportements plus subtils/inconnus.
y_pred_final = ((y_pred_rules == 1) | (y_pred_ml_ensemble == 1)).astype(int)

# ============================================================
# 6. COMPARAISON DES MÉTHODES
# ============================================================
methods = {
    "Règles seules": y_pred_rules,
    "Isolation Forest seul": y_pred_if,
    "Baseline seule": y_pred_baseline,
    "Ensemble ML (IF ET Baseline)": y_pred_ml_ensemble,
    "COMBINÉ FINAL (Règles OU Ensemble ML)": y_pred_final,
}

print("\n===== COMPARAISON DES MÉTHODES =====")
print(f"{'Méthode':<40}{'Précision':<12}{'Rappel':<12}{'F1-score':<10}")
for name, preds in methods.items():
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    p = report["1"]["precision"]
    r = report["1"]["recall"]
    f1 = report["1"]["f1-score"]
    print(f"{name:<40}{p:<12.3f}{r:<12.3f}{f1:<10.3f}")

print("\n===== RAPPORT DÉTAILLÉ -- COMBINÉ FINAL =====")
print(classification_report(y_test, y_pred_final, target_names=["Normal", "Attaque"]))

cm = confusion_matrix(y_test, y_pred_final)
print("===== MATRICE DE CONFUSION -- COMBINÉ FINAL =====")
print(f"                 Prédit Normal   Prédit Attaque")
print(f"Réel Normal      {cm[0][0]:<15} {cm[0][1]}")
print(f"Réel Attaque     {cm[1][0]:<15} {cm[1][1]}")

# ============================================================
# 7. SAUVEGARDE
# ============================================================
result_df = pd.DataFrame({
    "y_true": y_test.values,
    "pred_rules": y_pred_rules.values,
    "pred_isolation_forest": y_pred_if,
    "pred_baseline": y_pred_baseline,
    "pred_ml_ensemble": y_pred_ml_ensemble,
    "pred_final": y_pred_final,
})
result_df.to_csv("data/combined_predictions.csv", index=False)
print("\nPrédictions combinées sauvegardées dans data/combined_predictions.csv")
