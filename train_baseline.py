import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()
meta_train = pd.read_csv("data/meta_train.csv")
meta_test = pd.read_csv("data/meta_test.csv")

print(f"Train : {X_train.shape} | Test : {X_test.shape}")

# ============================================================
# 2. CHOIX DES FEATURES POUR LA BASELINE
# ============================================================
# Noms adaptés au dataset officiel CICIDS2017 (avec "s" à Packets,
# "Backward" au lieu de "Bwd" dans ce nom précis).
baseline_features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
]
baseline_features = [f for f in baseline_features if f in X_train.columns]
print(f"\nFeatures utilisées pour la baseline : {baseline_features}")

# ============================================================
# 3. CONSTRUCTION DU PROFIL "NORMAL" PAR MACHINE (Source IP)
# ============================================================
train_normal = X_train[baseline_features].copy()
train_normal["Source IP"] = meta_train["Source IP"].values
train_normal = train_normal[y_train.values == 0]

profile = train_normal.groupby("Source IP")[baseline_features].agg(["mean", "std"])
profile.columns = ["_".join(col) for col in profile.columns]
profile = profile.fillna(0)

print(f"\nNombre de machines profilées : {profile.shape[0]}")

global_mean = train_normal[baseline_features].mean()
global_std = train_normal[baseline_features].std()

# ============================================================
# 4. CALCUL DU Z-SCORE (VECTORISÉ)
# ============================================================
test_df = X_test[baseline_features].copy()
test_df["Source IP"] = meta_test["Source IP"].values

profile_reset = profile.reset_index()
merged = test_df.merge(profile_reset, on="Source IP", how="left")

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
z_scores = z_matrix.max(axis=1)

print(f"\nZ-scores calculés pour {len(z_scores)} lignes.")

# ============================================================
# 5. CALIBRATION DU SEUIL SUR DONNÉES LABELLISÉES
# ============================================================
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

precisions, recalls, thresholds = precision_recall_curve(y_test, z_scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]

print(f"\nMeilleur seuil (z-score) trouvé : {best_threshold:.3f}")
print(f"F1 attendu à ce seuil : {f1_scores[best_idx]:.3f}")

y_pred_baseline = (z_scores >= best_threshold).astype(int)

# ============================================================
# 6. ÉVALUATION
# ============================================================
print("\n===== RAPPORT DE CLASSIFICATION (Baseline statistique) =====")
print(classification_report(y_test, y_pred_baseline, target_names=["Normal", "Attaque"]))

cm = confusion_matrix(y_test, y_pred_baseline)
print("===== MATRICE DE CONFUSION =====")
print(f"                 Prédit Normal   Prédit Attaque")
print(f"Réel Normal      {cm[0][0]:<15} {cm[0][1]}")
print(f"Réel Attaque     {cm[1][0]:<15} {cm[1][1]}")

# ============================================================
# 7. SAUVEGARDE
# ============================================================
joblib.dump(profile, "data/baseline_profile.joblib")
joblib.dump((global_mean, global_std), "data/baseline_global_stats.joblib")
joblib.dump(best_threshold, "data/baseline_threshold.joblib")
joblib.dump(baseline_features, "data/baseline_features.joblib")
print("\nProfil, stats globales, seuil et liste de features sauvegardés dans data/")
