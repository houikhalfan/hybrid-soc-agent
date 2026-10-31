import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. CHARGEMENT DES DOSSIERS DE PREUVES (avec vérité terrain)
# ============================================================
evidence = pd.read_csv("data/incident_evidence_records_with_ground_truth.csv")
print(f"Dossiers de preuves chargés : {len(evidence)}")

# ============================================================
# 2. SÉLECTION DU PÉRIMÈTRE D'ENTRAÎNEMENT
# ============================================================
# On entraîne UNIQUEMENT sur les incidents "candidats" (au moins un
# signal détecté) -- cohérent avec le pipeline réel : le classificateur
# n'est appelé qu'APRÈS le filtrage (Étape 6), jamais sur tout le trafic
# brut. Ça évite aussi un déséquilibre écrasant vers la classe BENIGN.
candidates = evidence[
    (evidence["port_scan_triggered"] == True)
    | (evidence["brute_force_triggered"] == True)
    | (evidence["long_connection_triggered"] == True)
    | (evidence["large_flow_triggered"] == True)
    | (evidence["if_severity_triggered"] == True)
    | (evidence["baseline_severity_triggered"] == True)
    | (evidence["beaconing_triggered"] == True)
].copy()

print(f"Incidents candidats (périmètre d'entraînement) : {len(candidates)}")
print("\nRépartition des types d'attaque dominants parmi les candidats :")
print(candidates["dominant_attack_type"].value_counts())

# ============================================================
# 3. NETTOYAGE -- RETIRER LES CLASSES TROP RARES POUR STRATIFIER
# ============================================================
# Certaines classes (ex. Heartbleed, Infiltration) ont très peu
# d'exemples au niveau incident -- un split stratifié échoue si une
# classe a moins de 2 exemples. On les garde à part, documentées comme
# limite (pas assez de données pour les classifier de façon fiable pour
# l'instant), plutôt que de les faire planter le script.
class_counts = candidates["dominant_attack_type"].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()
if rare_classes:
    print(f"\n[!] Classes trop rares pour être incluses (< 2 exemples) : {rare_classes}")
    candidates = candidates[~candidates["dominant_attack_type"].isin(rare_classes)]

# ============================================================
# 4. FEATURES ET LABEL
# ============================================================
feature_cols = [
    "n_flows", "n_if_anomalies", "if_anomaly_mean", "if_anomaly_max",
    "n_distinct_dst_ports", "long_connection_triggered", "large_flow_triggered",
    "baseline_deviation_mean", "baseline_deviation_max", "host_confidence",
    "host_n_observations", "port_scan_triggered", "brute_force_triggered",
    "beaconing_triggered",
]
# Conversion des booléens en 0/1 pour le modèle
for col in feature_cols:
    if candidates[col].dtype == bool:
        candidates[col] = candidates[col].astype(int)

X = candidates[feature_cols]
y = candidates["dominant_attack_type"]

print(f"\nNombre de classes à prédire : {y.nunique()}")

# ============================================================
# 5. SPLIT TRAIN / TEST (stratifié, niveau incident)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train : {len(X_train)} incidents | Test : {len(X_test)} incidents")

# ============================================================
# 6. ENTRAÎNEMENT -- RANDOM FOREST
# ============================================================
# Random Forest plutôt qu'un modèle linéaire : gère bien les features
# hétérogènes (booléens + continus), pas sensible à l'échelle, et donne
# une importance des variables interprétable (utile pour justifier les
# prédictions face à un jury).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",  # compense le déséquilibre entre classes rares/fréquentes
    random_state=42,
    n_jobs=-1,
)

print("\nEntraînement en cours...")
model.fit(X_train_scaled, y_train)
print("Entraînement terminé.")

# ============================================================
# 7. ÉVALUATION
# ============================================================
y_pred = model.predict(X_test_scaled)

print("\n===== RAPPORT DE CLASSIFICATION (type d'attaque, niveau incident) =====")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n===== IMPORTANCE DES VARIABLES =====")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.to_string())

# ============================================================
# 8. SAUVEGARDE
# ============================================================
joblib.dump(model, "data/attack_classifier_model.joblib")
joblib.dump(scaler, "data/attack_classifier_scaler.joblib")
joblib.dump(feature_cols, "data/attack_classifier_features.joblib")

print("\nModèle, scaler et liste de features sauvegardés dans data/")
print("  - data/attack_classifier_model.joblib")
print("  - data/attack_classifier_scaler.joblib")
print("  - data/attack_classifier_features.joblib")
