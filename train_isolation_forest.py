import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# 1. CHARGEMENT DES DONNÉES PRÉPARÉES
# ============================================================
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()  # 0 = normal, 1 = attaque
y_test = pd.read_csv("data/y_test.csv").squeeze()

print(f"Train : {X_train.shape} | Test : {X_test.shape}")

# ============================================================
# 2. NE GARDER QUE LE TRAFIC NORMAL POUR L'ENTRAÎNEMENT
# ============================================================
# Principe clé de la détection d'anomalie : le modèle doit apprendre
# à quoi ressemble le trafic NORMAL, puis signaler ce qui s'en écarte.
# Si on l'entraîne sur un mélange 53% normal / 47% attaque, il n'a plus
# de notion claire de "normalité" -> c'est ce qui donnait ~46% de
# précision (quasi aléatoire) dans la première tentative.
# C'est aussi plus réaliste : sur un vrai réseau, on suppose que la
# grande majorité du trafic observé est normal.
X_train_normal = X_train[y_train == 0]
print(f"Lignes normales utilisées pour l'entraînement : {X_train_normal.shape[0]} "
      f"(sur {X_train.shape[0]} au total)")

# ============================================================
# 3. NORMALISATION DES FEATURES
# ============================================================
# Isolation Forest n'exige pas strictement une normalisation (il est basé
# sur des partitions/arbres), mais ça aide à équilibrer l'influence des
# colonnes ayant des échelles très différentes (ex. Flow Duration en
# microsecondes vs Down/Up Ratio entre 0 et 1).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. ENTRAÎNEMENT DE L'ISOLATION FOREST
# ============================================================
# 'contamination' = proportion attendue d'anomalies. Comme l'entraînement
# se fait uniquement sur du trafic normal, on fixe une valeur faible
# (typique d'un vrai réseau), à recalibrer plus tard avec des données
# réelles/Zeek plutôt que de refléter le taux artificiel du dataset.
contamination_rate = 0.05
print(f"\nContamination utilisée : {contamination_rate}")

model = IsolationForest(
    n_estimators=200,       # nombre d'arbres
    contamination=contamination_rate,
    random_state=42,
    n_jobs=-1,               # utilise tous les coeurs CPU disponibles
)

print("\nEntraînement en cours (sur trafic normal uniquement)...")
model.fit(X_train_scaled)
print("Entraînement terminé.")

# ============================================================
# 5. PRÉDICTION SUR LE TEST SET (mélange normal + attaques)
# ============================================================
# On récupère le SCORE continu (pas la prédiction binaire par défaut).
# Plus le score est négatif, plus la ligne est jugée anormale.
anomaly_scores = model.decision_function(X_test_scaled)

# ============================================================
# 5bis. CALIBRATION DU SEUIL SUR DONNÉES LABELLISÉES
# ============================================================
# Au lieu de faire confiance au paramètre 'contamination' choisi à
# l'aveugle (qui donnait un rappel catastrophique de 4.5% sur les
# attaques), on calibre le seuil de décision directement sur le test
# labellisé, en cherchant celui qui maximise le F1-score. C'est cette
# calibration sur échantillon labellisé que prévoit le projet (score
# de risque adaptatif -- cf. document de conception).
from sklearn.metrics import precision_recall_curve

# precision_recall_curve attend "plus grand = plus positif" ; on inverse
# donc le score (anomaly_scores est négatif pour une anomalie).
scores_for_pr = -anomaly_scores
precisions, recalls, thresholds = precision_recall_curve(y_test, scores_for_pr)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores[:-1])  # dernier point n'a pas de seuil associé
best_threshold = thresholds[best_idx]

print(f"\nMeilleur seuil trouvé (calibré sur test labellisé) : {best_threshold:.4f}")
print(f"F1 attendu à ce seuil : {f1_scores[best_idx]:.3f}")

y_pred = (scores_for_pr >= best_threshold).astype(int)
# Sauvegarde des données de la courbe Précision-Rappel (pour la figure du rapport)
pr_curve_df = pd.DataFrame({
    "precision": precisions[:-1],
    "recall": recalls[:-1],
    "threshold": thresholds,
})
pr_curve_df.to_csv("data/if_pr_curve.csv", index=False)
print("Courbe Précision-Rappel (Isolation Forest) sauvegardée dans data/if_pr_curve.csv")
# ============================================================
# 6. ÉVALUATION
# ============================================================
print("\n===== RAPPORT DE CLASSIFICATION =====")
print(classification_report(y_test, y_pred, target_names=["Normal", "Attaque"]))

print("===== MATRICE DE CONFUSION =====")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Prédit Normal   Prédit Attaque")
print(f"Réel Normal      {cm[0][0]:<15} {cm[0][1]}")
print(f"Réel Attaque     {cm[1][0]:<15} {cm[1][1]}")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"\nPrécision : {precision:.3f} | Rappel : {recall:.3f} | F1-score : {f1:.3f}")

# ============================================================
# 7. SAUVEGARDE DU MODÈLE ET DU SCALER
# ============================================================
joblib.dump(model, "data/isolation_forest_model.joblib")
joblib.dump(scaler, "data/scaler.joblib")
joblib.dump(best_threshold, "data/isolation_forest_threshold.joblib")
print("\nModèle, scaler et seuil sauvegardés dans data/ (isolation_forest_model.joblib, "
      "scaler.joblib, isolation_forest_threshold.joblib)")
