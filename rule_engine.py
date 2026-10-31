import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. CHARGEMENT DES DONNÉES (features + métadonnées + label)
# ============================================================
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()
meta_train = pd.read_csv("data/meta_train.csv")
meta_test = pd.read_csv("data/meta_test.csv")

train_df = pd.concat([X_train.reset_index(drop=True), meta_train.reset_index(drop=True)], axis=1)
train_df = train_df.loc[:, ~train_df.columns.duplicated()]
train_df["is_attack"] = y_train.values

test_df = pd.concat([X_test.reset_index(drop=True), meta_test.reset_index(drop=True)], axis=1)
test_df = test_df.loc[:, ~test_df.columns.duplicated()]
test_df["is_attack"] = y_test.values

print(f"Train : {train_df.shape} | Test : {test_df.shape}")

# ============================================================
# 2. CALCUL DES SEUILS SUR LE TRAFIC NORMAL DU TRAIN
# ============================================================
normal_train = train_df[train_df["is_attack"] == 0]

VOLUME_THRESHOLD = normal_train["Total Length of Fwd Packets"].quantile(0.99)
DURATION_THRESHOLD = normal_train["Flow Duration"].quantile(0.99)

print(f"\nSeuil volume anormal (99e percentile normal) : {VOLUME_THRESHOLD:.0f}")
print(f"Seuil connexion longue (99e percentile normal) : {DURATION_THRESHOLD:.0f}")

PORT_SCAN_THRESHOLD = 15
BRUTE_FORCE_THRESHOLD = 8
TIME_WINDOW = "60s"
SENSITIVE_PORTS = [22, 21, 23, 3389, 3306, 445, 1433]

# ============================================================
# 3. PARSING DU TIMESTAMP -- ENFIN UNE VRAIE DATE COMPLÈTE
# ============================================================
# Contrairement à l'ancien dataset (minute:seconde seulement), ce fichier
# officiel a un timestamp complet (ex. "7/7/2017 1:00") -> pd.to_datetime
# fonctionne directement, plus besoin de parsing manuel bricolé.
test_df["Timestamp_parsed"] = pd.to_datetime(test_df["Timestamp"], errors="coerce")
test_df["time_bin"] = test_df["Timestamp_parsed"].dt.floor(TIME_WINDOW)

n_unparsed = test_df["Timestamp_parsed"].isna().sum()
if n_unparsed > 0:
    print(f"\n[!] {n_unparsed} timestamps n'ont pas pu être parsés (ignorés dans le fenêtrage).")

# ============================================================
# 4. RÈGLE 1 : SCAN DE PORTS
# ============================================================
scan_counts = (
    test_df.groupby(["Source IP", "time_bin"])["Destination Port"]
    .transform("nunique")
)
test_df["rule_port_scan"] = (scan_counts >= PORT_SCAN_THRESHOLD).astype(int)

# ============================================================
# 5. RÈGLE 2 : BRUTE FORCE
# ============================================================
brute_counts = (
    test_df.groupby(["Source IP", "Destination Port", "time_bin"])["Destination Port"]
    .transform("count")
)
is_sensitive_port = test_df["Destination Port"].isin(SENSITIVE_PORTS)
test_df["rule_brute_force"] = (
    (brute_counts >= BRUTE_FORCE_THRESHOLD) & is_sensitive_port
).astype(int)

# ============================================================
# 6. RÈGLE 3 : VOLUME ANORMAL
# ============================================================
test_df["rule_volume_anormal"] = (
    test_df["Total Length of Fwd Packets"] >= VOLUME_THRESHOLD
).astype(int)

# ============================================================
# 7. RÈGLE 4 : CONNEXION LONGUE
# ============================================================
test_df["rule_long_connection"] = (
    test_df["Flow Duration"] >= DURATION_THRESHOLD
).astype(int)

# ============================================================
# 8. COMBINAISON ET ÉVALUATION
# ============================================================
rule_columns = ["rule_port_scan", "rule_brute_force", "rule_volume_anormal", "rule_long_connection"]
test_df["rule_alert"] = test_df[rule_columns].max(axis=1)

print("\n===== NOMBRE DE LIGNES DÉCLENCHÉES PAR RÈGLE =====")
for col in rule_columns:
    print(f"{col} : {test_df[col].sum()}")

y_pred_rules = test_df["rule_alert"]

print("\n===== RAPPORT DE CLASSIFICATION (Moteur de règles) =====")
print(classification_report(test_df["is_attack"], y_pred_rules, target_names=["Normal", "Attaque"]))

cm = confusion_matrix(test_df["is_attack"], y_pred_rules)
print("===== MATRICE DE CONFUSION =====")
print(f"                 Prédit Normal   Prédit Attaque")
print(f"Réel Normal      {cm[0][0]:<15} {cm[0][1]}")
print(f"Réel Attaque     {cm[1][0]:<15} {cm[1][1]}")

# ============================================================
# 9. SAUVEGARDE DES SEUILS
# ============================================================
rule_config = {
    "VOLUME_THRESHOLD": VOLUME_THRESHOLD,
    "DURATION_THRESHOLD": DURATION_THRESHOLD,
    "PORT_SCAN_THRESHOLD": PORT_SCAN_THRESHOLD,
    "BRUTE_FORCE_THRESHOLD": BRUTE_FORCE_THRESHOLD,
    "TIME_WINDOW": TIME_WINDOW,
    "SENSITIVE_PORTS": SENSITIVE_PORTS,
}
joblib.dump(rule_config, "data/rule_engine_config.joblib")
print("\nConfiguration du moteur de règles sauvegardée dans data/rule_engine_config.joblib")
