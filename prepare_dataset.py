import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# 1. CHARGEMENT ET COMBINAISON DES 8 FICHIERS DE LA SEMAINE
# ============================================================
# Chaque fichier correspond à une demi-journée/journée avec un scénario
# d'attaque différent. Le timestamp de ce dataset officiel contient une
# date complète -> plus de risque de collision entre jours (contrairement
# à l'ancien friday.csv qui n'avait que minute:seconde).
FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

dfs = []
for filename in FILES:
    path = f"data/{filename}"
    print(f"Chargement de {filename}...")
    try:
        df_day = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"  -> UTF-8 a échoué, tentative avec l'encodage latin-1...")
        df_day = pd.read_csv(path, encoding="latin1")
    df_day.columns = df_day.columns.str.strip()  # nettoie les espaces dans les noms
    df_day["source_file"] = filename  # utile pour debug/traçabilité
    dfs.append(df_day)

df = pd.concat(dfs, ignore_index=True)
print(f"\nDimensions combinées (8 fichiers) : {df.shape}")

# ============================================================
# 1bis. ÉCHANTILLONNAGE OPTIONNEL (pour itérer plus vite en développement)
# ============================================================
# Si SAMPLE_FRACTION < 1.0, on réduit la taille tout en gardant les mêmes
# proportions de classes (échantillonnage stratifié) -- important pour ne
# pas fausser la répartition BENIGN/attaques ni perdre les classes rares
# (Heartbleed n'a que 11 lignes au total, il ne faut surtout pas les rater).
SAMPLE_FRACTION = 1.0  # ex. 0.2 pour ne garder que 20% des données

if SAMPLE_FRACTION < 1.0:
    df = df.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(frac=SAMPLE_FRACTION, random_state=42)
    )
    print(f"Dimensions après échantillonnage ({SAMPLE_FRACTION*100:.0f}%) : {df.shape}")

# ============================================================
# 2. SÉPARATION DES MÉTADONNÉES (IP, Ports, Timestamp)
# ============================================================
metadata_cols = ["Source IP", "Destination IP", "Source Port", "Destination Port",
                  "Timestamp", "Flow ID", "source_file"]
metadata_cols = [c for c in metadata_cols if c in df.columns]
df_metadata = df[metadata_cols].copy()

cols_to_drop = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "source_file"]
df_features = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

print(f"Dimensions après séparation des métadonnées : {df_features.shape}")

# ============================================================
# 3. NETTOYAGE DES VALEURS INFINIES / ABERRANTES
# ============================================================
numeric_cols = df_features.select_dtypes(include=[np.number]).columns
df_features[numeric_cols] = df_features[numeric_cols].replace([np.inf, -np.inf], np.nan)

n_before = len(df_features)
df_features = df_features.dropna()
df_metadata = df_metadata.loc[df_features.index]
n_after = len(df_features)
print(f"Lignes supprimées (inf/NaN) : {n_before - n_after}")

# ============================================================
# 4. PRÉPARATION DU LABEL
# ============================================================
df_features["Label"] = df_features["Label"].str.strip()
df_features["is_attack"] = (df_features["Label"] != "BENIGN").astype(int)

print("\nRépartition multi-classe (toute la semaine) :")
print(df_features["Label"].value_counts())
print("\nRépartition binaire (0=normal, 1=attaque) :")
print(df_features["is_attack"].value_counts())

# ============================================================
# 5. SÉPARATION FEATURES / LABELS
# ============================================================
X = df_features.drop(columns=["Label", "is_attack"])
y_binary = df_features["is_attack"]
y_multiclass = df_features["Label"]

# ============================================================
# 6. SPLIT TRAIN / TEST (stratifié)
# ============================================================
X_train, X_test, y_train, y_test, y_train_multi, y_test_multi, meta_train, meta_test = train_test_split(
    X, y_binary, y_multiclass, df_metadata,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

print(f"\nTrain : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")

# ============================================================
# 7. SAUVEGARDE
# ============================================================
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
y_train_multi.to_csv("data/y_train_multiclass.csv", index=False)
y_test_multi.to_csv("data/y_test_multiclass.csv", index=False)
meta_train.to_csv("data/meta_train.csv", index=False)
meta_test.to_csv("data/meta_test.csv", index=False)

print("\nFichiers sauvegardés dans data/ (dataset combiné, semaine complète)")
print("Nombre d'IP sources distinctes :", df_metadata["Source IP"].nunique())
