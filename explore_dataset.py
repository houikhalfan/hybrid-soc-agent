import pandas as pd

# Charger le dataset
df = pd.read_csv("data/friday.csv")

print("===== DIMENSIONS =====")
print(df.shape)  # (nombre de lignes, nombre de colonnes)

print("\n===== COLONNES =====")
print(df.columns.tolist())

print("\n===== APERÇU DES 5 PREMIÈRES LIGNES =====")
print(df.head())

print("\n===== TYPES DE DONNÉES / VALEURS MANQUANTES =====")
print(df.info())

print("\n===== VALEURS MANQUANTES PAR COLONNE =====")
print(df.isnull().sum()[df.isnull().sum() > 0])

# La colonne de label s'appelle généralement "Label" dans CICIDS2017
# (parfois avec un espace au début : " Label")
label_col = None
for col in df.columns:
    if col.strip().lower() == "label":
        label_col = col
        break

if label_col:
    print(f"\n===== RÉPARTITION DES CLASSES (colonne '{label_col}') =====")
    print(df[label_col].value_counts())
else:
    print("\n[!] Colonne 'Label' non trouvée — vérifie le nom exact dans la liste des colonnes ci-dessus.")
