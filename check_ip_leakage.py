import pandas as pd

y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()
meta_train = pd.read_csv("data/meta_train.csv")
meta_test = pd.read_csv("data/meta_test.csv")

train_attack_ips = set(meta_train.loc[y_train.values == 1, "Src IP dec"].unique())
test_attack_ips = set(meta_test.loc[y_test.values == 1, "Src IP dec"].unique())

train_normal_ips = set(meta_train.loc[y_train.values == 0, "Src IP dec"].unique())
test_normal_ips = set(meta_test.loc[y_test.values == 0, "Src IP dec"].unique())

print(f"IP totales dans le dataset : {pd.concat([meta_train, meta_test])['Src IP dec'].nunique()}")

print(f"\nIP attaquantes dans le train : {len(train_attack_ips)}")
print(f"IP attaquantes dans le test  : {len(test_attack_ips)}")
overlap_attack = train_attack_ips & test_attack_ips
print(f"IP attaquantes présentes DANS LES DEUX : {len(overlap_attack)} "
      f"({100 * len(overlap_attack) / max(len(test_attack_ips), 1):.0f}% des IP attaquantes du test étaient déjà vues en train)")

print(f"\nIP normales dans le train : {len(train_normal_ips)}")
print(f"IP normales dans le test  : {len(test_normal_ips)}")
overlap_normal = train_normal_ips & test_normal_ips
print(f"IP normales présentes DANS LES DEUX : {len(overlap_normal)} "
      f"({100 * len(overlap_normal) / max(len(test_normal_ips), 1):.0f}% des IP normales du test étaient déjà vues en train)")

# Est-ce qu'une IP est TOUJOURS attaquante ou TOUJOURS normale (jamais les deux) ?
always_attack_or_normal = train_attack_ips.isdisjoint(train_normal_ips)
print(f"\nUne IP est-elle exclusivement attaquante OU exclusivement normale (jamais les deux) "
      f"dans le train ? {always_attack_or_normal}")
