import pandas as pd
df = pd.read_csv("data/incident_evidence_records_with_ground_truth.csv")

candidates = df[
    (df["port_scan_triggered"]) | (df["brute_force_triggered"]) |
    (df["long_connection_triggered"]) | (df["large_flow_triggered"]) |
    (df["if_severity_triggered"]) | (df["baseline_severity_triggered"])
]
print("Candidats (nouveau filtre) :", len(candidates), "/", len(df))
print("Vrais incidents parmi eux :", candidates["true_incident"].sum())
print("Faux positifs parmi eux   :", (candidates["true_incident"] == 0).sum())

for col in ["n_if_anomalies", "if_severity_triggered", "baseline_severity_triggered", "risk_score"]:
    print(f"corr({col}, true_incident) =", df[col].astype(float).corr(df["true_incident"]))

print("\nPires pollueurs (faux positifs répétés par paire IP) :")
fp = candidates[candidates["true_incident"] == 0]
print(fp.groupby(["Source IP", "Destination IP"]).size().sort_values(ascending=False).head(10))