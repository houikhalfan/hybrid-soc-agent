import pandas as pd
df = pd.read_csv("data/incident_evidence_records_with_ground_truth.csv")

old_filter = (
    df["port_scan_triggered"] | df["brute_force_triggered"] |
    df["long_connection_triggered"] | df["large_flow_triggered"] |
    (df["n_if_anomalies"] >= 10)
)
rescued = df[df["if_severity_triggered"] & ~old_filter]
print("Incidents QUE if_severity_triggered rattrape :", len(rescued))
print("Dont vrais incidents parmi eux :", rescued["true_incident"].sum())
print(rescued[["Source IP", "Destination IP", "n_flows", "if_anomaly_max", "true_incident"]].head(10))

rescued_b = df[df["baseline_severity_triggered"] & ~old_filter]
print("\nPareil pour baseline_severity_triggered :", len(rescued_b), "dont vrais:", rescued_b["true_incident"].sum())