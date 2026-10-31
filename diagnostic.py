import pandas as pd
import joblib

df = pd.read_csv("data/incident_evidence_records_with_ground_truth.csv")
print(f"Total incident-groups in file: {len(df)}")

real_attacks = df[df["dominant_attack_type"] != "BENIGN"].copy()
print(f"Total REAL (non-BENIGN) incident-groups: {len(real_attacks)}")

any_trigger = (
    real_attacks["port_scan_triggered"] | real_attacks["brute_force_triggered"]
    | real_attacks["long_connection_triggered"] | real_attacks["large_flow_triggered"]
    | real_attacks["if_severity_triggered"] | real_attacks["baseline_severity_triggered"]
    | real_attacks["beaconing_triggered"]
)
triggered = real_attacks[any_trigger].copy()
print(f"Of those, {len(triggered)} trigger at least one rule/model condition (candidates)")
print()
print("Breakdown by true attack type:")
print(real_attacks.groupby("dominant_attack_type")["beaconing_triggered"].agg(
    beaconing_hits="sum", total="count"
))

if len(triggered) == 0:
    print("\nNo triggered real attacks found — stopping before classifier check.")
else:
    clf_model = joblib.load("data/attack_classifier_model.joblib")
    clf_scaler = joblib.load("data/attack_classifier_scaler.joblib")
    clf_features = joblib.load("data/attack_classifier_features.joblib")

    for col in clf_features:
        if triggered[col].dtype == bool:
            triggered[col] = triggered[col].astype(int)

    X = triggered[clf_features]
    X_scaled = clf_scaler.transform(X)
    preds = clf_model.predict(X_scaled)
    probs = clf_model.predict_proba(X_scaled)
    class_names = list(clf_model.classes_)

    triggered["predicted_attack_type"] = preds
    triggered["predicted_confidence"] = [p[class_names.index(pr)] for p, pr in zip(probs, preds)]

    print()
    print(f"Of the {len(triggered)} rule-triggered REAL attacks, classifier predictions:")
    print(triggered["predicted_attack_type"].value_counts())

    print()
    print("Cross-tab: true type vs. whether classifier called it BENIGN")
    triggered["called_benign"] = triggered["predicted_attack_type"] == "BENIGN"
    print(triggered.groupby("dominant_attack_type")["called_benign"].agg(["sum", "count"]))

    correct_attack = triggered[triggered["predicted_attack_type"] == triggered["dominant_attack_type"]]
    print()
    print(f"Exact-match accuracy on rule-triggered real attacks: {len(correct_attack)}/{len(triggered)}")
    if len(correct_attack) > 0:
        print("Confidence distribution for correctly-classified rows:")
        print(correct_attack["predicted_confidence"].describe())
