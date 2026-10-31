import pandas as pd, numpy as np, joblib, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
os.makedirs("images", exist_ok=True)

# ---------- 0. Timestamp problem check ----------
m = pd.read_csv("data/meta_test.csv")
y = pd.read_csv("data/y_test_multiclass.csv").squeeze()
bad = pd.to_datetime(m["Timestamp"], errors="coerce").isna()
print("Unparsed timestamps:", bad.sum())
print(y[bad].value_counts())
print(m.loc[bad, "Timestamp"].head(10).tolist())

# ---------- 1. Detection coverage per attack type ----------
e = pd.read_csv("data/incident_evidence_records_with_ground_truth.csv")
flags = ["port_scan_triggered","brute_force_triggered","long_connection_triggered",
         "large_flow_triggered","if_severity_triggered","baseline_severity_triggered","beaconing_triggered"]
e["cand"] = e[flags].any(axis=1)
att = e[e.dominant_attack_type != "BENIGN"]
cov = att.groupby("dominant_attack_type").agg(total=("cand","size"), candidates=("cand","sum"))
cov["coverage_%"] = (100*cov.candidates/cov.total).round(1)
print(cov); cov.to_csv("images/coverage.csv")
print(att[att.dominant_attack_type=="Bot"][flags].sum())

# ---------- 2. Confusion matrices (values from your runs) ----------
def cm_plot(cm, title, fn):
    fig, ax = plt.subplots(figsize=(4.5,4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > cm.max()/2 else "black")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Prédit normal","Prédit attaque"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Réel normal","Réel attaque"])
    ax.set_title(title); plt.tight_layout(); plt.savefig(fn, dpi=200); plt.close()

cm_plot(np.array([[418763,35502],[61416,49895]]), "Isolation Forest", "images/cm_if.png")
cm_plot(np.array([[423805,30460],[76903,34408]]), "Moteur de règles", "images/cm_rules.png")
cm_plot(np.array([[434690,19575],[50720,60591]]), "Baseline statistique", "images/cm_baseline.png")
cm_plot(np.array([[345008,109257],[15072,96239]]), "Score de risque composite", "images/cm_risk.png")

# ---------- 3. Comparison bar chart ----------
methods = ["Règles","Isolation Forest","Baseline","Score composite"]
P = [0.53,0.58,0.76,0.47]; R = [0.31,0.45,0.54,0.86]; F = [0.39,0.51,0.63,0.61]
x = np.arange(4); w = 0.27
plt.figure(figsize=(7,4))
plt.bar(x-w,P,w,label="Précision"); plt.bar(x,R,w,label="Rappel"); plt.bar(x+w,F,w,label="F1")
plt.xticks(x, methods); plt.ylim(0,1); plt.legend(); plt.tight_layout()
plt.savefig("images/comparison.png", dpi=200); plt.close()

# ---------- 4. PR curve of the composite score ----------
r = pd.read_csv("data/risk_score_results.csv")
p, rc, _ = precision_recall_curve(r.y_true, r.risk_score)
plt.figure(figsize=(5,4)); plt.plot(rc, p)
plt.xlabel("Rappel"); plt.ylabel("Précision"); plt.title("Courbe Précision-Rappel (score composite)")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("images/pr_curve.png", dpi=200); plt.close()

# ---------- PR curve: Isolation Forest specifically ----------
if_pr = pd.read_csv("data/if_pr_curve.csv")
plt.figure(figsize=(5,4))
plt.plot(if_pr["recall"], if_pr["precision"])
plt.axhline(0.197, ls="--", c="gray", label="Classifieur aléatoire")
plt.scatter([0.448], [0.584], c="red", zorder=5, label="Seuil retenu (-0,044)")
plt.xlabel("Rappel"); plt.ylabel("Précision"); plt.legend(); plt.grid(alpha=.3)
plt.title("Courbe Précision-Rappel — Isolation Forest")
plt.tight_layout(); plt.savefig("images/pr_curve_if.png", dpi=200); plt.close()
# ---------- 5. Risk classes distribution ----------
r["risk_class"].value_counts().reindex(["faible","moyen","élevé","critique"]).plot(kind="bar", logy=True)
plt.ylabel("Nombre de flux (échelle log)"); plt.tight_layout()
plt.savefig("images/risk_classes.png", dpi=200); plt.close()

# ---------- 6. Random Forest: normalized confusion matrix + importances ----------
from sklearn.metrics import ConfusionMatrixDisplay
c = e[e[flags].any(axis=1)].copy()
feats = joblib.load("data/attack_classifier_features.joblib")
model = joblib.load("data/attack_classifier_model.joblib")
scaler = joblib.load("data/attack_classifier_scaler.joblib")
for col in feats:
    if c[col].dtype == bool: c[col] = c[col].astype(int)
_, Xt, _, yt = train_test_split(c[feats], c["dominant_attack_type"],
                                test_size=0.25, random_state=42, stratify=c["dominant_attack_type"])
pred = model.predict(scaler.transform(Xt))
fig, ax = plt.subplots(figsize=(9,8))
ConfusionMatrixDisplay.from_predictions(yt, pred, normalize="true", xticks_rotation=90,
                                        ax=ax, colorbar=False, values_format=".2f")
plt.tight_layout(); plt.savefig("images/cm_rf.png", dpi=200); plt.close()

imp = pd.Series(model.feature_importances_, index=feats).sort_values()
imp.plot(kind="barh", figsize=(6,5)); plt.tight_layout()
plt.savefig("images/rf_importance.png", dpi=200); plt.close()
print("Done, see images/")
# --- RF matrix, cleaned ---
clean = lambda s: pd.Series(s).astype(str).str.replace("\x96", "-", regex=False).str.replace("\ufffd", "-", regex=False).str.replace("Web Attack  ", "Web Attack - ", regex=False)
yt_c, pred_c = clean(yt.values), clean(pred)
labels = sorted(set(yt_c))                      # only classes present in the test set
counts = yt_c.value_counts()
names = [f"{l} (n={counts[l]})" for l in labels]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yt_c, pred_c, labels=labels)
cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(9,8))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(len(labels)):
    for j in range(len(labels)):
        if cm[i][j]:
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cmn[i][j] > .5 else "black", fontsize=8)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(names)
ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe réelle (n = effectif test)")
plt.tight_layout(); plt.savefig("images/cm_rf.png", dpi=200); plt.close()

# --- PR curve, cleaned ---
p, rc, _ = precision_recall_curve(r.y_true, r.risk_score)
plt.figure(figsize=(5,4)); plt.plot(rc[1:], p[1:])
plt.axhline(r.y_true.mean(), ls="--", c="gray", label="Classifieur aléatoire (0,20)")
plt.scatter([0.86],[0.47], c="red", zorder=5, label="Seuil retenu (12/100)")
plt.xlabel("Rappel"); plt.ylabel("Précision"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("images/pr_curve.png", dpi=200); plt.close()

# --- Risk classes, cleaned ---
ax = r["risk_class"].value_counts().reindex(["faible","moyen","élevé","critique"]).plot(kind="bar", logy=True)
ax.set_xlabel("Classe de risque"); ax.set_ylabel("Nombre de flux (échelle log)")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig("images/risk_classes.png", dpi=200); plt.close()