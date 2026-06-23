import os, sys, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result"
OUT_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part4_visualization\output_all"
os.makedirs(OUT_DIR, exist_ok=True)

EXPS = {
    "exp_001": "Berlin1",
    "exp_002": "Berlin2",
    "exp_003": "Frankfurt1",
    "exp_004": "Frankfurt2",
    "exp_hk": "HongKong",
}

all_metrics = {}
for exp_name, label in EXPS.items():
    pred_file = os.path.join(RESULT_DIR, exp_name, "predictions.json")
    if not os.path.exists(pred_file):
        print(f"SKIP {exp_name}: no predictions.json")
        continue
    with open(pred_file) as f:
        data = json.load(f)
    m = data["metrics"]
    m["nlos_pct"] = data["dataset_stats"]["nlos_pct"]
    all_metrics[label] = m
    print(f"{label}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} NLOS={m['nlos_pct']:.1f}%")

labels = list(all_metrics.keys())
colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

# Plot 1: Acc + F1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
acc_vals = [all_metrics[l]["accuracy"] for l in labels]
f1_vals = [all_metrics[l]["f1"] for l in labels]
ax1.bar(labels, acc_vals, color=colors)
ax1.set_title("Accuracy"); ax1.set_ylim(0, 1.05); ax1.tick_params(axis="x", rotation=30)
for i,v in enumerate(acc_vals): ax1.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=8)
ax2.bar(labels, f1_vals, color=colors)
ax2.set_title("F1 Score"); ax2.set_ylim(0, 1.05); ax2.tick_params(axis="x", rotation=30)
for i,v in enumerate(f1_vals): ax2.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "01_classification_metrics.png"), dpi=150); plt.close()
print("[1/5] Classification metrics")

# Plot 2: p_los gap
fig, ax = plt.subplots(figsize=(10, 5))
p_gaps = [all_metrics[l].get("plos_gap", 0) for l in labels]
ax.bar(labels, p_gaps, color=colors)
ax.set_title("p_los Gap (LOS mean - NLOS mean)"); ax.axhline(y=0, color="gray", linestyle="--"); ax.tick_params(axis="x", rotation=30)
for i,v in enumerate(p_gaps): ax.text(i, v+0.01 if v>=0 else v-0.05, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "02_plos_gap.png"), dpi=150); plt.close()
print("[2/5] p_los gap")

# Plot 3: p_los distribution overlay
fig, ax = plt.subplots(figsize=(12, 6))
for exp_name, label in EXPS.items():
    pred_file = os.path.join(RESULT_DIR, exp_name, "predictions.json")
    with open(pred_file) as f:
        data = json.load(f)
    p_los = np.array(data["p_los"])
    lbs = np.array(data["labels"])
    ax.hist(p_los[lbs==0], bins=50, alpha=0.3, density=True, label=f"{label} LOS")
    if (lbs==1).any():
        ax.hist(p_los[lbs==1], bins=50, alpha=0.5, density=True, label=f"{label} NLOS")
ax.set_xlabel("p_los"); ax.set_ylabel("Density"); ax.set_title("p_los Distribution: LOS vs NLOS (5 datasets)")
ax.legend(fontsize=6, ncol=2)
plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "03_plos_distribution.png"), dpi=150); plt.close()
print("[3/5] p_los distribution")

# Plot 4: NLOS rate vs F1
fig, ax = plt.subplots(figsize=(10, 5))
nlos_rates = [all_metrics[l]["nlos_pct"] for l in labels]
ax.scatter(nlos_rates, f1_vals, s=200, c=colors, zorder=5)
for i, l in enumerate(labels):
    ax.annotate(l, (nlos_rates[i], f1_vals[i]), fontsize=9, ha="center", va="bottom")
ax.set_xlabel("NLOS Rate (%)"); ax.set_ylabel("F1 Score")
ax.set_title("NLOS Rate vs Classification Performance"); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "04_nlos_vs_f1.png"), dpi=150); plt.close()
print("[4/5] NLOS vs F1")

# Plot 5: Summary table
fig, ax = plt.subplots(figsize=(12, 4)); ax.axis("off")
table_data = [["Dataset", "Acc", "F1", "Prec", "Recall", "p_los Gap", "NLOS%"]]
for l in labels:
    m = all_metrics[l]
    table_data.append([l, f"{m['accuracy']:.4f}", f"{m['f1']:.4f}",
                       f"{m.get('precision',0):.4f}", f"{m.get('recall',0):.4f}",
                       f"{m.get('plos_gap',0):.4f}", f"{m['nlos_pct']:.1f}"])
table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.2, 1.5)
for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        table[i,j].set_facecolor("#f0f0f0" if i==0 else "white")
ax.set_title("5-Dataset NLOS Classification Summary")
plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "05_summary_table.png"), dpi=150, bbox_inches="tight"); plt.close()
print("[5/5] Summary table")

print(f"\nAll 5 plots saved to {OUT_DIR}")
