"""
plot_obs_vs_pred.py
-------------------
Dumbbell chart comparing observed vs predicted delta SOC across parameter sets.

Per case (one row):
  ● filled circle   = observed (calibration / training case)
  ◆ filled diamond  = observed (validation / test case)
  □ open square     = default parameters
  ○ open circle     = Phase 2 sequential optimisation
  △ open triangle   = Cal-Val calibration set optimisation

Color = calibration group. Case number shown to the left of each row.
Group labels on the right show Cal-Val bias and RMSE.

Cases sorted by group (GROUP_ORDER) then by observed value within group.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from optimization import precompute_data, objective, PARAM_CONFIG

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
OUTPUT_PNG = BASE_DIR / "outputs" / "obs_vs_pred_calval.png"

# Which param sets to show: (label, source, marker, linestyle)
# source can be:
#   None                               → PARAM_CONFIG defaults
#   "<name>.csv"                       → outputs/<name>.csv  (columns: param, value)
#   {"seq_checkpoint": "<name>.json"}  → outputs/phase2_sequential_checkpoints/<name>.json
#   {"calval_checkpoint": "<name>.json"} → outputs/calval_checkpoints/<name>.json
PARAM_SETS = [
    ("Default",  None,                                          "s", "--"),
    ("Phase 2",  {"seq_checkpoint":    "all.json"},             "o", "-"),
    ("Cal-Val",  {"calval_checkpoint": "all.json"},             "^", "-."),
]

MS        = 32    # marker size
MS_OBS    = 40    # observed slightly larger
LW        = 0.7   # connector line width

GROUP_ORDER = [
    "amendment",
    "cropresid",
    "covercrop",
    "covercrop_amendment",
    "covercrop_cropresid",
    "covercrop_pruning",
    "grass",
    "grass_annuals",
    "grass_pruning",
]

GROUP_COLORS = {
    "amendment":            "#4CAF50",
    "cropresid":            "#8BC34A",
    "covercrop":            "#2196F3",
    "covercrop_amendment":  "#00BCD4",
    "covercrop_cropresid":  "#03A9F4",
    "covercrop_pruning":    "#673AB7",
    "grass":                "#FF9800",
    "grass_annuals":        "#F44336",
    "grass_pruning":        "#E91E63",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params(source):
    """Return (param_names, param_values) from a source spec.

    source = None                      → PARAM_CONFIG defaults
    source = "file.csv"                → outputs/file.csv  (columns: param, value)
    source = {"checkpoint": …}         → outputs/phase2_de_checkpoints/<name>.json
    source = {"seq_checkpoint": …}     → outputs/phase2_sequential_checkpoints/<name>.json
    source = {"calval_checkpoint": …}  → outputs/calval_checkpoints/<name>.json
    """
    import json
    if source is None:
        names  = list(PARAM_CONFIG.keys())
        values = [PARAM_CONFIG[p]["default"] for p in names]
    elif isinstance(source, dict) and "checkpoint" in source:
        ckpt = json.loads(
            (BASE_DIR / "outputs" / "phase2_de_checkpoints" / source["checkpoint"]).read_text()
        )
        names  = list(ckpt["params"].keys())
        values = list(ckpt["params"].values())
    elif isinstance(source, dict) and "seq_checkpoint" in source:
        ckpt = json.loads(
            (BASE_DIR / "outputs" / "phase2_sequential_checkpoints" / source["seq_checkpoint"]).read_text()
        )
        names  = list(ckpt["params"].keys())
        values = list(ckpt["params"].values())
    elif isinstance(source, dict) and "calval_checkpoint" in source:
        ckpt = json.loads(
            (BASE_DIR / "outputs" / "calval_checkpoints" / source["calval_checkpoint"]).read_text()
        )
        names  = list(ckpt["params"].keys())
        values = list(ckpt["params"].values())
    else:
        df     = pd.read_csv(BASE_DIR / "outputs" / source)
        names  = df["param"].tolist()
        values = df["value"].tolist()
    return names, values


def get_predictions(param_names, param_values, data):
    _, details = objective(param_values, param_names, data, return_details=True)
    df = details["comparison_df"].copy().rename(columns={
        "delta_treatment_control_per_year": "predicted",
        "delta_soc_t_ha_y": "observed",
    })
    return df[["case", "observed", "predicted"]].copy()


# ── Load data (once) ──────────────────────────────────────────────────────────

print("Loading and precomputing model data...")
data       = precompute_data(repo_root=BASE_DIR)
cases_info = data["cases_info_df"][["case", "group_calib"]]

# ── Load cal-val split (train / test flags) ───────────────────────────────────

calval_split = pd.read_csv(BASE_DIR / "outputs" / "calval_split.csv")
test_cases   = set(calval_split.loc[calval_split["split"] == "test", "case"])

# ── Run each param set ────────────────────────────────────────────────────────

predictions = {}
for label, source, _marker, _ls in PARAM_SETS:
    print(f"Running model: {label}...")
    names, values = load_params(source)
    pred_df       = get_predictions(names, values, data)
    pred_df       = pred_df.merge(cases_info, on="case")
    predictions[label] = pred_df

# ── Build master frame sorted by group / observed ────────────────────────────

p4_label = PARAM_SETS[-1][0]   # last entry — reference for group annotations
base_df  = predictions[p4_label][["case", "observed", "group_calib"]].copy()
base_df["group_order"] = base_df["group_calib"].map({g: i for i, g in enumerate(GROUP_ORDER)})
base_df = base_df.sort_values(["group_order", "observed"]).reset_index(drop=True)
base_df["y"]        = range(len(base_df))
base_df["is_test"]  = base_df["case"].isin(test_cases)

for label, _, _, _ in PARAM_SETS:
    col = f"pred_{label}"
    base_df = base_df.merge(
        predictions[label][["case", "predicted"]].rename(columns={"predicted": col}),
        on="case",
    )

# ── Group summary stats ───────────────────────────────────────────────────────

group_stats = (
    base_df.groupby("group_calib")
    .apply(lambda g: pd.Series({
        "n":       len(g),
        "n_test":  g["is_test"].sum(),
        "bias_p4": (g["observed"] - g[f"pred_{p4_label}"]).mean(),
        "rmse_p4": np.sqrt(((g["observed"] - g[f"pred_{p4_label}"])**2).mean()),
        "y_min":   g["y"].min(),
        "y_max":   g["y"].max(),
    }))
    .reset_index()
)

# ── Plot ──────────────────────────────────────────────────────────────────────

FIG_HEIGHT = max(14, len(base_df) * 0.24)
fig, ax    = plt.subplots(figsize=(11, FIG_HEIGHT))

# Alternating group shading
for i, gs in group_stats.iterrows():
    if i % 2 == 1:
        ax.axhspan(gs["y_min"] - 0.5, gs["y_max"] + 0.5, color="#f4f4f4", zorder=0)

# Per-case dumbbells
for _, row in base_df.iterrows():
    color = GROUP_COLORS.get(row["group_calib"], "gray")
    y     = row["y"]
    obs   = row["observed"]

    # Connector lines (observed → each prediction)
    for label, _, marker, ls in PARAM_SETS:
        pred = row[f"pred_{label}"]
        ax.plot([obs, pred], [y, y], color=color, lw=LW, ls=ls, alpha=0.45, zorder=1)

    # Predicted markers (open), drawn before observed so obs sits on top
    for label, _, marker, ls in PARAM_SETS:
        pred = row[f"pred_{label}"]
        ax.scatter(pred, y, s=MS, marker=marker,
                   facecolors="white", edgecolors=color,
                   zorder=3, linewidths=0.9)

    # Observed: filled circle (train) or filled diamond (test) — topmost
    obs_marker = "D" if row["is_test"] else "o"
    obs_size   = MS_OBS * 0.85 if row["is_test"] else MS_OBS
    ax.scatter(obs, y, s=obs_size, marker=obs_marker,
               facecolors=color, edgecolors=color, zorder=4, linewidths=0.5)

    # Case number to the left of the leftmost point
    x_left = min(obs, *[row[f"pred_{lbl}"] for lbl, _, __, ___ in PARAM_SETS])
    ax.text(x_left - 0.07, y, str(int(row["case"])),
            fontsize=5.5, va="center", ha="right", color=color, alpha=0.85)

# Group labels on the right
ax.autoscale()
xlim = ax.get_xlim()

for _, gs in group_stats.iterrows():
    mid_y = (gs["y_min"] + gs["y_max"]) / 2
    txt   = (
        f"  {gs['group_calib']}"
        f"  (n={int(gs['n'])}, val={int(gs['n_test'])}, "
        f"bias={gs['bias_p4']:+.2f}, "
        f"RMSE={gs['rmse_p4']:.2f})"
    )
    ax.text(xlim[1], mid_y, txt,
            fontsize=6.5, va="center", ha="left",
            color=GROUP_COLORS.get(gs["group_calib"], "black"),
            clip_on=False)

# Axes
ax.set_yticks(base_df["y"])
ax.set_yticklabels(base_df["case"].astype(str), fontsize=5.5)
ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.35)
ax.set_xlabel("Δ SOC  (t C ha⁻¹ yr⁻¹)", fontsize=10)
ax.set_title(
    "Observed vs Predicted ΔSoC — Default / Phase 2 / Cal-Val\n"
    "● obs (train)  ◆ obs (validation)  □ default  ○ Phase 2  △ Cal-Val",
    fontsize=10,
)
ax.set_ylim(-0.5, base_df["y"].max() + 0.5)
ax.invert_yaxis()

# Legend
legend_handles = [
    plt.scatter([], [], s=MS_OBS, marker="o", facecolors="gray",
                edgecolors="gray", label="Observed (train)"),
    plt.scatter([], [], s=MS_OBS * 0.85, marker="D", facecolors="gray",
                edgecolors="gray", label="Observed (validation)"),
]
for label, _, marker, ls in PARAM_SETS:
    legend_handles.append(
        plt.scatter([], [], s=MS, marker=marker, facecolors="white",
                    edgecolors="gray", label=f"Predicted ({label})")
    )
ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.85)

plt.tight_layout(rect=[0, 0, 0.70, 1])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {OUTPUT_PNG}")
