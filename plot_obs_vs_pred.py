"""
plot_obs_vs_pred.py
-------------------
Dumbbell chart comparing observed vs predicted delta SOC across parameter sets.

Per case (one row):
  ● filled circle  = observed
  □ open square    = default parameters
  ○ open circle    = Phase 4 (group-specific optimisation)
  △ open triangle  = Phase 5 (outlier removal + re-optimisation)

Color = calibration group. Case number shown to the left of each row.
Group labels on the right show Phase 4 bias and RMSE (best overall result).

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
OUTPUT_PNG = BASE_DIR / "outputs" / "obs_vs_pred_all_phases.png"

# Which param sets to show: (label, csv filename, marker, linestyle)
PARAM_SETS = [
    ("Default", None,                         "s", "--"),   # None = use PARAM_CONFIG defaults
    ("Phase 4", "phase4_final_params.csv",    "o", "-"),
    ("Phase 5", "phase5_final_params.csv",    "^", ":"),
]

MS        = 32    # marker size
MS_OBS    = 40    # observed slightly larger
LW        = 0.7   # connector line width

GROUP_ORDER = [
    "annuals_covercrops",
    "annuals_resid",
    "annuals_amend",
    "annuals_to_pasture",
    "perennials_herb",
    "perennials_herb+resid",
    "perennials_amend",
]

GROUP_COLORS = {
    "annuals_covercrops":    "#2196F3",
    "annuals_resid":         "#00BCD4",
    "annuals_amend":         "#4CAF50",
    "annuals_to_pasture":    "#8BC34A",
    "perennials_herb":       "#FF9800",
    "perennials_herb+resid": "#F44336",
    "perennials_amend":      "#9C27B0",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params(csv_name):
    """Return (param_names, param_values) from outputs/ CSV, or model defaults."""
    if csv_name is None:
        names  = list(PARAM_CONFIG.keys())
        values = [PARAM_CONFIG[p]["default"] for p in names]
    else:
        df     = pd.read_csv(BASE_DIR / "outputs" / csv_name)
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

# ── Run each param set ────────────────────────────────────────────────────────

predictions = {}
for label, csv_name, _marker, _ls in PARAM_SETS:
    print(f"Running model: {label}...")
    names, values = load_params(csv_name)
    pred_df       = get_predictions(names, values, data)
    pred_df       = pred_df.merge(cases_info, on="case")
    predictions[label] = pred_df

# ── Build master frame sorted by group / observed ────────────────────────────

p4_label = PARAM_SETS[1][0]   # "Phase 4" — reference for group annotations
base_df  = predictions[p4_label][["case", "observed", "group_calib"]].copy()
base_df["group_order"] = base_df["group_calib"].map({g: i for i, g in enumerate(GROUP_ORDER)})
base_df = base_df.sort_values(["group_order", "observed"]).reset_index(drop=True)
base_df["y"] = range(len(base_df))

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

    # Observed: filled circle (topmost)
    ax.scatter(obs, y, s=MS_OBS, marker="o",
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
        f"  (n={int(gs['n'])}, "
        f"P4 bias={gs['bias_p4']:+.2f}, "
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
    "Observed vs Predicted ΔSoC — Default / Phase 4 / Phase 5\n"
    "● observed  □ default  ○ Phase 4  △ Phase 5",
    fontsize=10,
)
ax.set_ylim(-0.5, base_df["y"].max() + 0.5)
ax.invert_yaxis()

# Legend
legend_handles = [
    plt.scatter([], [], s=MS_OBS, marker="o", facecolors="gray",
                edgecolors="gray", label="Observed"),
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
