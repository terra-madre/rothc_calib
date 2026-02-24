"""
plot_covercrop_test.py
----------------------
Dumbbell chart comparing three param sets for the cover crop test:
  ● observed
  □ Default params
  ○ Phase 2 (all.json)
  △ Covercrop Test (covercrop_test.json)

Covercrop groups are shown with full opacity; other groups are faded.
Cases sorted by group then by observed value within group.
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from optimization import precompute_data, objective, PARAM_CONFIG

BASE_DIR   = Path(__file__).parent.parent
OUTPUT_PNG = BASE_DIR / "outputs" / "obs_vs_pred_covercrop_test.png"

COVERCROP_GROUPS = ["covercrop", "covercrop_amendment", "covercrop_cropresid", "covercrop_pruning"]

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

# ── Param sets to compare ─────────────────────────────────────────────────────

PARAM_SETS = [
    ("Default",        None,    "s", "Default"),
    ("Phase 2",        "phase2_sequential_checkpoints/all.json", "o", "Phase 2 (all)"),
    ("Covercrop Test", "covercrop_test_checkpoints/covercrop_test.json", "^", "Covercrop Test"),
]

MS_OBS = 50
MS     = 35
LW     = 0.8

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params(source):
    if source is None:
        names  = list(PARAM_CONFIG.keys())
        values = [PARAM_CONFIG[p]["default"] for p in names]
        return names, values
    ckpt = json.loads((BASE_DIR / "outputs" / source).read_text())
    names  = list(ckpt["params"].keys())
    values = list(ckpt["params"].values())
    return names, values


def get_predictions(param_names, param_values, data):
    _, details = objective(param_values, param_names, data, return_details=True)
    df = details["comparison_df"].copy().rename(columns={
        "delta_treatment_control_per_year": "predicted",
        "delta_soc_t_ha_y": "observed",
    })
    return df[["case", "observed", "predicted"]].copy()


# ── Load & run ────────────────────────────────────────────────────────────────

print("Loading and precomputing model data...")
data       = precompute_data(repo_root=BASE_DIR)
cases_info = data["cases_info_df"][["case", "group_calib"]]

predictions = {}
for label, source, _m, _desc in PARAM_SETS:
    print(f"Running model: {label}...")
    names, values = load_params(source)
    pred_df       = get_predictions(names, values, data)
    pred_df       = pred_df.merge(cases_info, on="case")
    predictions[label] = pred_df

# ── Build sorted master frame ─────────────────────────────────────────────────

ref_label = PARAM_SETS[-1][0]
base_df   = predictions[ref_label][["case", "observed", "group_calib"]].copy()
base_df["group_order"] = base_df["group_calib"].map({g: i for i, g in enumerate(GROUP_ORDER)})
base_df = base_df.sort_values(["group_order", "observed"]).reset_index(drop=True)
base_df["y"] = range(len(base_df))
base_df["is_cc"] = base_df["group_calib"].isin(COVERCROP_GROUPS)

for label, _, _, _ in PARAM_SETS:
    col = f"pred_{label}"
    base_df = base_df.merge(
        predictions[label][["case", "predicted"]].rename(columns={"predicted": col}),
        on="case",
    )

# ── Per-group stats for annotations ──────────────────────────────────────────

def group_stats(df, pred_col):
    stats = {}
    for grp, g in df.groupby("group_calib"):
        res   = g["observed"] - g[pred_col]
        bias  = res.mean()
        rmse  = np.sqrt((res**2).mean())
        ss_res = (res**2).sum()
        ss_tot = ((g["observed"] - g["observed"].mean())**2).sum()
        r2    = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        stats[grp] = {"bias": bias, "rmse": rmse, "r2": r2}
    return stats

stats_default = group_stats(base_df, "pred_Default")
stats_p2      = group_stats(base_df, "pred_Phase 2")
stats_cc      = group_stats(base_df, "pred_Covercrop Test")

# ── Figure ────────────────────────────────────────────────────────────────────

n_cases = len(base_df)
fig_height = max(10, n_cases * 0.30)
fig, ax = plt.subplots(figsize=(13, fig_height))

# Draw each case row
for _, row in base_df.iterrows():
    y        = row["y"]
    color    = GROUP_COLORS.get(row["group_calib"], "#888888")
    alpha    = 1.0 if row["is_cc"] else 0.30
    lw       = LW if row["is_cc"] else LW * 0.7

    obs  = row["observed"]
    pd0  = row["pred_Default"]
    pp2  = row["pred_Phase 2"]
    pcc  = row["pred_Covercrop Test"]

    # Connector spanning all preds + obs
    xvals = [obs, pd0, pp2, pcc]
    ax.plot([min(xvals), max(xvals)], [y, y],
            color=color, lw=lw, alpha=alpha, zorder=1)

    # Points
    ax.scatter(obs, y, s=MS_OBS, marker="o", facecolor=color, edgecolor="k",
               linewidth=0.6, zorder=5, alpha=alpha)
    ax.scatter(pd0, y, s=MS, marker="s", facecolor="none", edgecolor=color,
               linewidth=0.8, zorder=3, alpha=alpha)
    ax.scatter(pp2, y, s=MS, marker="o", facecolor="none", edgecolor=color,
               linewidth=0.8, zorder=3, alpha=alpha)
    ax.scatter(pcc, y, s=MS, marker="^", facecolor=color, edgecolor="k",
               linewidth=0.6, zorder=4, alpha=alpha)

# ── Group banding & labels ────────────────────────────────────────────────────

groups_in_order = base_df.drop_duplicates("group_calib").set_index("group_calib")["group_order"].sort_values().index.tolist()
text_x = ax.get_xlim()[1] if ax.get_xlim()[1] != 1.0 else 2.5  # will adjust after autoscale

# Horizontal dividers + right-side annotations
for i, grp in enumerate(groups_in_order):
    g = base_df[base_df["group_calib"] == grp]
    y_min, y_max = g["y"].min(), g["y"].max()
    y_mid = (y_min + y_max) / 2
    is_cc = grp in COVERCROP_GROUPS

    if i > 0:
        ax.axhline(y_min - 0.5, color="#cccccc", lw=0.8, ls="--", zorder=0)

    # Group shading for covercrop groups
    if is_cc:
        ax.axhspan(y_min - 0.5, y_max + 0.5, color="#E3F2FD", alpha=0.4, zorder=0)

    s_p2  = stats_p2[grp]
    s_cc  = stats_cc[grp]
    color = GROUP_COLORS.get(grp, "#888888")
    fa    = 1.0 if is_cc else 0.45

    label_text = (
        f"{grp}\n"
        f"P2:  RMSE={s_p2['rmse']:.2f}  bias={s_p2['bias']:+.2f}\n"
        f"CC:  RMSE={s_cc['rmse']:.2f}  bias={s_cc['bias']:+.2f}"
    )
    ax.text(1.01, y_mid, label_text,
            transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=6.5, color=color,
            alpha=fa, family="monospace")

# Case number labels on the left
for _, row in base_df.iterrows():
    alpha = 1.0 if row["is_cc"] else 0.25
    ax.text(-0.01, row["y"], str(int(row["case"])),
            transform=ax.get_yaxis_transform(),
            va="center", ha="right", fontsize=6, color="#555555", alpha=alpha)

# ── Axes & decorations ────────────────────────────────────────────────────────

ax.axvline(0, color="#999999", lw=0.8, ls=":", zorder=0)
ax.set_xlabel("Annual ΔSOC (t C ha⁻¹ y⁻¹)", fontsize=11)
ax.set_yticks([])
ax.set_title("Obs vs Predicted ΔSOC — Covercrop Test\n"
             "(covercrop groups highlighted; faded = other groups)", fontsize=12)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markeredgecolor="#555",
           markersize=8, label="Observed"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="none", markeredgecolor="#555",
           markersize=8, markeredgewidth=1, label="Default"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#555",
           markersize=8, markeredgewidth=1, label="Phase 2 (all)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#555", markeredgecolor="#555",
           markersize=8, label="Covercrop Test"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.85)

plt.tight_layout(rect=[0.03, 0, 0.78, 1])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_PNG}")

# ── Summary stats ─────────────────────────────────────────────────────────────

cc_df    = base_df[base_df["is_cc"]]
other_df = base_df[~base_df["is_cc"]]

print("\n=== COVERCROP CASES ===")
for col, lbl in [("pred_Default", "Default  "), ("pred_Phase 2", "Phase 2  "), ("pred_Covercrop Test", "CC Test  ")]:
    res  = cc_df["observed"] - cc_df[col]
    rmse = np.sqrt((res**2).mean())
    bias = res.mean()
    ss_res = (res**2).sum()
    ss_tot = ((cc_df["observed"] - cc_df["observed"].mean())**2).sum()
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  {lbl}: RMSE={rmse:.4f}  MAE={np.abs(res).mean():.4f}  Bias={bias:+.4f}  R²={r2:.4f}")

print("\n=== ALL CASES ===")
for col, lbl in [("pred_Default", "Default  "), ("pred_Phase 2", "Phase 2  "), ("pred_Covercrop Test", "CC Test  ")]:
    res  = base_df["observed"] - base_df[col]
    rmse = np.sqrt((res**2).mean())
    bias = res.mean()
    ss_res = (res**2).sum()
    ss_tot = ((base_df["observed"] - base_df["observed"].mean())**2).sum()
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  {lbl}: RMSE={rmse:.4f}  MAE={np.abs(res).mean():.4f}  Bias={bias:+.4f}  R²={r2:.4f}")
