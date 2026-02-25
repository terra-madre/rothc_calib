"""
plot_scatter_obs_vs_pred.py
---------------------------
1×3 scatter plot: Observed vs Predicted Δ SOC for three parameter sets.

Panels (left to right):
  1. Default parameters
  2. Phase 2 sequential calibration (all 70 cases)
  3. Cal-Val calibration (train ● / test ◆ distinguished)

Each panel:
  - Observed (x-axis) vs Predicted (y-axis)
  - 1:1 line + ±RMSE_cal envelope (grey shading)
  - Coloured by calibration group
  - RMSE / R² / Bias annotation box

Shared group colour legend at the bottom.

Output: outputs/scatter_obs_vs_pred.png
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from optimization import precompute_data, objective, PARAM_CONFIG

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
OUTPUT_PNG = BASE_DIR / "outputs" / "scatter_obs_vs_pred.png"

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

GROUP_LABELS = {
    "amendment":            "Amendment",
    "cropresid":            "Crop residue",
    "covercrop":            "Cover crop",
    "covercrop_amendment":  "Cover crop + amendment",
    "covercrop_cropresid":  "Cover crop + residue",
    "covercrop_pruning":    "Cover crop + pruning",
    "grass":                "Grass",
    "grass_annuals":        "Grass (annuals)",
    "grass_pruning":        "Grass + pruning",
}

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

MS_TRAIN = 40   # marker size for train / all cases
MS_TEST  = 45   # slightly larger for test diamonds (they're visually smaller)
ALPHA    = 0.80


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params_from_checkpoint(path):
    ckpt = json.loads(Path(path).read_text())
    return list(ckpt["params"].keys()), list(ckpt["params"].values())


def run_model(param_names, param_values, data):
    """Return DataFrame with case, observed, predicted, group_calib."""
    _, details = objective(param_values, param_names, data, return_details=True)
    comp = details["comparison_df"].copy()
    comp = comp.rename(columns={
        "delta_soc_t_ha_y":                  "observed",
        "delta_treatment_control_per_year":  "predicted",
    })
    comp = comp.merge(data["cases_info_df"][["case", "group_calib"]], on="case")
    return comp[["case", "observed", "predicted", "group_calib"]]


def metrics(df):
    res  = df["predicted"] - df["observed"]
    rmse = np.sqrt(np.mean(res**2))
    bias = np.mean(res)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((df["observed"] - df["observed"].mean())**2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, bias, r2


def draw_panel(ax, df, title,
               test_cases=None,
               show_envelope=False, envelope_rmse=None):
    """Draw one scatter panel onto ax."""

    # Axis range — pad 10% beyond data range, keep square
    all_vals = pd.concat([df["observed"], df["predicted"]])
    lo = all_vals.min() - 0.1 * all_vals.abs().max()
    hi = all_vals.max() + 0.1 * all_vals.abs().max()
    pad = (hi - lo) * 0.05
    lo -= pad; hi += pad

    # ±RMSE envelope
    if show_envelope and envelope_rmse is not None:
        x_line = np.array([lo, hi])
        ax.fill_between(x_line,
                        x_line - envelope_rmse,
                        x_line + envelope_rmse,
                        color="grey", alpha=0.10, zorder=0,
                        label=f"±RMSE_cal ({envelope_rmse:.2f})")

    # 1:1 line
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.9, ls="--", alpha=0.55, zorder=1)

    # Scatter per group
    for grp in GROUP_ORDER:
        sub = df[df["group_calib"] == grp]
        if sub.empty:
            continue
        color = GROUP_COLORS[grp]

        if test_cases is not None:
            # Split train / test within group
            is_test = sub["case"].isin(test_cases)
            train = sub[~is_test]
            test  = sub[ is_test]
            if not train.empty:
                ax.scatter(train["observed"], train["predicted"],
                           s=MS_TRAIN, marker="o",
                           color=color, alpha=ALPHA, zorder=3,
                           linewidths=0.4, edgecolors="white")
            if not test.empty:
                ax.scatter(test["observed"], test["predicted"],
                           s=MS_TEST, marker="D",
                           color=color, alpha=ALPHA, zorder=4,
                           linewidths=0.5, edgecolors="white")
        else:
            ax.scatter(sub["observed"], sub["predicted"],
                       s=MS_TRAIN, marker="o",
                       color=color, alpha=ALPHA, zorder=3,
                       linewidths=0.4, edgecolors="white")

    # Metrics
    rmse, bias, r2 = metrics(df)
    stats_txt = (
        f"RMSE = {rmse:.3f}\n"
        f"Bias = {bias:+.3f}\n"
        f"R²   = {r2:.3f}\n"
        f"n    = {len(df)}"
    )
    ax.text(0.04, 0.97, stats_txt,
            transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.80, lw=0.5))

    # Formatting
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(labelsize=8)
    ax.axhline(0, color="grey", lw=0.4, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.4, alpha=0.5)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading and precomputing model data...")
data = precompute_data(repo_root=BASE_DIR)

# Cal-val split
calval_split = pd.read_csv(BASE_DIR / "outputs" / "calval_split.csv")
test_cases   = set(calval_split.loc[calval_split["split"] == "test", "case"])

# ── Run three param sets ──────────────────────────────────────────────────────

print("Running Default...")
default_names  = list(PARAM_CONFIG.keys())
default_values = [PARAM_CONFIG[p]["default"] for p in default_names]
df_default = run_model(default_names, default_values, data)

print("Running Phase 2...")
p2_names, p2_values = load_params_from_checkpoint(
    BASE_DIR / "outputs" / "phase2_sequential_checkpoints" / "all.json"
)
df_phase2 = run_model(p2_names, p2_values, data)

print("Running Cal-Val...")
cv_names, cv_values = load_params_from_checkpoint(
    BASE_DIR / "outputs" / "calval_checkpoints" / "all.json"
)
df_calval = run_model(cv_names, cv_values, data)

# Cal-Val metrics split
df_calval_train = df_calval[~df_calval["case"].isin(test_cases)]
df_calval_test  = df_calval[ df_calval["case"].isin(test_cases)]
rmse_cal, bias_cal, r2_cal = metrics(df_calval_train)
rmse_val, bias_val, r2_val = metrics(df_calval_test)

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.subplots_adjust(wspace=0.32, bottom=0.22)

draw_panel(axes[0], df_default, "Default parameters")

draw_panel(axes[1], df_phase2,  "Phase 2 — sequential calibration\n(all 70 cases)")

# Cal-Val panel has two metric boxes: calibration + validation
draw_panel(axes[2], df_calval,
           "Cal-Val — calibration set\n● train (n=47)  ◆ test (n=23)",
           test_cases=test_cases,
           show_envelope=True,
           envelope_rmse=rmse_cal)

# Override the single metric box on the cal-val panel with train+test split
axes[2].texts[-1].remove()   # remove the auto-added single box
stats_txt = (
    f"Calibration (train, n={len(df_calval_train)})\n"
    f"  RMSE={rmse_cal:.3f}  Bias={bias_cal:+.3f}  R²={r2_cal:.3f}\n"
    f"\nValidation (test, n={len(df_calval_test)})\n"
    f"  RMSE={rmse_val:.3f}  Bias={bias_val:+.3f}  R²={r2_val:.3f}"
)
axes[2].text(0.04, 0.97, stats_txt,
             transform=axes[2].transAxes,
             fontsize=7.5, va="top", ha="left",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.80, lw=0.5))

# Shared axis labels
for ax in axes:
    ax.set_xlabel("Observed  Δ SOC  (t C ha⁻¹ yr⁻¹)", fontsize=9)
axes[0].set_ylabel("Predicted  Δ SOC  (t C ha⁻¹ yr⁻¹)", fontsize=9)

fig.suptitle(
    "RothC calibration — Observed vs Predicted Δ SOC",
    fontsize=12, fontweight="bold", y=1.01
)

# ── Legend ────────────────────────────────────────────────────────────────────

group_handles = [
    mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
    for g in GROUP_ORDER
]
marker_handles = [
    plt.scatter([], [], s=MS_TRAIN, marker="o", color="grey", alpha=0.8,
                edgecolors="white", linewidths=0.4, label="Train / all cases"),
    plt.scatter([], [], s=MS_TEST,  marker="D", color="grey", alpha=0.8,
                edgecolors="white", linewidths=0.5, label="Validation (test)"),
    plt.Line2D([0], [0], color="black", lw=0.9, ls="--", alpha=0.55, label="1:1 line"),
    mpatches.Patch(color="grey", alpha=0.20, label="±RMSE_cal envelope"),
]

fig.legend(
    handles=group_handles + marker_handles,
    loc="lower center",
    ncol=6,
    fontsize=8,
    framealpha=0.9,
    bbox_to_anchor=(0.5, -0.06),
)

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {OUTPUT_PNG}")
