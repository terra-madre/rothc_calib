"""
plot_residuals.py
-----------------
Plots model residuals (predicted − observed Δ SOC) against case number
for the calibrated parameter sets.

Two vertical panels:
  Top    — Phase 2 params (all 70 cases)
  Bottom — Cal-Val params (train ● / test ◆ distinguished)

Each panel:
  - Case number on x-axis (sorted by group, then case ID)
  - Residual on y-axis
  - Zero line + ±ME band (grey shading)
  - Coloured by calibration group
  - ME / RMSE / Bias annotation box

Output: outputs/residuals_vs_case.png
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "git_code"))

from calc_model_uncertainty import calc_model_uncertainty
from optimization import precompute_data

# ── Style (shared with other plot scripts) ─────────────────────────────────────

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

OUTPUT_PNG = BASE_DIR / "outputs" / "residuals_vs_case.png"

# ── helpers ────────────────────────────────────────────────────────────────────

def load_params(path):
    return json.loads(Path(path).read_text())["params"]


def build_df(res: dict, cases_info_df: pd.DataFrame) -> pd.DataFrame:
    """Attach group_calib to the comparison dataframe."""
    df = res["comparison"].copy()
    df = df.merge(cases_info_df[["case", "group_calib"]], on="case", how="left")
    # Sort by group order then case id for consistent x-axis
    df["group_order"] = df["group_calib"].map(
        {g: i for i, g in enumerate(GROUP_ORDER)}
    )
    df = df.sort_values(["group_order", "case"]).reset_index(drop=True)
    df["x"] = range(len(df))          # sequential x position
    return df


def draw_panel(ax, df, me, rmse, bias, title,
               train_cases=None, test_cases=None):
    """Draw a single residuals panel."""
    # ±ME band
    ax.axhspan(-me, me, color="lightgrey", alpha=0.50, zorder=0, label=f"±ME ({me:.2f})")
    ax.axhline(0, color="black", linewidth=0.8, zorder=1)

    n = len(df)

    for _, row in df.iterrows():
        color = GROUP_COLORS.get(row["group_calib"], "grey")
        x     = row["x"]
        y     = row["residual"]

        if test_cases is not None:
            # Cal-Val panel: distinguish train / test
            if row["case"] in test_cases:
                ax.scatter(x, y, marker="D", s=40, color=color, alpha=0.85,
                           linewidths=0.4, edgecolors="white", zorder=3)
            else:
                ax.scatter(x, y, marker="o", s=30, color=color, alpha=0.80,
                           linewidths=0.4, edgecolors="white", zorder=3)
        else:
            ax.scatter(x, y, marker="o", s=30, color=color, alpha=0.80,
                       linewidths=0.4, edgecolors="white", zorder=3)

    # Group separator lines + labels at top
    prev_group = None
    group_starts = {}
    for _, row in df.iterrows():
        g = row["group_calib"]
        if g != prev_group:
            if prev_group is not None:
                ax.axvline(row["x"] - 0.5, color="grey", linewidth=0.5,
                           linestyle="--", alpha=0.5, zorder=2)
            group_starts[g] = row["x"]
            prev_group = g

    # Group label ticks at bottom
    group_centers = {}
    for i, g in enumerate(GROUP_ORDER):
        rows = df[df["group_calib"] == g]
        if len(rows):
            group_centers[g] = (rows["x"].min() + rows["x"].max()) / 2

    # Individual case ID ticks
    ax.set_xticks(df["x"].tolist())
    ax.set_xticklabels(df["case"].astype(int).tolist(),
                       rotation=90, ha="center", fontsize=5.5)
    ax.set_xlim(-0.8, n - 0.2)

    # Group labels below case-number ticks using blended (data-x, axes-y) transform
    blended = matplotlib.transforms.blended_transform_factory(
        ax.transData, ax.transAxes
    )
    for g, cx in group_centers.items():
        ax.text(cx, -0.22, GROUP_LABELS.get(g, g),
                transform=blended,
                ha="center", va="top", fontsize=6.5, rotation=30,
                color=GROUP_COLORS.get(g, "grey"), fontweight="bold")

    # Annotation box
    bias_sign = "+" if bias >= 0 else ""
    textstr = (
        f"n = {n}\n"
        f"ME   = {me:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"Bias = {bias_sign}{bias:.3f}"
    )
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8, ec="lightgrey"))

    ax.set_ylabel("Residual (pred − obs)\n[t C ha⁻¹ yr⁻¹]", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ckpt_dir = BASE_DIR / "outputs"

    print("Loading data …")
    data = precompute_data()
    cases_info_df = data["cases_info_df"]

    sequential_groups_params = load_params(ckpt_dir / "sequential_groups_checkpoints" / "all.json")
    calval_params = load_params(ckpt_dir / "calval_checkpoints" / "all.json")

    splits_df   = pd.read_csv(ckpt_dir / "calval_split.csv")
    train_cases = set(splits_df.loc[splits_df["split"] == "train", "case"])
    test_cases  = set(splits_df.loc[splits_df["split"] == "test",  "case"])

    print("Running model for sequential_groups params …")
    res_p2 = calc_model_uncertainty(sequential_groups_params, data=data)
    df_p2  = build_df(res_p2, cases_info_df)

    print("Running model for Cal-Val params …")
    res_cv = calc_model_uncertainty(calval_params, data=data)
    df_cv  = build_df(res_cv, cases_info_df)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
    fig.subplots_adjust(hspace=0.70, bottom=0.28)

    draw_panel(
        axes[0], df_p2,
        me=res_p2["me"], rmse=res_p2["rmse"], bias=res_p2["bias"],
        title="Sequential groups — calibrated on all 70 cases"
    )

    draw_panel(
        axes[1], df_cv,
        me=res_cv["me"], rmse=res_cv["rmse"], bias=res_cv["bias"],
        title="Cal-Val params — evaluated on all 70 cases  (● train  ◆ test)",
        train_cases=train_cases, test_cases=test_cases
    )

    # ── shared group colour legend ────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
        for g in GROUP_ORDER
    ]
    # Add train/test marker legend entries
    legend_handles += [
        plt.scatter([], [], marker="o", s=30, color="grey", label="Train"),
        plt.scatter([], [], marker="D", s=30, color="grey", label="Test"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=7.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle("Model Residuals vs Case  (pred − obs Δ SOC)", fontsize=11, y=0.99)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUTPUT_PNG.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
