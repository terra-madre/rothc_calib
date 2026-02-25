"""
plot_report_figures.py
----------------------
Generates five supplementary report figures and saves them to report/.

  fig3_calval_diagram.png        – Cal-val procedure flowchart
  fig4_sequential_flow.png       – Sequential calibration sub-run chain
  fig5_param_changes.png         – Parameter % change bar chart
  fig6_residual_histogram.png    – Calibration vs validation residual histogram
  fig7_group_errors.png          – Per-group RMSE and bias (cal-val params)

Run from project root:
    python -u git_code/plot_report_figures.py
"""

import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUTDIR  = ROOT / "report"
sys.path.insert(0, str(Path(__file__).parent))

PALETTE = {
    "blue_light":   "#DBEAFE",
    "blue":         "#2563EB",
    "green_light":  "#D1FAE5",
    "green":        "#059669",
    "red_light":    "#FEE2E2",
    "red":          "#DC2626",
    "grey_light":   "#F3F4F6",
    "grey":         "#6B7280",
    "purple_light": "#EDE9FE",
    "purple":       "#7C3AED",
    "orange_light": "#FEF3C7",
    "orange":       "#D97706",
    "border":       "#374151",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared flowchart helpers
# ═══════════════════════════════════════════════════════════════════════════════

def box(ax, cx, cy, text, w=2.4, h=0.52, fc="#DBEAFE", fontsize=8.5,
        bold=False, ec=None):
    """Draw a rounded box centred at (cx, cy)."""
    ec = ec or PALETTE["border"]
    p  = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.06", linewidth=0.9,
        edgecolor=ec, facecolor=fc, zorder=3,
        clip_on=False
    )
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight,
            multialignment="center", zorder=4)


def arrow(ax, x1, y1, x2, y2, color=None, style="->"):
    color = color or PALETTE["border"]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=1.1),
        zorder=2
    )


def label(ax, x, y, text, fontsize=7.5, color=None, ha="center"):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=fontsize, color=color or PALETTE["grey"],
            style="italic", zorder=4)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — Cal-Val procedure diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_calval_diagram():
    fig, ax = plt.subplots(figsize=(8.0, 11.5))
    ax.set_xlim(-1.0, 8.5); ax.set_ylim(-0.5, 10.1)
    ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    # ── Row 1: full dataset ────────────────────────────────────────────────
    box(ax, 4, 9.4, "Full dataset\n70 paired cases",
        w=2.8, h=0.55, fc=PALETTE["blue_light"], bold=True)

    arrow(ax, 4, 9.12, 4, 8.67)

    # ── Row 2: split ───────────────────────────────────────────────────────
    box(ax, 4, 8.38, "Stratified 70/30 split\nby calibration super-group  |  seed = 42",
        w=4.6, h=0.55, fc=PALETTE["grey_light"])

    # Two branches
    arrow(ax, 2.8, 8.11, 2.2, 7.63)   # left → train
    arrow(ax, 5.2, 8.11, 5.8, 7.63)   # right → test

    # ── Row 3: train / test ────────────────────────────────────────────────
    box(ax, 2.0, 7.35, "Training set\nn = 47 cases",
        w=2.2, h=0.52, fc=PALETTE["green_light"])
    box(ax, 6.0, 7.35, "Test set\nn = 23 cases\n(held out)",
        w=2.2, h=0.65, fc=PALETTE["red_light"])

    arrow(ax, 2.0, 7.08, 2.0, 6.52)   # train → calibration

    # test set arrow skips down to validation sim (drawn later)
    ax.annotate("", xy=(6.0, 4.42), xytext=(6.0, 7.02),
                arrowprops=dict(arrowstyle="->", color=PALETTE["border"],
                                lw=1.1, connectionstyle="arc3,rad=0"), zorder=2)

    # ── Row 4: calibration algorithm ──────────────────────────────────────
    box(ax, 2.0, 6.18, "Sequential DE calibration\n6 sub-runs, warm-starting\n"
        "Objective: RMSE(Δ SOC)",
        w=3.2, h=0.72, fc=PALETTE["blue_light"])

    arrow(ax, 2.0, 5.82, 2.0, 5.37)

    # ── Row 5: calibrated params ───────────────────────────────────────────
    box(ax, 2.0, 5.08, "Calibrated parameters\n(10 values)",
        w=2.6, h=0.52, fc=PALETTE["purple_light"])

    # Two branches from params
    arrow(ax, 1.2, 4.82, 1.1, 4.45)   # → cal sim
    arrow(ax, 2.8, 4.82, 5.05, 4.45)  # → val sim

    # ── Row 6: simulations ─────────────────────────────────────────────────
    box(ax, 1.0, 4.15, "Forward sim.\non training set",
        w=2.1, h=0.52, fc=PALETTE["green_light"])
    box(ax, 5.4, 4.15, "Forward sim.\non test set",
        w=2.1, h=0.52, fc=PALETTE["red_light"])

    arrow(ax, 1.0, 3.88, 1.0, 3.40)
    arrow(ax, 5.4, 3.88, 5.4, 3.40)

    # ── Row 7: metrics ────────────────────────────────────────────────────
    box(ax, 1.0, 3.08,
        "Calibration metrics\nRMSE = 0.924\nR² = 0.812 | Bias = +0.051",
        w=2.5, h=0.72, fc=PALETTE["green_light"], fontsize=8)
    box(ax, 5.4, 3.08,
        "Validation metrics\nRMSE = 0.592\nR² = 0.572 | Bias = +0.124",
        w=2.5, h=0.72, fc=PALETTE["red_light"], fontsize=8)

    # PI from cal metrics
    arrow(ax, 1.0, 2.72, 1.0, 2.22)

    # ── Row 8: PI ─────────────────────────────────────────────────────────
    box(ax, 1.0, 1.92, "90% Prediction Interval\n± 1.645 × RMSE_cal = ± 1.52",
        w=2.8, h=0.52, fc=PALETTE["grey_light"])

    # Both PI and val metrics → acceptance
    arrow(ax, 1.0, 1.66, 2.25, 0.97)
    arrow(ax, 5.4, 2.72, 4.1, 0.97)

    # ── Row 9: acceptance criteria ─────────────────────────────────────────
    box(ax, 3.2, 0.62,
        "Acceptance criteria\n"
        "C1 -- bias <= PMU:  [N/A]\n"
        "C2 -- >=90% PI coverage:  [PASS]\n"
        "C3 -- R2 > 0 on test set:  [PASS]",
        w=3.8, h=0.82, fc=PALETTE["orange_light"], bold=False, fontsize=8.5)

    ax.set_title("Figure 3. Calibration–Validation Procedure",
                 fontsize=11, fontweight="bold", pad=6)

    fig.tight_layout()
    out = OUTDIR / "fig3_calval_diagram.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — Sequential calibration sub-run chain
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_sequential_flow():
    RUNS = [
        ("1 — amendment",      "n = 15",  "dr_ratio_fym",                          PALETTE["green_light"]),
        ("2 — cropresid",      "n = 9",   "residue_frac_remaining",                PALETTE["green_light"]),
        ("3 — covercrop_all",  "n = 29",  "dr_ratio_annuals\ncc_yield_mod\ndecomp_mod", PALETTE["blue_light"]),
        ("4 — grass_all",      "n = 12",  "grass_rsr_b\nturnover_bg_grass\ndr_ratio_treegrass", PALETTE["orange_light"]),
        ("5 — pruning",        "n = 5",   "dr_ratio_wood",                         PALETTE["purple_light"]),
        ("6 — all (final)",    "n = 70",  "All 10 params\n+ plant_cover_modifier\n+ L-BFGS-B polish",
         "#C7F7D4"),
    ]
    RMSE = [0.976, 0.371, 1.078, 0.401, 0.953, 0.823]

    fig, ax = plt.subplots(figsize=(8.5, 12.5))
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(-0.5, 11.2)
    ax.axis("off"); fig.patch.set_facecolor("white")

    # Starting box: Phase 2 initial params (warm-start source)
    box(ax, 4, 10.4, "Phase 2 warm-start: default RothC parameters",
        w=4.8, h=0.50, fc=PALETTE["grey_light"], fontsize=8.5)

    y_top = 9.65
    step_h = 1.45   # vertical spacing between runs

    for i, (run_name, cases, params, fc) in enumerate(RUNS):
        cy = y_top - i * step_h

        # Arrow from above
        arrow(ax, 4, cy + 0.55 + (0.18 if i == 0 else 0.0), 4, cy + 0.52)

        # Main box (run name + cases)
        run_h = 0.52
        box(ax, 4, cy, f"Sub-run {run_name}  |  {cases}",
            w=4.6, h=run_h, fc=fc, bold=True, fontsize=9)

        # Parameters sidebar
        n_lines = params.count("\n") + 1
        param_h = 0.22 + n_lines * 0.20
        box(ax, 6.55, cy, f"Params:\n{params}",
            w=2.1, h=param_h, fc="white", fontsize=7.5, ec=PALETTE["grey"])

        # RMSE badge
        ax.text(1.1, cy, f"RMSE\n{RMSE[i]:.3f}",
                ha="center", va="center", fontsize=7.5,
                color=PALETTE["border"],
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=PALETTE["grey"], lw=0.7))

        # Warm-start label on arrows (except last)
        if i < len(RUNS) - 1:
            label(ax, 4.62, cy - run_h/2 - 0.22, "warm-start ↓")

    # Final arrow into output
    cy_last = y_top - (len(RUNS) - 1) * step_h
    arrow(ax, 4, cy_last - 0.52, 4, 0.57)

    box(ax, 4, 0.27, "Final calibrated parameter set  (10 values)",
        w=4.2, h=0.44, fc=PALETTE["blue_light"], bold=True, fontsize=9)

    ax.set_title("Figure 4. Sequential Calibration Sub-Run Chain",
                 fontsize=11, fontweight="bold", pad=6)
    fig.tight_layout()
    out = OUTDIR / "fig4_sequential_flow.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — Parameter % change bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_param_changes():
    cfg = pd.read_csv(ROOT / "inputs/optimization/param_config.csv")

    phase2_json = ROOT / "outputs/phase2_sequential_checkpoints/all.json"
    calval_json = ROOT / "outputs/calval_checkpoints/all.json"
    p2 = json.loads(phase2_json.read_text())["params"]
    cv = json.loads(calval_json.read_text())["params"]

    rows = []
    for _, r in cfg.iterrows():
        name = r["name"]
        if name not in p2:
            continue
        default = r["default"]
        p2_val  = p2[name]
        cv_val  = cv.get(name, np.nan)
        if abs(default) < 1e-9:
            continue
        rows.append(dict(
            name=name,
            description=r["description"].split("(")[0].strip(),
            default=default,
            p2_pct=100 * (p2_val - default) / abs(default),
            cv_pct=100 * (cv_val - default) / abs(default),
        ))

    df = pd.DataFrame(rows)
    df = df.sort_values("p2_pct")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    y  = np.arange(len(df))
    bh = 0.32

    bars_p2 = ax.barh(y + bh/2, df["p2_pct"], height=bh, label="Phase 2 (n=70)",
                      color="#2563EB", alpha=0.85, zorder=3)
    bars_cv = ax.barh(y - bh/2, df["cv_pct"], height=bh, label="Cal-Val train (n=47)",
                      color="#059669", alpha=0.75, zorder=3)

    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["name"], fontsize=9, fontfamily="monospace")
    ax.set_xlabel("% change from default value", fontsize=9)
    ax.set_title("Figure 5. Calibrated Parameter Values Relative to RothC Defaults",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(axis="x", lw=0.5, alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels
    for bar in bars_p2:
        w = bar.get_width()
        ax.text(w + (1.5 if w >= 0 else -1.5), bar.get_y() + bar.get_height()/2,
                f"{w:+.0f}%", va="center", ha="left" if w >= 0 else "right",
                fontsize=7, color="#2563EB")

    fig.tight_layout()
    out = OUTDIR / "fig5_param_changes.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared: load predictions
# ═══════════════════════════════════════════════════════════════════════════════

def load_predictions():
    from optimization import precompute_data, objective, PARAM_CONFIG

    data = precompute_data(repo_root=ROOT)

    split_df = pd.read_csv(ROOT / "outputs/calval_split.csv")
    test_cases  = set(split_df[split_df["split"] == "test"]["case"])
    train_cases = set(split_df[split_df["split"] == "train"]["case"])

    def run(json_path):
        params_d = json.loads(Path(json_path).read_text())["params"]
        names  = list(params_d.keys())
        values = list(params_d.values())
        _, details = objective(values, names, data, return_details=True)
        comp = details["comparison_df"].copy()
        comp = comp.rename(columns={
            "delta_soc_t_ha_y":                 "obs",
            "delta_treatment_control_per_year": "pred",
        })
        comp = comp.merge(data["cases_info_df"][["case", "group_calib"]], on="case")
        comp["residual"] = comp["pred"] - comp["obs"]
        return comp

    df_cv = run(ROOT / "outputs/calval_checkpoints/all.json")
    df_cv["set"] = df_cv["case"].apply(
        lambda c: "test" if c in test_cases else "train"
    )
    return df_cv, train_cases, test_cases


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — Residual histogram
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_residual_histogram(df_cv):
    train = df_cv[df_cv["set"] == "train"]["residual"].values
    test  = df_cv[df_cv["set"] == "test"]["residual"].values

    lim   = max(abs(np.concatenate([train, test]))) * 1.10
    bins  = np.linspace(-lim, lim, 18)
    x_fit = np.linspace(-lim, lim, 200)

    from scipy.stats import norm

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), sharey=False)
    fig.patch.set_facecolor("white")

    for ax, resid, label_str, color, n in zip(
        axes,
        [train, test],
        ["Calibration (train)", "Validation (test)"],
        ["#2563EB", "#DC2626"],
        [len(train), len(test)],
    ):
        mu, sd = resid.mean(), resid.std()
        ax.hist(resid, bins=bins, color=color, alpha=0.55, density=True,
                edgecolor="white", lw=0.6, zorder=2, label="Observed")
        ax.plot(x_fit, norm.pdf(x_fit, mu, sd), color=color, lw=1.8,
                ls="--", zorder=3, label=f"N({mu:+.3f}, {sd:.3f})")
        ax.axvline(0, color="black", lw=0.9, ls=":", alpha=0.6)
        ax.axvline(mu, color=color, lw=1.2, ls="-", alpha=0.9,
                   label=f"Mean = {mu:+.3f}")
        ax.set_xlabel("Residual  (pred − obs)  [t C ha⁻¹ yr⁻¹]", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"{label_str}  (n = {n})", fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=7.8, framealpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.4)

    fig.suptitle("Figure 6. Model Residual Distributions — Calibration vs Validation",
                 fontsize=10.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUTDIR / "fig6_residual_histogram.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 7 — Per-group RMSE and bias
# ═══════════════════════════════════════════════════════════════════════════════

GROUP_ORDER = [
    "amendment", "cropresid",
    "covercrop", "covercrop_amendment", "covercrop_cropresid", "covercrop_pruning",
    "grass", "grass_annuals", "grass_pruning",
]
GROUP_LABELS = {
    "amendment":            "amendment (n=15)",
    "cropresid":            "cropresid (n=9)",
    "covercrop":            "covercrop (n=21)",
    "covercrop_amendment":  "cvcrp+amend (n=4)",
    "covercrop_cropresid":  "cvcrp+resid (n=4)",
    "covercrop_pruning":    "cvcrp+prun (n=3)",
    "grass":                "grass (n=4)",
    "grass_annuals":        "grass_annuals (n=8)",
    "grass_pruning":        "grass+prun (n=2)",
}

def fig7_group_errors(df_cv):
    records = []
    for grp in GROUP_ORDER:
        sub = df_cv[df_cv["group_calib"] == grp]
        if sub.empty:
            continue
        for subset_name, subset_flag in [("train", "train"), ("test", "test")]:
            s = sub[sub["set"] == subset_flag]
            if s.empty:
                continue
            res  = s["residual"].values
            rmse = np.sqrt(np.mean(res**2))
            bias = np.mean(res)
            records.append(dict(group=grp, subset=subset_name,
                                rmse=rmse, bias=bias, n=len(s)))

    df = pd.DataFrame(records)
    groups = [g for g in GROUP_ORDER if g in df["group"].values]
    y = np.arange(len(groups))
    bh = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, metric, title, xlbl in zip(
        axes,
        ["rmse", "bias"],
        ["RMSE by Group", "Mean Bias by Group"],
        ["RMSE  [t C ha⁻¹ yr⁻¹]", "Bias  (pred − obs)  [t C ha⁻¹ yr⁻¹]"],
    ):
        for subset, color, dy in [("train", "#2563EB", bh/2), ("test", "#DC2626", -bh/2)]:
            vals = []
            positions = []
            for i, grp in enumerate(groups):
                row = df[(df["group"] == grp) & (df["subset"] == subset)]
                if not row.empty:
                    vals.append(row[metric].values[0])
                    positions.append(y[i] + dy)
            if vals:
                ax.barh(positions, vals, height=bh,
                        color=color, alpha=0.78, zorder=3,
                        label=f"{'Train (n=47)' if subset=='train' else 'Test (n=23)'}")

        ax.set_yticks(y)
        ax.set_yticklabels([GROUP_LABELS.get(g, g) for g in groups], fontsize=8.5)
        ax.set_xlabel(xlbl, fontsize=9)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.5)
        ax.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle("Figure 7. Per-Group Model Performance — Cal-Val Parameter Set",
                 fontsize=10.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUTDIR / "fig7_group_errors.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.chdir(ROOT)

    print("Generating fig3 (cal-val diagram)...")
    fig3_calval_diagram()

    print("Generating fig4 (sequential flow)...")
    fig4_sequential_flow()

    print("Generating fig5 (parameter changes)...")
    fig5_param_changes()

    print("Loading model predictions for fig6 and fig7 (this may take ~30 s)...")
    df_cv, train_cases, test_cases = load_predictions()

    print("Generating fig6 (residual histogram)...")
    fig6_residual_histogram(df_cv)

    print("Generating fig7 (group errors)...")
    fig7_group_errors(df_cv)

    print("\nAll figures saved to report/")
