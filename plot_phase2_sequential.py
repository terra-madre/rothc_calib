"""
Plot Phase 2 Sequential Optimization Results.

Produces 3 figures saved to outputs/:
  1. phase2_subrun_rmse.png       - RMSE per sub-run vs baseline
  2. phase2_param_changes.png     - Optimized vs default values (all run)
  3. phase2_obs_vs_pred.png       - Obs vs pred delta SOC: baseline vs optimized
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from optimization import precompute_data, objective, PARAM_CONFIG

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"

# ─────────────────────────────────────────────
# Load results
# ─────────────────────────────────────────────
summary = pd.read_csv(OUT_DIR / "phase2_sequential_summary.csv")
params_df = pd.read_csv(OUT_DIR / "phase2_sequential_params.csv")
all_params = params_df[params_df['set_name'] == 'all'].set_index('parameter')

baseline_rmse = summary['baseline_rmse'].iloc[0]

# ─────────────────────────────────────────────
# Fig 1: Sub-run RMSE vs baseline
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

labels = summary['set_name'].tolist()
rmses  = summary['rmse'].tolist()
n_cases = summary['n_cases'].tolist()

colors = ['#4878D0', '#4878D0', '#4878D0', '#4878D0', '#4878D0', '#D65F5F']
bars = ax.barh(labels, rmses, color=colors, height=0.6, zorder=2)
ax.axvline(baseline_rmse, color='#888', linestyle='--', linewidth=1.4, label=f'Baseline RMSE = {baseline_rmse:.4f}')

# Annotate bars with RMSE and n_cases
for bar, rmse, n in zip(bars, rmses, n_cases):
    n_label = f'n={int(n)}' if pd.notna(n) else 'all'
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{rmse:.4f}  ({n_label})', va='center', fontsize=9)

ax.set_xlabel('RMSE (t C/ha/y)', fontsize=11)
ax.set_title('Phase 2: RMSE per Sub-run vs Baseline', fontsize=13, fontweight='bold')
ax.set_xlim(0, baseline_rmse * 1.25)
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.4, zorder=1)
ax.invert_yaxis()

# Highlight 'all' run label
for label, tick in zip(labels, ax.get_yticklabels()):
    if label == 'all':
        tick.set_fontweight('bold')
        tick.set_color('#D65F5F')

plt.tight_layout()
fig.savefig(OUT_DIR / "phase2_subrun_rmse.png", dpi=150)
plt.close()
print("Saved: phase2_subrun_rmse.png")


# ─────────────────────────────────────────────
# Fig 2: Parameter changes (all run)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

param_order = all_params.index.tolist()
opt_vals    = all_params['optimized_value'].values
def_vals    = all_params['default_value'].values
bound_mins  = all_params['bound_min'].values
bound_maxs  = all_params['bound_max'].values
pct_changes = all_params['pct_change'].values

y = np.arange(len(param_order))

# Left panel: absolute values with default + bounds
ax = axes[0]
ax.barh(y, opt_vals, height=0.5, color='#4878D0', zorder=2, label='Optimized')
ax.scatter(def_vals, y, marker='|', s=200, color='#222', zorder=3, linewidths=2, label='Default')
# Bound range lines
for i, (lo, hi) in enumerate(zip(bound_mins, bound_maxs)):
    ax.plot([lo, hi], [y[i], y[i]], color='#aaa', linewidth=2, zorder=1)
    ax.plot([lo, lo], [y[i]-0.2, y[i]+0.2], color='#aaa', linewidth=1.5, zorder=1)
    ax.plot([hi, hi], [y[i]-0.2, y[i]+0.2], color='#aaa', linewidth=1.5, zorder=1)

ax.set_yticks(y)
ax.set_yticklabels(param_order, fontsize=9)
ax.set_xlabel('Parameter value', fontsize=10)
ax.set_title('Optimized values\n(bars = optimized  |  lines = bounds  |  ticks = default)', fontsize=10)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.legend(fontsize=8, loc='lower right')

# Right panel: % change from default
ax = axes[1]
bar_colors = ['#D65F5F' if p < 0 else '#4CAF50' for p in pct_changes]
bars = ax.barh(y, pct_changes, height=0.5, color=bar_colors, zorder=2)
ax.axvline(0, color='#222', linewidth=1)

# At-bound markers
for i, (opt, lo, hi) in enumerate(zip(opt_vals, bound_mins, bound_maxs)):
    tol = 1e-4 * (hi - lo)
    if abs(opt - lo) < tol:
        ax.text(pct_changes[i] - 1, y[i], '◄ lower', va='center', ha='right', fontsize=7.5, color='#888')
    elif abs(opt - hi) < tol:
        ax.text(pct_changes[i] + 1, y[i], 'upper ►', va='center', ha='left', fontsize=7.5, color='#888')

ax.set_yticks(y)
ax.set_yticklabels(param_order, fontsize=9)
ax.set_xlabel('% change from default', fontsize=10)
ax.set_title('% change from default\n(all-cases run)', fontsize=10)
ax.grid(axis='x', alpha=0.3, zorder=0)

# Annotate bars
for bar, pct in zip(bars, pct_changes):
    xpos = bar.get_width() + (2 if pct >= 0 else -2)
    ha = 'left' if pct >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height()/2, f'{pct:+.1f}%', va='center', ha=ha, fontsize=8)

fig.suptitle('Phase 2: Optimized Parameters (all-cases run)', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / "phase2_param_changes.png", dpi=150)
plt.close()
print("Saved: phase2_param_changes.png")


# ─────────────────────────────────────────────
# Fig 3: Obs vs Pred — baseline vs optimized
# ─────────────────────────────────────────────
print("Computing baseline and optimized predictions...")
data = precompute_data(repo_root=BASE_DIR)
cases_info = data['cases_info_df']

# Baseline (all defaults)
default_names = list(PARAM_CONFIG.keys())
default_values = [PARAM_CONFIG[p]['default'] for p in default_names]
_, base_details = objective(default_values, default_names, data, return_details=True)
base_comp = base_details['comparison_df'].merge(cases_info[['case', 'group_calib']], on='case')

# Optimized (all-run params)
opt_names  = all_params.index.tolist()
opt_values = all_params['optimized_value'].tolist()
_, opt_details = objective(opt_values, opt_names, data, return_details=True)
opt_comp = opt_details['comparison_df'].merge(cases_info[['case', 'group_calib']], on='case')

groups = sorted(cases_info['group_calib'].unique())
cmap = plt.get_cmap('tab10')
group_colors = {g: cmap(i) for i, g in enumerate(groups)}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

for ax, comp, title, rmse, r2 in [
    (axes[0], base_comp, 'Baseline (defaults)', base_details['rmse'], base_details['r2']),
    (axes[1], opt_comp,  'Optimized (Phase 2)',  opt_details['rmse'],  opt_details['r2']),
]:
    obs = comp['delta_soc_t_ha_y']
    pred = comp['delta_treatment_control_per_year']
    for grp in groups:
        mask = comp['group_calib'] == grp
        ax.scatter(obs[mask], pred[mask], color=group_colors[grp], label=grp,
                   alpha=0.75, s=40, edgecolors='none', zorder=3)

    lims = [min(obs.min(), pred.min()) - 0.2, max(obs.max(), pred.max()) + 0.2]
    ax.plot(lims, lims, 'k--', linewidth=1, zorder=2, label='1:1')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.set_xlabel('Observed Δ SOC (t C/ha/y)', fontsize=10)
    ax.set_ylabel('Predicted Δ SOC (t C/ha/y)', fontsize=10)
    ax.set_title(f'{title}\nRMSE={rmse:.4f}  R²={r2:.4f}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, zorder=1)

# Legend on right panel only
handles = [mpatches.Patch(color=group_colors[g], label=g) for g in groups]
handles.append(plt.Line2D([0],[0], color='k', linestyle='--', label='1:1'))
axes[1].legend(handles=handles, fontsize=7.5, loc='upper left', framealpha=0.8)

fig.suptitle('Phase 2: Observed vs Predicted ΔSOC', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / "phase2_obs_vs_pred.png", dpi=150)
plt.close()
print("Saved: phase2_obs_vs_pred.png")

print("\nAll plots saved to outputs/")
