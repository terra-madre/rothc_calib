"""
run_calibration.py
------------------
Main orchestrator for the RothC calibration pipeline.

Controls every step from data preparation to cross-validation via a
single command.  Each step can be toggled on/off with ``--steps``.#1
Warm-start and checkpoint resumption are preserved.

Steps
  prepare   — Run prepare_data.py  (compute derived files + outlier filtering)
    sequential_groups — Sequential group optimization
  calval    — Calibration-validation (stratified 70/30 split)
  kfold     — Stratified K-fold cross-validation

Examples
  # Full pipeline, no-outliers dataset:
  cd git_code && conda run -n terra-plus python run_calibration.py \
      --steps prepare,sequential_groups,calval,kfold \
      --proc-subdir no_outliers --output-dir ../outputs/no_outliers

  # Only sequential_groups + calval (data already prepared):
  python run_calibration.py --steps sequential_groups,calval \
      --proc-subdir no_outliers --output-dir ../outputs/no_outliers
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).resolve().parents[1]

ALL_STEPS = ['prepare', 'sequential_groups', 'calval', 'kfold']


def parse_args():
    ap = argparse.ArgumentParser(
        description="RothC calibration pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
           '--steps', default='sequential_groups,calval,kfold',
        help=f'Comma-separated list of steps to run. '
               f'Available: {",".join(ALL_STEPS)}  (default: sequential_groups,calval,kfold)',
    )
    ap.add_argument(
        '--output-dir', default=None,
        help='Output directory (default: outputs/)',
    )
    ap.add_argument(
        '--proc-subdir', default=None,
        help='Subdirectory under inputs/processed/ to read data from '
             '(e.g. "no_outliers").  When set with --steps=prepare, '
             'outlier filtering is enabled automatically.',
    )
    return ap.parse_args()


def main():
    args = parse_args()
    steps = [s.strip() for s in args.steps.split(',')]

    for s in steps:
        if s not in ALL_STEPS:
            print(f"ERROR: unknown step '{s}'.  Choose from: {ALL_STEPS}")
            sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"RothC Calibration Pipeline")
    print(f"  Steps:       {steps}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Proc subdir: {args.proc_subdir or '(default)'}")
    print()

    # ── Step 0: Data preparation ──────────────────────────────────────────────
    if 'prepare' in steps:
        from prepare_data import run as run_prepare

        # If the user requested a proc_subdir like 'no_outliers', automatically
        # enable outlier filtering so the subset directory is populated.
        do_remove_outliers = (args.proc_subdir == 'no_outliers')

        print("=" * 70)
        print("STEP: prepare")
        print("=" * 70)
        run_prepare(
            do_preprocess_cases=False,
            do_get_st_yields=False,
            do_compute_derived=True,
            do_remove_outliers=do_remove_outliers,
        )
        print()

    # ── Load data (shared by sequential_groups, calval, kfold) ───────────────
    needs_data = any(s in steps for s in ('sequential_groups', 'calval', 'kfold'))
    if needs_data:
        from optimization import precompute_data
        print("Loading precomputed data...")
        data = precompute_data(repo_root=BASE_DIR, proc_subdir=args.proc_subdir)
        print(f"  {len(data['cases_info_df'])} cases loaded\n")

    # ── Sequential Groups ─────────────────────────────────────────────────────
    if 'sequential_groups' in steps:
        from run_sequential_groups import run_sequential_groups

        print("=" * 70)
        print("STEP: sequential_groups")
        print("=" * 70)
        run_sequential_groups(data, output_dir)
        print()

    # ── Cal-Val ───────────────────────────────────────────────────────────────
    if 'calval' in steps:
        from run_calval_singlesplit import run_calval

        warmstart = output_dir / "sequential_groups_checkpoints" / "all.json"
        print("=" * 70)
        print("STEP: calval")
        print("=" * 70)
        run_calval(data, output_dir, warmstart_path=warmstart)
        print()

    # ── K-Fold CV ─────────────────────────────────────────────────────────────
    if 'kfold' in steps:
        from run_kfold import run_kfold

        warmstart = output_dir / "sequential_groups_checkpoints" / "all.json"
        print("=" * 70)
        print("STEP: kfold")
        print("=" * 70)
        run_kfold(data, output_dir, warmstart_path=warmstart)
        print()

    print("=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
