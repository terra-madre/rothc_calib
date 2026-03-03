"""
run_calibration.py
------------------
Main orchestrator for the RothC calibration pipeline.

Controls every step from data preparation to cross-validation via a
single command. Each step can be toggled on/off with ``--steps``.
Warm-start and checkpoint resumption are preserved.

Steps
  prepare           — Run prepare_data.py (compute derived files + outlier filtering)
  sequential_groups — Sequential group optimization
  calval            — Calibration-validation (stratified 70/30 split)
  kfold             — Stratified K-fold cross-validation
  postprocess       — Run post scripts against selected output directory
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).resolve().parents[1]

ALL_STEPS = ['prepare', 'sequential_groups', 'calval', 'kfold', 'postprocess']

DEFAULT_CONFIG = {
    "steps": ['prepare', 'sequential_groups', 'calval', 'kfold', 'postprocess'],
    "exclude_outliers": True,
    "prepare": {
        "do_preprocess_cases": True,
        "do_get_st_yields": True,
        "do_compute_derived": True,
        "do_remove_outliers": True,
    },
    "postprocess": {
        "enabled": True,
        "scripts": [
            "calc_model_uncertainty.py",
            "plot_scatter_obs_vs_pred.py",
            "plot_residuals.py",
            "plot_obs_vs_pred.py",
            "plot_sequential_groups.py",
        ],
    },
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="RothC calibration pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        '--steps', default=None,
        help=f'Comma-separated list of steps to run. Available: {",".join(ALL_STEPS)}',
    )
    ap.add_argument(
        '--config', default=None,
        help='Optional JSON config file (e.g. inputs/optimization/run_calibration_config.json).',
    )
    ap.add_argument(
        '--exclude-outliers',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Override outlier handling (true/false). '
             'When omitted, value comes from config exclude_outliers.',
    )
    return ap.parse_args()


def resolve_path(path_str, base):
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else base / path


def load_config(config_path):
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path is None:
        return config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    user_cfg = json.loads(config_path.read_text())
    for key in ("steps", "exclude_outliers"):
        if key in user_cfg:
            config[key] = user_cfg[key]


    if "prepare" in user_cfg and isinstance(user_cfg["prepare"], dict):
        config["prepare"].update(user_cfg["prepare"])
    if "postprocess" in user_cfg and isinstance(user_cfg["postprocess"], dict):
        config["postprocess"].update(user_cfg["postprocess"])
    return config


def run_postprocess(output_dir, proc_subdir, scripts):
    for script_name in scripts:
        script_path = BASE_DIR / "git_code" / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Postprocess script not found: {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "--output-dir",
            str(output_dir),
        ]
        if proc_subdir:
            cmd.extend(["--proc-subdir", proc_subdir])

        print(f"Running postprocess: {script_name}")
        subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    config_path = resolve_path(args.config, BASE_DIR)
    cfg = load_config(config_path)

    steps_value = args.steps if args.steps is not None else cfg["steps"]
    if isinstance(steps_value, str):
        steps = [s.strip() for s in steps_value.split(',') if s.strip()]
    else:
        steps = [str(s).strip() for s in steps_value if str(s).strip()]

    for s in steps:
        if s not in ALL_STEPS:
            print(f"ERROR: unknown step '{s}'. Choose from: {ALL_STEPS}")
            sys.exit(1)

    if args.exclude_outliers is not None:
        exclude_outliers = bool(args.exclude_outliers)
    else:
        exclude_outliers = bool(cfg.get("exclude_outliers", True))
    outliers_mode = "exclude" if exclude_outliers else "include"

    output_dir = BASE_DIR / "outputs" / ("no_outliers" if exclude_outliers else "fullset")
    output_dir.mkdir(parents=True, exist_ok=True)

    proc_subdir = "no_outliers" if exclude_outliers else None

    print("RothC Calibration Pipeline")
    print(f"  Steps:       {steps}")
    print(f"  Outliers:    {outliers_mode}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Proc subdir: {proc_subdir or '(default)'}")
    if config_path:
        print(f"  Config:      {config_path}")
    print()

    if 'prepare' in steps:
        from prepare_data import run as run_prepare

        prepare_cfg = dict(cfg.get("prepare", {}))
        prepare_cfg["do_remove_outliers"] = exclude_outliers

        print("=" * 70)
        print("STEP: prepare")
        print("=" * 70)
        run_prepare(
            do_preprocess_cases=prepare_cfg.get("do_preprocess_cases", False),
            do_get_st_yields=prepare_cfg.get("do_get_st_yields", False),
            do_compute_derived=prepare_cfg.get("do_compute_derived", True),
            do_remove_outliers=prepare_cfg.get("do_remove_outliers", False),
        )
        print()

    needs_data = any(s in steps for s in ('sequential_groups', 'calval', 'kfold'))
    data = None
    if needs_data:
        from optimization import precompute_data
        print("Loading precomputed data...")
        data = precompute_data(repo_root=BASE_DIR, proc_subdir=proc_subdir)
        print(f"  {len(data['cases_info_df'])} cases loaded\n")

    if 'sequential_groups' in steps:
        from run_sequential_groups import run_sequential_groups

        print("=" * 70)
        print("STEP: sequential_groups")
        print("=" * 70)
        run_sequential_groups(data, output_dir)
        print()

    if 'calval' in steps:
        from run_calval_singlesplit import run_calval

        warmstart = output_dir / "sequential_groups_checkpoints" / "all.json"
        print("=" * 70)
        print("STEP: calval")
        print("=" * 70)
        run_calval(data, output_dir, warmstart_path=warmstart)
        print()

    if 'kfold' in steps:
        from run_kfold import run_kfold

        warmstart = output_dir / "sequential_groups_checkpoints" / "all.json"
        print("=" * 70)
        print("STEP: kfold")
        print("=" * 70)
        run_kfold(data, output_dir, warmstart_path=warmstart)
        print()

    if 'postprocess' in steps:
        post_cfg = dict(cfg.get("postprocess", {}))
        if not post_cfg.get("enabled", False):
            print("=" * 70)
            print("STEP: postprocess (skipped; set postprocess.enabled=true in config)")
            print("=" * 70)
            print()
        else:
            print("=" * 70)
            print("STEP: postprocess")
            print("=" * 70)
            run_postprocess(
                output_dir=output_dir,
                proc_subdir=proc_subdir,
                scripts=post_cfg.get("scripts", []),
            )
            print()

    print("=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
