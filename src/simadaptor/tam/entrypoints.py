"""TAM console entrypoints.

These commands expose the public Torque Adaptation Module workflow and forward
to the small set of scripts kept in this release branch.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
import runpy

from .presets import PUBLIC_ROBOT_KEYS, resolve_robot, robot_choices_help


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_script(script_path: str, argv: Sequence[str] | None = None) -> None:
    path = REPO_ROOT / script_path
    if not path.exists():
        raise SystemExit(f"Script not found: {path}")
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(path), *(list(argv) if argv is not None else sys.argv[1:])]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old_argv


def _split_extra(raw: str | None) -> list[str]:
    return [] if raw is None or not str(raw).strip() else shlex.split(str(raw))


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _robot_key_or_custom(token: str) -> str:
    try:
        return resolve_robot(token).key
    except ValueError:
        custom = str(token).strip()
        if not custom:
            raise
        return custom


def _add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool | None,
    help: str,
) -> None:
    parser.add_argument(
        name,
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help,
    )


def generate_data(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate TAM rollout datasets for one or more robot presets.",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        default=["panda_pandagripper"],
        help=f"Robot presets to generate. Choices: {robot_choices_help()}",
    )
    parser.add_argument("--dataset-base-path", required=True, help="Dataset root shared by all robot subdirectories.")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--history-batch", type=int, default=None)
    parser.add_argument("--history-duration", type=float, default=None)
    parser.add_argument("--num-waypoints-history", type=int, default=None)
    parser.add_argument("--sim-timestep", type=float, default=None)
    parser.add_argument("--external-force-num-impulses", type=int, default=None)
    _add_bool_arg(
        parser,
        "--save-original-split",
        default=None,
        help="Also save the unperturbed rollout split.",
    )
    parser.add_argument(
        "--extra",
        default="",
        help="Quoted extra args forwarded to scripts/data/generate_dataset.py.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print generated commands without running them.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    for robot_token in args.robots:
        preset = resolve_robot(robot_token)
        forwarded = [
            "--xml-path",
            str(preset.xml_path),
            "--dataset-base-path",
            str(args.dataset_base_path),
            "--datagen-profile-key",
            preset.profile_key,
        ]
        optional_pairs = [
            ("--num-steps", args.num_steps),
            ("--history-batch", args.history_batch),
            ("--history-duration", args.history_duration),
            ("--num-waypoints-history", args.num_waypoints_history),
            ("--sim-timestep", args.sim_timestep),
            ("--external-force-num-impulses", args.external_force_num_impulses),
        ]
        for flag, value in optional_pairs:
            if value is not None:
                forwarded.extend([flag, str(value)])
        if args.save_original_split is not None:
            forwarded.append("--save-original-split" if args.save_original_split else "--no-save-original-split")
        forwarded.extend(_split_extra(args.extra))
        if args.dry_run:
            print("python scripts/data/generate_dataset.py " + shlex.join(forwarded))
            continue
        print(f"[tam-generate-data] robot={preset.key} xml={preset.xml_path}")
        _run_script("scripts/data/generate_dataset.py", forwarded)


def _build_train_argv(args: argparse.Namespace, robot_keys: Sequence[str]) -> list[str]:
    forwarded = [
        "--history-batch",
        str(args.history_batch),
        "--training-seq-length",
        str(args.training_seq_length),
        "--tau-map-sample-no",
        str(args.tau_map_sample_no),
        "--emb-dim",
        str(args.emb_dim),
        "--tam-hidden",
        str(args.tam_hidden),
        "--tam-seq-length",
        str(args.tam_seq_length),
        "--ckpt.workdir",
        str(args.ckpt_workdir),
        "--ckpt.max-to-keep",
        str(args.ckpt_max_to_keep),
        "--wandb.mode",
        str(args.wandb_mode),
        "--data.dataset-base-path",
        str(args.dataset_base_path),
        "--robot-key",
        *list(robot_keys),
    ]
    if args.run_name:
        forwarded.extend(["--ckpt.run-name", str(args.run_name)])
    if args.wandb_project:
        forwarded.extend(["--wandb.project", str(args.wandb_project)])
    if args.max_steps is not None:
        forwarded.extend(["--max-steps", str(args.max_steps)])
    if args.ckpt_interval is not None:
        forwarded.extend(["--ckpt-interval", str(args.ckpt_interval)])
    if args.num_workers is not None:
        forwarded.extend(["--num-workers", str(args.num_workers)])
    if args.num_data_limit is not None:
        forwarded.extend(["--num-data-limit", str(args.num_data_limit)])
    if args.hz_filter:
        forwarded.extend(["--hz-filter", *(str(v) for v in args.hz_filter)])
    forwarded.extend(_split_extra(args.extra))
    return forwarded


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--ckpt-workdir", default="checkpoints/tam")
    parser.add_argument("--ckpt-max-to-keep", type=int, default=0)
    parser.add_argument("--history-batch", type=int, default=64)
    parser.add_argument("--training-seq-length", type=int, default=4)
    parser.add_argument("--tau-map-sample-no", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--tam-seq-length", type=int, default=8)
    parser.add_argument("--tam-hidden", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ckpt-interval", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-data-limit", type=int, default=None)
    parser.add_argument("--hz-filter", type=int, nargs="*", default=[200, 500, 1000])
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--extra",
        default="",
        help="Quoted extra args forwarded to scripts/train/tam/train.py.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the generated command without running it.")


def train_multi_robot(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train TAM on multiple robot datasets.")
    parser.add_argument(
        "--robots",
        nargs="+",
        default=list(PUBLIC_ROBOT_KEYS),
        help=f"Robot presets or custom dataset robot keys. Presets: {robot_choices_help()}",
    )
    _add_train_args(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    robot_keys = [_robot_key_or_custom(token) for token in args.robots]
    forwarded = _build_train_argv(args, robot_keys)
    if args.dry_run:
        print("python scripts/train/tam/train.py " + shlex.join(forwarded))
        return
    _run_script("scripts/train/tam/train.py", forwarded)


def train_robot(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train or finetune TAM on one robot dataset.")
    parser.add_argument(
        "--robot",
        default="panda_pandagripper",
        help=f"Robot preset or custom dataset robot key. Presets: {robot_choices_help()}",
    )
    _add_train_args(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    robot_key = _robot_key_or_custom(args.robot)
    forwarded = _build_train_argv(args, [robot_key])
    if args.dry_run:
        print("python scripts/train/tam/train.py " + shlex.join(forwarded))
        return
    _run_script("scripts/train/tam/train.py", forwarded)


def dagger_finetune(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Online DAgger-style finetuning of a trained TAM checkpoint.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Base TAM checkpoint (run directory containing save_dict.pkl).",
    )
    parser.add_argument(
        "--robot-preset",
        default="panda_pandagripper",
        help=f"Robot preset providing the XML and datagen profile. Choices: {robot_choices_help()}",
    )
    parser.add_argument(
        "--history-torque-mode",
        default="base_tam_fusion",
        choices=("applied", "base_tam_fusion"),
        help="Online torque-history conditioning: applied-torque only, or fused applied/base/TAM-residual streams.",
    )
    parser.add_argument(
        "--attention-history-s",
        type=float,
        default=None,
        help="Optional bounded attention history window in seconds.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Number of finetuning steps.")
    parser.add_argument("--wandb-mode", default="disabled", choices=("online", "offline", "disabled"))
    parser.add_argument("--outdir", default="checkpoints/tam_online_dagger")
    parser.add_argument(
        "--extra",
        default="",
        help="Quoted extra args forwarded to scripts/train/tam/dagger_finetune.py.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the generated command without running it.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    preset = resolve_robot(args.robot_preset)
    forwarded = [
        "--ckpt",
        str(args.ckpt),
        "--xml",
        str(preset.xml_path),
        "--profile-key",
        preset.profile_key,
        "--history-torque-mode",
        str(args.history_torque_mode),
        "--wandb-mode",
        str(args.wandb_mode),
        "--workdir",
        str(args.outdir),
    ]
    if args.attention_history_s is not None:
        forwarded.extend(["--attention-history-s", str(args.attention_history_s)])
    if args.steps is not None:
        forwarded.extend(["--max-steps", str(args.steps)])
    forwarded.extend(_split_extra(args.extra))
    if args.dry_run:
        print("python scripts/train/tam/dagger_finetune.py " + shlex.join(forwarded))
        return
    print(f"[tam-dagger-finetune] robot={preset.key} ckpt={args.ckpt}")
    _run_script("scripts/train/tam/dagger_finetune.py", forwarded)


def mapping_server(argv: Sequence[str] | None = None) -> None:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if not _has_flag(forwarded, "--backend"):
        forwarded = ["--backend", "tam", *forwarded]
    _run_script("scripts/deploy/mapping_server.py", forwarded)


def eval_wrapper(argv: Sequence[str] | None = None) -> None:
    _run_script("scripts/deploy/trajectory_tracking_eval.py", list(sys.argv[1:] if argv is None else argv))


def eval_source_to_osc_sim(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the representative simulated source-to-OSC TAM evaluation.",
    )
    parser.add_argument("--robot-preset", default="panda")
    parser.add_argument("--conditions", nargs="+", default=["direct_osc", "tam_carried"])
    parser.add_argument("--sim-backend", default="auto", choices=("auto", "legacy", "batched"))
    parser.add_argument("--num-iterations", type=int, default=8)
    parser.add_argument("--outdir", type=Path, default=Path("eval_logs") / "source_to_osc_tam_sim")
    parser.add_argument("--tam-ckpt-path", type=Path, default=None)
    parser.add_argument("--source-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--extra",
        default="",
        help="Quoted extra args forwarded to scripts/deploy/source_to_osc_tam_sim.py.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    sim_backend = str(args.sim_backend)
    if sim_backend == "auto" and args.source_only:
        # The experiment's auto backend resolution only inspects condition keys
        # and would pick the batched table backend for the default conditions,
        # but the batched backend does not implement --source-only.
        sim_backend = "legacy"
    forwarded = [
        "--robot-preset",
        str(args.robot_preset),
        "--conditions",
        *list(args.conditions),
        "--sim-backend",
        sim_backend,
        "--num-iterations",
        str(args.num_iterations),
        "--outdir",
        str(args.outdir),
    ]
    if args.tam_ckpt_path is not None:
        forwarded.extend(["--tam-ckpt-path", str(args.tam_ckpt_path)])
    if args.source_only:
        forwarded.append("--source-only")
    if args.dry_run:
        print("python scripts/deploy/source_to_osc_tam_sim.py " + shlex.join([*forwarded, "--dry-run"]))
        return
    forwarded.extend(_split_extra(args.extra))
    _run_script("scripts/deploy/source_to_osc_tam_sim.py", forwarded)
