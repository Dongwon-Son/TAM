#!/usr/bin/env python3
"""Level-3 system smoke: TAM NUC bridge + RCS-law MuJoCo backend (+ C++ TamHook), no robot.

Starts ``rcs_tam.TamBridge`` (RCS fork, ``extensions/rcs_tam``) on top of
:class:`simadaptor.deploy.rcs_mujoco_backend.RcsMujocoBackend` (RCS joint-PD /
OSC replica stepping a MuJoCo Panda at 1 kHz wall-clock, RCS-fork ``hw.TamHook``
in the loop when ``rcs_panda``/``rcs_fr3`` is importable) and streams a
sinusoidal joint reference to it at 20 Hz like an RCS policy would.  Run the
unchanged workstation ``mapping_server.py`` against the same endpoints in a
second process to exercise the full history -> embedding -> residual loop::

    # process A (RCS env: rcs_panda + rcs_tam installed, or --rcs-root <checkout>)
    python scripts/deploy/rcs_tam_bridge_sim_smoke.py --rcs-root ~/research/robot-control-stack \
        --host 127.0.0.1 --duration-s 60 --summary-out outputs/rcs_tam_bridge_sim_smoke/summary.json

    # process B (JAX env)
    python scripts/deploy/mapping_server.py --backend simadaptor \
        --history-endpoint tcp://127.0.0.1:5555 --command-endpoint tcp://127.0.0.1:5556 \
        --request-endpoint tcp://127.0.0.1:5557 --xml assets/franka_panda/panda_pandagripper.xml \
        --ckpt-path <ckpt> --attention-history-s 4.0 --history-torque-mode auto --send-bin \
        --embedding-interval-s 0.2 --enable-after-first-embedding --reset-on-controller-reset

The summary records bridge/hook counters (embedding updates received, adaptor
active ticks, residual magnitude, publish count, sim overruns) so the run can
be judged without a human watching.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_XML = ROOT / "assets/franka_panda/panda_pandagripper.xml"
HOME_Q = np.asarray([0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483])


def _import_hook(extension: str):
    if extension == "none":
        return None
    try:
        if extension == "panda":
            from rcs_panda._core import hw  # type: ignore
        else:
            from rcs_fr3._core import hw  # type: ignore
        return hw.TamHook(4096)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[smoke] RCS extension {extension!r} not importable ({exc}); using the Python history stand-in (delta=0)")
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rcs-root", type=Path, default=None,
                   help="robot-control-stack checkout (tam branch); adds extensions/rcs_tam/src to sys.path when rcs_tam is not installed.")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--history-port", type=int, default=5555)
    p.add_argument("--command-port", type=int, default=5556)
    p.add_argument("--request-port", type=int, default=5557)
    p.add_argument("--extension", choices=("panda", "fr3", "none"), default="panda")
    p.add_argument("--adaptor-bin", type=Path, default=None, help="Load this .bin directly (else wait for the mapping server upload).")
    p.add_argument("--enable-adaptor", action="store_true", help="Enable the adaptor immediately after --adaptor-bin load.")
    p.add_argument("--torque-limit", type=str, default="87,87,87,87,12,12,12")
    p.add_argument("--payload-mass-delta", type=float, default=0.0, help="kg added to the plant 'hand' body (unmodeled payload).")
    p.add_argument("--policy", choices=("hold", "sine"), default="sine")
    p.add_argument("--policy-rate-hz", type=float, default=20.0)
    p.add_argument("--sine-amp-deg", type=float, default=15.0)
    p.add_argument("--sine-period-s", type=float, default=6.0)
    p.add_argument("--duration-s", type=float, default=60.0)
    p.add_argument("--no-realtime", action="store_true", help="Step the sim as fast as possible (protocol test only).")
    p.add_argument("--summary-out", type=Path, default=None)
    p.add_argument("--status-interval-s", type=float, default=2.0)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    if args.rcs_root is not None:
        sys.path.insert(0, str((Path(args.rcs_root).expanduser().resolve() / "extensions" / "rcs_tam" / "src")))
    import zmq

    from rcs_tam.bridge import BridgeEndpoints, TamBridge

    from simadaptor.deploy.rcs_controller_replica import RcsReplicaConfig
    from simadaptor.deploy.rcs_mujoco_backend import RcsMujocoBackend

    torque_limit = tuple(float(x) for x in args.torque_limit.split(","))
    if len(torque_limit) == 1:
        torque_limit = torque_limit * 7
    hook = _import_hook(args.extension)

    def plant_modifier(model) -> None:
        if args.payload_mass_delta != 0.0:
            import mujoco

            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            if body_id < 0:
                raise ValueError("body 'hand' not found for payload perturbation")
            model.body_mass[body_id] += float(args.payload_mass_delta)
            print(f"[smoke] plant hand mass += {args.payload_mass_delta} kg")

    backend = RcsMujocoBackend(
        xml_path=str(args.xml),
        hook=hook,
        replica_cfg=RcsReplicaConfig(torque_limit=torque_limit),
        realtime=not args.no_realtime,
        plant_modifier=plant_modifier,
    )
    if args.adaptor_bin is not None:
        ok = backend.load_adaptor_path(str(args.adaptor_bin))
        print(f"[smoke] load_adaptor({args.adaptor_bin}) -> {ok}")
        if ok and args.enable_adaptor:
            backend.enable_adaptor(True)
    endpoints = BridgeEndpoints.from_host(
        args.host, history_port=args.history_port, command_port=args.command_port, request_port=args.request_port
    )
    bridge = TamBridge(backend, endpoints, startup_remote_actuation_ignore_s=0.0, status_print_interval_s=0.0)
    backend.start()
    bridge.start()
    print(f"[smoke] bridge up at {endpoints}; native_hook={backend.hook_is_native}")

    ctx = zmq.Context.instance()
    push = ctx.socket(zmq.PUSH)
    push.connect(endpoints.command_bind)
    time.sleep(0.2)
    period = 1.0 / float(args.policy_rate_hz)
    t0 = time.perf_counter()
    next_send = t0
    next_status = t0
    cmd_id = 0
    amp = np.deg2rad(float(args.sine_amp_deg)) * np.asarray([1.0, 0.6, 0.8, 0.6, 1.0, 0.8, 1.0])
    embedding_seq_seen: list[int] = []
    adaptor_active_ticks = 0
    delta_max = 0.0
    status_rows = []
    try:
        while time.perf_counter() - t0 < float(args.duration_s):
            now = time.perf_counter()
            if now >= next_send:
                t_rel = now - t0
                if args.policy == "sine":
                    q_ref = HOME_Q + amp * np.sin(2.0 * np.pi * t_rel / float(args.sine_period_s))
                else:
                    q_ref = HOME_Q
                cmd_id += 1
                push.send_json({"target_q": q_ref.tolist(), "target_dq": [0.0] * 7, "command_source": "smoke", "command_id": cmd_id})
                next_send += period
            if now >= next_status:
                st = backend.status()
                seq = int(st.get("tam_embedding_seq", -1))
                embedding_seq_seen.append(seq)
                delta_max = max(delta_max, float(st.get("sim_last_delta_max", 0.0)))
                if st.get("tam_last_skip_reason", "") == "" and bool(st.get("tam_enabled", False)):
                    adaptor_active_ticks += 1
                row = {
                    "t": round(now - t0, 2),
                    "embedding_seq": seq,
                    "enabled": bool(st.get("tam_enabled", False)),
                    "skip": st.get("tam_last_skip_reason", ""),
                    "delta_max": round(float(st.get("sim_last_delta_max", 0.0)), 4),
                    "sim_ticks": int(st.get("sim_ticks", 0)),
                    "overruns": int(st.get("sim_overruns", 0)),
                    "publishes": int(bridge.stats["publishes"]),
                    "async": int(bridge.stats["async_commands"]),
                    "reliable": int(bridge.stats["reliable_requests"]),
                    "q_err_deg": round(float(np.rad2deg(np.max(np.abs(backend.last_q - HOME_Q)))), 2),
                }
                status_rows.append(row)
                print("[smoke]", json.dumps(row))
                next_status += float(args.status_interval_s)
            time.sleep(0.002)
    finally:
        push.close(linger=0)
        bridge.stop()
        backend.stop()
    summary = {
        "native_hook": bool(backend.hook_is_native),
        "duration_s": float(args.duration_s),
        "commands_sent": cmd_id,
        "bridge_stats": dict(bridge.stats),
        "sim_ticks": int(backend.ticks),
        "sim_overruns": int(backend.overruns),
        "embedding_seq_final": int(embedding_seq_seen[-1]) if embedding_seq_seen else -1,
        "embedding_updates_observed": int(max(embedding_seq_seen) if embedding_seq_seen else 0),
        "adaptor_active_status_samples": int(adaptor_active_ticks),
        "delta_max_nm": float(delta_max),
        "status_rows": status_rows,
        "hook_status": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in backend.status().items()},
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"[smoke] summary -> {args.summary_out}")
    print(f"[smoke] done: ticks={backend.ticks} overruns={backend.overruns} publishes={bridge.stats['publishes']} "
          f"embedding_updates={summary['embedding_updates_observed']} delta_max={delta_max:.3f} Nm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
