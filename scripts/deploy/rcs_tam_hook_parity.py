#!/usr/bin/env python3
"""Numeric parity check between the RCS-fork C++ ``TamHook`` and the JAX torque adaptor.

Two subcommands so the JAX side (Sim2realAdaptor env) and the RCS side
(``rcs_panda``/``rcs_fr3`` ``tam`` fork env) can run in different Python
environments:

  export  -- load a TAM checkpoint, export its adaptor to the controller .bin,
             draw random adaptor inputs (local window + embedding) and store the
             JAX residual for each case (fixture.npz);
  check   -- feed the same windows row-by-row through ``hw.TamHook`` (exact
             production code path incl. gravity handling) and compare residuals.

Example::

  # TAM (JAX) environment
  python scripts/deploy/rcs_tam_hook_parity.py export \
      --ckpt <checkpoint_dir> --out outputs/rcs_tam_hook_parity
  # RCS environment (rcs_panda / rcs_fr3 from the tam branch importable)
  RCS_PREFIX=<rcs checkout> python scripts/deploy/rcs_tam_hook_parity.py check \
      --dir outputs/rcs_tam_hook_parity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def cmd_export(args: argparse.Namespace) -> int:
    from simadaptor.deploy.inf_util import SimAdaptorInference

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    inf = SimAdaptorInference(simadaptor_ckpt_path=str(args.ckpt), xml_path=args.xml)
    bin_path = out / "adaptor.bin"
    inf.export_simadaptor_weights_cpp(bin_path)
    cfg = inf.cfg
    dof = 7
    T = int(inf.adaptor_seq_length)
    emb_dim = int(getattr(cfg, "emb_dim"))
    # The public release ships jointwise_flat adaptors only.
    adaptor_mode = "jointwise_flat"
    jointwise = True
    ideal_model_has_gravity = bool(getattr(inf, "ideal_model_has_gravity", True))
    rng = np.random.default_rng(int(args.seed))
    n = int(args.num_cases)
    home = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
    q = home[None, None, :] + rng.normal(scale=0.3, size=(n, T, dof)).astype(np.float32)
    dq = rng.normal(scale=0.5, size=(n, T, dof)).astype(np.float32)
    gravity = rng.normal(scale=3.0, size=(n, T, dof)).astype(np.float32)
    u = rng.normal(scale=4.0, size=(n, T, dof)).astype(np.float32)  # model-space torque (with gravity)
    u[np.abs(u) < 1e-3] = 1e-3  # keep every row valid_for_history
    if jointwise:
        z = rng.normal(scale=float(args.embedding_scale), size=(n, dof, emb_dim)).astype(np.float32)
    else:
        z = rng.normal(scale=float(args.embedding_scale), size=(n, emb_dim)).astype(np.float32)
    deltas = []
    for i in range(n):
        # SimAdaptorInference.adaptor() returns the corrected torque tau^0 + delta.
        tau_c = np.asarray(inf.adaptor(q[i], dq[i], u[i], z[i]), dtype=np.float32).reshape(-1)[:dof]
        deltas.append(tau_c - u[i, -1, :dof])
    delta_jax = np.stack(deltas, axis=0)
    # float64 reference: the adaptor is numerically sensitive for a few inputs, so the
    # check calibrates its tolerance with the float32-vs-float64 spread per case.
    delta_jax64 = _jax_float64_reference(inf, q, dq, u, z, dof, jointwise)
    np.savez(
        out / "fixture.npz",
        q=q,
        dq=dq,
        u=u,
        gravity=gravity,
        z_flat=z.reshape(n, -1),
        delta_jax=delta_jax,
        delta_jax64=delta_jax64,
    )
    meta = {
        "ckpt": str(args.ckpt),
        "bin": str(bin_path),
        "dof": dof,
        "history_steps": T,
        "emb_dim": emb_dim,
        "adaptor_mode": adaptor_mode,
        "ideal_model_has_gravity": ideal_model_has_gravity,
        "num_cases": n,
        "seed": int(args.seed),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[export] bin={bin_path} T={T} emb_dim={emb_dim} mode={adaptor_mode} cases={n} -> {out / 'fixture.npz'}")
    print(f"[export] |delta_jax| mean={np.abs(delta_jax).mean():.4f} max={np.abs(delta_jax).max():.4f}")
    return 0


def _jax_float64_reference(inf, q, dq, u, z, dof, jointwise) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    if not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)
    inf._ensure_checkpoint_loaded()
    params64 = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=jnp.float64), inf._simadaptor_params["adaptor"])
    norm64 = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=jnp.float64), inf._norm_stats)
    model = inf._simadaptor_model[1]
    out = []
    for i in range(q.shape[0]):
        z_i = z[i].reshape(dof, -1) if jointwise else z[i]
        d64, _ = model.apply(
            params64,
            jnp.asarray(q[i][None], jnp.float64),
            jnp.asarray(dq[i][None], jnp.float64),
            jnp.asarray(u[i][None], jnp.float64),
            jnp.asarray(z_i[None], jnp.float64),
            norm_stats=norm64,
        )
        out.append(np.asarray(d64, dtype=np.float64).reshape(-1)[:dof])
    return np.stack(out, axis=0)


def _import_hw(extension: str):
    if extension == "panda":
        from rcs_panda._core import hw  # type: ignore
    else:
        from rcs_fr3._core import hw  # type: ignore
    return hw


def cmd_check(args: argparse.Namespace) -> int:
    d = Path(args.dir).expanduser().resolve()
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    fx = np.load(d / "fixture.npz")
    hw = _import_hw(args.extension)
    hook = hw.TamHook(4096)
    if not hook.load_adaptor(str(d / "adaptor.bin")):
        print("[check] FAILED to load adaptor bin")
        return 2
    st = hook.status()
    T = int(meta["history_steps"])
    dof = int(meta["dof"])
    if int(st["history_steps"]) != T:
        print(f"[check] history_steps mismatch: bin={st['history_steps']} fixture={T}")
        return 2
    grav_flag = bool(meta["ideal_model_has_gravity"])
    hook.set_ideal_model_has_gravity(grav_flag)
    hook.set_enable_ramp_s(0.0)
    hook.set_torque_limits(np.full(7, 1e6))
    hook.enable(True)
    q, dq, u, g, z = fx["q"], fx["dq"], fx["u"], fx["gravity"], fx["z_flat"]
    delta_jax = fx["delta_jax"]
    delta_jax64 = fx["delta_jax64"] if "delta_jax64" in fx.files else None
    n = q.shape[0]
    worst = 0.0
    worst_calibrated = 0.0
    rows = []
    for i in range(n):
        hook.on_control_start()
        hook.set_embedding(np.asarray(z[i], dtype=np.float64))
        delta = None
        for t in range(T):
            grav = np.asarray(g[i, t], dtype=np.float64)
            tau_model = np.asarray(u[i, t], dtype=np.float64)
            tau_base = tau_model - grav if grav_flag else tau_model
            delta = np.asarray(
                hook.apply(0.001, np.asarray(q[i, t], dtype=np.float64), np.asarray(dq[i, t], dtype=np.float64),
                           tau_base, grav, np.zeros(7), np.zeros(7))
            )
            hook.finalize_row(tau_base)
        assert delta is not None
        st = hook.status()
        if st["last_skip_reason"]:
            print(f"[check] case {i}: adaptor skipped: {st['last_skip_reason']}")
            return 2
        tau_hist = np.asarray(hook.last_tau_hist()).reshape(-1)
        expected_hist = np.asarray(u[i], dtype=np.float32).reshape(-1)
        hist_err = float(np.max(np.abs(tau_hist - expected_hist)))
        err = float(np.max(np.abs(delta[:dof] - delta_jax[i][:dof])))
        worst = max(worst, err)
        if delta_jax64 is not None:
            spread32 = np.abs(delta_jax[i][:dof] - delta_jax64[i][:dof])
            err64 = np.abs(delta[:dof] - delta_jax64[i][:dof])
            # C++ float32 vs JAX float64, allowing the JAX float32 spread of this case.
            calibrated = float(np.max(np.maximum(err64 - 3.0 * spread32, 0.0)))
            worst_calibrated = max(worst_calibrated, calibrated)
            extra = f"  max|cpp - jax64|={float(err64.max()):.3e} (jax32 spread {float(spread32.max()):.3e})"
        else:
            extra = ""
        rows.append((i, err, hist_err))
        print(f"[check] case {i}: max|delta_cpp - delta_jax|={err:.3e}  max|tau_hist - u|={hist_err:.3e}  |delta|max={np.abs(delta_jax[i]).max():.3f}{extra}")
    # Clip semantics: default limits clip a large residual.
    hook.set_torque_limits(np.asarray([10.0, 10.0, 10.0, 10.0, 8.0, 8.0, 8.0]))
    hook.on_control_start()
    hook.set_embedding(np.asarray(z[0], dtype=np.float64))
    for t in range(T):
        delta = np.asarray(hook.apply(0.001, q[0, t].astype(np.float64), dq[0, t].astype(np.float64),
                                      (u[0, t] - g[0, t]).astype(np.float64) if grav_flag else u[0, t].astype(np.float64),
                                      g[0, t].astype(np.float64), np.zeros(7), np.zeros(7)))
        hook.finalize_row(u[0, t].astype(np.float64))
    clipped_ok = bool(np.all(np.abs(delta[:4]) <= 10.0 + 1e-9) and np.all(np.abs(delta[4:7]) <= 8.0 + 1e-9))
    if delta_jax64 is not None:
        ok = worst_calibrated <= float(args.atol) and clipped_ok
        print(f"[check] worst residual parity error vs jax32 {worst:.3e} Nm; calibrated excess vs jax64 "
              f"{worst_calibrated:.3e} Nm (atol {args.atol}); clip semantics ok={clipped_ok}; "
              f"adaptor_forward_dt_ms={hook.status()['adaptor_forward_dt_ms']:.3f}")
    else:
        ok = worst <= float(args.atol) and clipped_ok
        print(f"[check] worst residual parity error {worst:.3e} Nm (atol {args.atol}); clip semantics ok={clipped_ok}; "
              f"adaptor_forward_dt_ms={hook.status()['adaptor_forward_dt_ms']:.3f}")
    print("[check] PASS" if ok else "[check] FAIL")
    return 0 if ok else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--xml", default=None)
    e.add_argument("--out", required=True)
    e.add_argument("--num-cases", type=int, default=8)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--embedding-scale", type=float, default=0.5)
    e.set_defaults(func=cmd_export)
    c = sub.add_parser("check")
    c.add_argument("--dir", required=True)
    c.add_argument("--extension", choices=("panda", "fr3"), default="panda")
    c.add_argument("--atol", type=float, default=5e-3, help="Allowed C++-vs-JAX64 excess beyond 3x the JAX float32 spread (Nm).")
    c.set_defaults(func=cmd_check)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
