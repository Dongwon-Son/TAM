# TAM on robot-control-stack (RCS)

TAM can run underneath [robot-control-stack](https://github.com/RobotControlStack/robot-control-stack)
(RCS) on a Franka Panda/FR3. The controller side lives in the RCS fork
[`Dongwon-Son/robot-control-stack`](https://github.com/Dongwon-Son/robot-control-stack),
branch **`tam`**:

* a C++ `TamHook` inside `hw::Franka`'s 1 kHz torque controllers
  (`osc()` / `joint_controller()`), applying the TAM residual between the RCS
  torque law and its rate-limit/clamp tail while recording the 1 kHz history;
* a pure-Python bridge extension `extensions/rcs_tam` (`python -m rcs_tam`)
  that serves the TAM history/embedding ZMQ protocol, so the workstation side
  of this repository (`tam-mapping-server`, the evaluation wrapper) works
  unchanged.

```
 workstation (GPU)                        control PC (RT kernel)                    Franka
 ┌──────────────────────────┐    ZMQ     ┌─────────────────────────────────────┐   ┌─────────────┐
 │ tam-mapping-server        │ 5555 PUB  │ python -m rcs_tam (extensions/rcs_tam)│   │ FCI         │
 │  (history encoder → z)    │◄──────────┤   bridge + RcsBackend (Python)       │◄─▶│ 1 kHz torque│
 │ tam-eval-wrapper / policy ├──────────►│ RCS hw.Franka + TamHook (C++)        │FCI│             │
 └──────────────────────────┘ 5556/5557  └─────────────────────────────────────┘   └─────────────┘
```

Torque path inside RCS (async torque mode, `async_control=True`):

```
tau_base = <RCS law>                       # gravity-free; libfranka adds gravity
tau_d    = tau_base + TamHook.apply(...)   # TAM residual (SimAdaptor .bin, ~0.1 ms)
tau_d    = limitRate(...); clamp(torque_limit)
TamHook.finalize_row(tau_d)                # 1 kHz history row for the workstation
```

## Controller side

Build the fork on the realtime control PC and run the bridge (details in the
fork's `extensions/rcs_tam/README.md` and `extensions/rcs_fr3/README.md`):

```bash
git clone -b tam https://github.com/Dongwon-Son/robot-control-stack.git
cd robot-control-stack
pip install -ve . --no-build-isolation
pip install -ve extensions/rcs_panda --no-build-isolation    # or extensions/rcs_fr3
pip install -e extensions/rcs_tam
export RCS_PREFIX=$PWD
python -m rcs_tam --robot panda --config rcs_tam_config.json
```

Important: raise `FrankaConfig.torque_limit` — RCS clamps the whole
gravity-free command to 5 Nm per joint by default, which clips the TAM
residual. The bridge default is `87,87,87,87,12,12,12`; the TAM residual is
additionally clipped per joint (default `10,10,10,10,8,8,8`).

## Workstation side

Unchanged — point the mapping server at the bridge (see
[deployment.md](deployment.md) for all options):

```bash
tam-mapping-server \
  --ckpt-path <tam checkpoint dir> \
  --xml assets/franka_panda/panda_pandagripper.xml \
  --history-endpoint tcp://<controller-host>:5555 \
  --command-endpoint tcp://<controller-host>:5556 \
  --request-endpoint tcp://<controller-host>:5557 \
  --attention-history-s 4.0 --history-torque-mode auto --send-bin \
  --embedding-interval-s 0.2 --enable-after-first-embedding
```

Protocol features the RCS bridge does not implement (SysID torque maps,
external-torque prediction, soft-block, test disturbances, feedforward torque,
live gain changes) are answered `ok=False, unsupported=True`; disabling them is
a no-op, so the mapping server's prepare sequence completes normally. RCS reads
joint gains when a control thread starts, so `stiffness`/`damping` commands
apply on the next controller (re)start.

## Verifying without a robot

Residual parity — the C++ `TamHook` against the JAX adaptor of a checkpoint
(run `export` in this repo's environment, `check` in the RCS environment):

```bash
python scripts/deploy/rcs_tam_hook_parity.py export \
    --ckpt <tam checkpoint dir> --xml assets/franka_panda/panda_pandagripper.xml \
    --out outputs/rcs_tam_hook_parity --num-cases 32
python scripts/deploy/rcs_tam_hook_parity.py check --dir outputs/rcs_tam_hook_parity
```

Closed loop with a MuJoCo Panda in place of the robot — the bridge, the RCS
torque-law replica, and the C++ hook run against the real mapping server:

```bash
python scripts/deploy/rcs_tam_bridge_sim_smoke.py --extension panda \
    --duration-s 300 --payload-mass-delta 1.0 \
    --summary-out outputs/rcs_tam_bridge_sim_smoke/summary.json
# second terminal: the tam-mapping-server command above with tcp://127.0.0.1 endpoints
```

Simulation study of TAM under RCS's torque law — the source-to-OSC evaluator
can replay RCS's `joint_controller()`/`osc()` (20 Hz setpoint interpolation,
error deadbands, static-posture nullspace, joint-limit avoidance, rate limit,
`torque_limit` clamp, libfranka command low-pass) instead of the TAM reference
law:

```bash
tam-eval-source-to-osc-sim --conditions direct_osc tam_carried \
    --controller-law rcs --rcs-torque-limit 87,87,87,87,12,12,12 \
    --tam-ckpt-path <tam checkpoint dir> --num-iterations 5
```

The MuJoCo replica of the RCS controllers lives in
`src/simadaptor/deploy/rcs_controller_replica.py`; the bridge backend that
steps it at 1 kHz is `src/simadaptor/deploy/rcs_mujoco_backend.py`.
