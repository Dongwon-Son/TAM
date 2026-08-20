# TAM Deployment

This repo includes the workstation-side TAM mapping server and the external
trajectory-evaluation wrapper. The low-level robot controller bridge runs on the
deployment controller machine and is not included here; a ready-made controller
side for robot-control-stack (Franka Panda/FR3) is available — see
[rcs_integration.md](rcs_integration.md).

## Connection Roles

| Endpoint | Example | Direction | Purpose |
| --- | --- | --- | --- |
| `--history-endpoint` | `tcp://<controller-host>:5555` | controller PUB -> mapping/eval SUB | Streams history windows, reset events, joint state, and torque fields. |
| `--command-endpoint` | `tcp://<controller-host>:5556` | mapping/eval PUSH -> controller PULL | Sends target commands, TAM embeddings, and feedforward fields. |
| `--request-endpoint` | `tcp://<controller-host>:5557` | mapping/eval REQ -> controller REP | Sends reliable requests such as loading TAM weights and enabling/disabling TAM. |
| `--control-endpoint` | `tcp://127.0.0.1:5560` (default) | operator/eval REQ -> mapping server REP | Sends direct mapping-server commands such as reset, hold, resume, and status. |

Always pass the first three endpoints explicitly with your controller host's
address; the built-in defaults point at a placeholder LAN address. Use a bind
address for `--control-endpoint` on the workstation that runs
`tam-mapping-server` (for example `tcp://0.0.0.0:5560` to accept remote
control connections; the default binds localhost only).

## Mapping Server

```bash
tam-mapping-server \
  --ckpt-path checkpoints/tam/tam_rby1 \
  --xml assets/rby1a/rby1_onearm.xml \
  --history-endpoint tcp://<controller-host>:5555 \
  --command-endpoint tcp://<controller-host>:5556 \
  --request-endpoint tcp://<controller-host>:5557 \
  --control-endpoint tcp://0.0.0.0:5560 \
  --require-control-enable
```

The server waits for controller history, prepares the TAM runtime, uploads the
exported TAM weight blob when `--send-bin` is enabled, and only enables TAM after
the first valid embedding unless configured otherwise.

`--require-control-enable` keeps TAM disabled until an operator or evaluation
wrapper sends a control command to the mapping-server control endpoint. This is
the recommended deployment default because it prevents stale history from
enabling TAM before the robot is homed.

## Fused-History Deployment

DAgger-finetuned checkpoints trained with the fused input mode
(`base_tam_fusion`) consume three parallel torque-history streams: applied
torque, base-policy torque, and TAM residual torque. `--history-torque-mode`
defaults to `auto` and resolves the mode from checkpoint metadata
(`dagger_cfg` plus `params['history_fusion']`); the server refuses to run
fused when the checkpoint lacks fusion weights.

```bash
tam-mapping-server \
  --ckpt-path checkpoints/tam_online_dagger/<run-name>/checkpoint_<step> \
  --history-endpoint tcp://<controller-host>:5555 \
  --command-endpoint tcp://<controller-host>:5556 \
  --request-endpoint tcp://<controller-host>:5557 \
  --control-endpoint tcp://0.0.0.0:5560 \
  --history-torque-mode base_tam_fusion \
  --attention-history-s 4.0 \
  --history-buffer 6000 \
  --min-patches-before-send 2 \
  --embedding-interval-s 0.2 \
  --send-bin \
  --enable-after-first-embedding \
  --reset-on-controller-reset
```

- `--history-torque-mode {auto,applied,base_tam_fusion}` (default `auto`)
  selects the torque-history input mode.
- `--attention-history-s` bounds the attention window in seconds (default
  4.0); values `<= 0` keep the full cache.
- `--require-explicit-fused-history` (default on) enforces the controller-side
  history contract below; `--no-require-explicit-fused-history` disables the
  check.

To pre-warm the JAX compile cache with deployment-shaped inputs before going
live, run `python scripts/deploy/prepare_history_encoder_cache.py --ckpt-path
<checkpoint> --attention-history-s 4.0` on the deployment workstation.

Controller history contract: in fused mode each history row must carry explicit
`tau_base` and `tau_adaptor_delta` fields plus `publish_ready=True`. Fused
windows missing these fields are rejected and the server reports health state
`fused_history_contract_missing`.

Buffer sizing: fused deployments need a larger client buffer than applied-mode
deployments to cover the 4 s attention window. Raise `--history-buffer` from its
default of `500` to `6000`; `--min-patches-before-send` can stay at its default
of `2`.

## Evaluation Wrapper

```bash
tam-eval-wrapper \
  --reference sine \
  --history-endpoint tcp://<controller-host>:5555 \
  --command-endpoint tcp://<controller-host>:5556 \
  --request-endpoint tcp://<controller-host>:5557
```

The wrapper talks to the same controller bridge as the mapping server. Keep the
mapping server running when evaluating TAM-enabled control; run the wrapper
without a TAM mapping server only for direct-controller comparisons.

## Startup Order

1. Start the deployment-side controller bridge on the robot controller machine.
2. Start `tam-mapping-server` on the workstation with the matching checkpoint
   and robot XML.
3. Confirm the mapping server reports fresh history and a prepared TAM runtime.
4. Send a reset/resume command through the mapping-server control endpoint or
   start `tam-eval-wrapper`.
5. Stop the evaluation wrapper before stopping the mapping server or controller
   bridge.

## Network Notes

- Keep clocks reasonably synchronized so history timestamps and evaluation logs
  are easy to inspect.
- The controller bridge should bind the history, command, and request ports.
  The workstation connects to those ports.
- The mapping server binds the control port. Operator tools and evaluation
  scripts connect to it.
- Use SSH tunnels or firewall rules when the controller network is not directly
  reachable from the workstation.
