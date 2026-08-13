# Torque Adaptation Module (TAM)

TAM is a sim-to-real dynamics adaptation toolkit for robot manipulators. It
learns a history-conditioned torque correction module from simulated rollouts
and applies the learned correction during controller-side deployment or
evaluation.

## Workflows

1. Generate TAM data for multiple robots.
2. Train one TAM model across multiple robots.
3. Train or finetune TAM for one robot.
4. DAgger-finetune a trained TAM checkpoint online with the fused
   TAM-residual input mode.
5. Run the mapping server and deployment-side evaluation wrapper.
6. Run the representative simulated source-to-OSC evaluation.

## Install

The default setup targets Linux with MuJoCo/MJX and JAX. Training uses JAX for
GPU compute and uses PyTorch only for CPU-side dataset loading, so install the
CPU PyTorch wheel before installing the repo.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -U "jax[cuda12]"
pip install -e ".[train,deploy]"
```

For CPU-only smoke tests, install CPU PyTorch the same way, then use
`pip install -e ".[cpu,train,deploy,dev]"`.
Set `WANDB_API_KEY` only when training with `--wandb-mode online`.

## Robot Presets

| Key | Robot |
| --- | --- |
| `panda_pandagripper` | Franka Panda with Panda gripper |
| `piper_description` | AgileX Piper |
| `rby1_onearm` | RBY-1 one-arm |
| `iiwa14` | KUKA iiwa14 |
| `google_robot` | Google Robot arm |
| `unitree_z1` | Unitree Z1 |
| `flexiv_rizon4` | Flexiv Rizon4 |

Short aliases such as `panda`, `piper`, `rby1`, `iiwa14`, `unitree`, and
`flexiv` are accepted by the TAM entrypoints.

## Generate Data

```bash
tam-generate-data \
  --robots panda_pandagripper piper_description rby1_onearm \
  --dataset-base-path datasets/tam \
  --num-steps 4000 \
  --history-batch 128
```

This writes one dataset subdirectory per robot. Each subdirectory contains Zarr
rollout shards plus `manifest.json` and `data_generation_config.json`.

## Train Multi-Robot TAM

```bash
tam-train-multi-robot \
  --dataset-base-path datasets/tam \
  --robots panda_pandagripper piper_description rby1_onearm \
  --run-name tam_multi_demo \
  --ckpt-workdir checkpoints/tam \
  --wandb-mode disabled \
  --max-steps 200000
```

Checkpoints are written to `checkpoints/tam/<run-name>/`.

## Train One Robot

```bash
tam-train-robot \
  --robot rby1_onearm \
  --dataset-base-path datasets/tam \
  --run-name tam_rby1 \
  --ckpt-workdir checkpoints/tam \
  --wandb-mode disabled \
  --max-steps 200000
```

To continue the same run, reuse `--run-name` and `--ckpt-workdir`; the trainer
restores the latest checkpoint in that run directory when one exists.

## Public Inference Checkpoint

A sanitized multirobot TAM inference checkpoint is available from the
[tam-public-multirobot-ckpt3472000 release](https://github.com/Dongwon-Son/TAM/releases/tag/tam-public-multirobot-ckpt3472000).

```bash
mkdir -p checkpoints/tam
curl -L \
  -o /tmp/tam_public_multirobot_ckpt3472000.tar.gz \
  https://github.com/Dongwon-Son/TAM/releases/download/tam-public-multirobot-ckpt3472000/tam_public_multirobot_ckpt3472000.tar.gz
tar -xzf /tmp/tam_public_multirobot_ckpt3472000.tar.gz -C checkpoints/tam
```

Use the extracted directory wherever a TAM checkpoint path is required:

```bash
--tam-ckpt-path checkpoints/tam/tam_public_multirobot_ckpt3472000
```

## DAgger Finetuning

`tam-dagger-finetune` runs an online DAgger finetune of a trained TAM
checkpoint. By default it trains with the fused TAM-residual input mode
(`base_tam_fusion`); pass `--history-torque-mode applied` to keep the
applied-torque-only input.

```bash
tam-dagger-finetune \
  --ckpt checkpoints/tam/tam_public_multirobot_ckpt3472000 \
  --robot-preset panda \
  --history-torque-mode base_tam_fusion \
  --attention-history-s 4.0 \
  --wandb-mode disabled
```

In fused mode the model consumes three parallel torque-history streams —
applied torque, base-policy torque, and TAM residual torque — each encoded by
the same weight-shared history encoder, with the three embeddings combined by a
small trained linear fusion layer. The history-torque mode is stored in
checkpoint metadata (`dagger_cfg` plus `params['history_fusion']`) and is
resolved automatically at load time; the attention window is NOT auto-resolved,
so pass the same `--attention-history-s` to the mapping server at deployment
that you used for training (both the example above and the deployment default
use 4.0 s).

Key knobs are `--history-torque-mode {applied,base_tam_fusion}` and
`--attention-history-s` (attention window in seconds; unbounded when unset —
pass 4.0 as in the example to match the deployment default); `--wandb-mode`
defaults to `disabled`, and `--extra`/`--dry-run` forward arguments like the
`tam-train-robot` entrypoint. Checkpoints are written to
`checkpoints/tam_online_dagger/<run-name>/`.

## Mapping Server And Evaluation Wrapper

See [docs/deployment.md](docs/deployment.md) for endpoint roles and deployment
connection details.

```bash
tam-mapping-server \
  --ckpt-path checkpoints/tam/tam_rby1 \
  --xml assets/rby1a/rby1_onearm.xml \
  --history-endpoint tcp://<controller-host>:5555 \
  --command-endpoint tcp://<controller-host>:5556 \
  --request-endpoint tcp://<controller-host>:5557 \
  --control-endpoint tcp://0.0.0.0:5560
```

For a fused (`base_tam_fusion`) checkpoint, raise the history buffer to cover
the attention window:

```bash
tam-mapping-server \
  --ckpt-path checkpoints/tam_online_dagger/<run-name> \
  --history-torque-mode auto \
  --attention-history-s 4.0 \
  --history-buffer 6000 \
  --min-patches-before-send 2 \
  --history-endpoint tcp://<controller-host>:5555 \
  --command-endpoint tcp://<controller-host>:5556 \
  --request-endpoint tcp://<controller-host>:5557 \
  --control-endpoint tcp://0.0.0.0:5560
```

`--history-torque-mode auto` (the default) resolves the mode from checkpoint
metadata. In fused mode the server maintains three history streams and refuses
to run fused when the checkpoint lacks fusion weights. See
[docs/deployment.md](docs/deployment.md) for the fused controller-side history
contract.

```bash
tam-eval-wrapper \
  --reference sine \
  --history-endpoint tcp://<controller-host>:5555 \
  --command-endpoint tcp://<controller-host>:5556 \
  --request-endpoint tcp://<controller-host>:5557
```

The low-level robot controller bridge is deployment-side infrastructure and is
not included in this repository.

## Source-To-OSC Sim Evaluation

This representative evaluation warms up on a source joint trajectory, switches
to operational-space control in simulation, and compares direct OSC against TAM.

```bash
tam-eval-source-to-osc-sim \
  --robot-preset panda \
  --tam-ckpt-path checkpoints/tam/tam_public_multirobot_ckpt3472000 \
  --conditions direct_osc tam_carried \
  --num-iterations 20 \
  --sim-backend batched
```

The evaluation auto-detects the checkpoint's history mode; the resolved value
is recorded as `resolved_history_torque_mode` in the per-iteration setup
metadata. `--sim-backend` defaults to `auto`. Fused (`base_tam_fusion`)
checkpoints require `--sim-backend legacy` because the batched backend supports
applied-history checkpoints only.

Outputs are written under `eval_logs/source_to_osc_tam_sim/<timestamp>/`:

- `summary.json`, `summary.csv`, and `summary.md`
- `summary_aggregate.json`, `summary_aggregate.csv`, and
  `summary_aggregate.md`
- per-iteration references and trajectory logs

Use `--conditions tam_all` to include both reset and carried TAM rows.

## Public Scripts

| TAM command | Script |
| --- | --- |
| `tam-generate-data` | `scripts/data/generate_dataset.py` |
| `tam-train-multi-robot` | `scripts/train/tam/train.py` |
| `tam-train-robot` | `scripts/train/tam/train.py` |
| `tam-dagger-finetune` | `scripts/train/tam/dagger_finetune.py` |
| `tam-mapping-server` | `scripts/deploy/mapping_server.py` |
| `tam-eval-wrapper` | `scripts/deploy/trajectory_tracking_eval.py` |
| `tam-eval-source-to-osc-sim` | `scripts/deploy/source_to_osc_tam_sim.py` |
