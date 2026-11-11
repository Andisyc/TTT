# Server training resource control

This runbook defines the operator-side resource envelope for foreground training on a single-GPU
Linux server. It complements the selected Hydra task profile; it does not replace the profile or
authorize a run.

## Why SSH can become unresponsive

MuJoCo environment collection is CPU-heavy. A large `algo.num_envs` can saturate all logical CPUs
and create memory pressure, while BLAS/OpenMP libraries may create additional worker threads. GPU
utilization can therefore coexist with an unresponsive SSH session.

The controls have different jobs:

- `taskset` reserves logical CPUs for SSH and system services;
- `nice` and `ionice` let interactive and system work pre-empt training;
- BLAS/OpenMP limits prevent nested thread oversubscription;
- `algo.num_envs` controls the dominant MuJoCo collection load.

Thread variables alone do not reserve CPU capacity. For a shared or modest server, use CPU
affinity as well.

## UniLab Planner-IDM resource plan

`uv run --frozen --no-sync --offline python scripts/plan_distill_resources.py` emits JSON configuration
only. It does not accept an Oracle checkpoint, load a policy, generate shell code, or start training.
The output separates three concerns:

- `config_name` selects the existing `oracle_fada` profile.
- `training.fada` contains the complete training overrides: parallel environment count, batch 4096,
  Replay 524288, 131072 windows per iteration, 31 iterations, and 2000 updates per IDM/Planner phase.
- `resources` contains recommendations for CPU affinity, RAM/VRAM budgets, priority, and thread limits.

Allowed CPUs are grouped by physical core. Two complete allowed cores are excluded from training
for SSH/system headroom; training uses all remaining allowed cores, with one environment per
training physical core and one compute thread per process. Memory planning includes readable
inherited cgroup v2 limits when available, without requiring controller delegation. It reserves
8 GiB of currently available RAM and 2048 MiB of free VRAM. Insufficient remaining capacity is
reported without reducing the learning budget. Environment count is a core-based starting point,
not a measured optimum or a per-environment memory-fit guarantee.

The generator only reads capacity and queries `nvidia-smi`. It does not create a systemd scope,
verify runtime limits, require sudo, or require CPU/cpuset delegation. Resource fields are proposed
budgets, not verified or enforced limits.

This is configuration planning, not a launcher. A subsequent command-generation step must apply
`resources` to the training process and its collector children, and merge `training` over the
selected base profile. Merely printing or saving JSON applies no limits to a later training process.
Resource values are a snapshot; regenerate before launch. These conservative budgets do not prove
that the full workload fits or that SSH stays responsive; other users' processes remain outside
this plan.

Only environment parallelism and execution resources change. Oracle, DR, mirror training, model,
optimizer settings, and learning budget remain untouched; collection may take longer.

## Preflight

Run these read-only checks before starting a new process:

```bash
cd /ssd1/cyx/ICE-Cal
pgrep -af 'scripts/train_offpolicy.py'
nvidia-smi
nproc
```

Do not start a second training process unless concurrent training was explicitly intended.

## Foreground privileged-Oracle command

The phase-neutral privileged FADA source uses `mujoco_fada_source`:
live privileged inputs, 30% standing commands, no gait-phase input, no phase
Reward or height cost. The current v024 Oracle instead selects
`mujoco_fada_phase_contact`. Targeted actuator faults are not part of either Oracle-training
profile. Left-knee gain attenuation is applied only when collecting downstream failed rollouts
after Oracle freeze; generic all-joint Kp/Kd and the remaining physical domain randomization stay
enabled during Oracle training.
Superseded command/simple/height experiment tasks are no longer active selectors.
Do not resume their checkpoints into this source lineage unless their checkpoint
metadata passes the canonical source contract.

This command intentionally does not use `nohup`. It reserves at least two, or roughly one eighth,
of the logical CPUs for SSH and starts from `1024` MuJoCo environments instead of the profile's
`2048`-environment maximum.

```bash
cd /ssd1/cyx/ICE-Cal

TOTAL_CPUS=$(nproc)
RESERVED_CPUS=$((TOTAL_CPUS / 8))
if [ "$RESERVED_CPUS" -lt 2 ]; then RESERVED_CPUS=2; fi
TRAIN_LAST_CPU=$((TOTAL_CPUS - RESERVED_CPUS - 1))
if [ "$TRAIN_LAST_CPU" -lt 0 ]; then
  echo "Not enough logical CPUs to reserve an SSH core set" >&2
  exit 1
fi

RUN_DIR=/ssd1/cyx/ICE-Cal/logs/FADAPrivilegedOracle_no_gait_seed1
mkdir -p "$RUN_DIR"

taskset -c "0-${TRAIN_LAST_CPU}" \
nice -n 5 \
ionice -c2 -n5 \
env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/ssd1/cyx/ICE-Cal/src \
  ICE_CAL_ORACLE_LINEAGE_ID=no-gait-baseline-seed1 \
uv run --frozen --no-sync python scripts/train_offpolicy.py \
  algo=sac \
  task=sac/g1_walk_flat/mujoco_fada_source \
  training.device=cuda:0 \
  training.no_play=true \
  training.log_dir="$RUN_DIR" \
  algo.num_envs=1024
```

The selected task profile still owns the algorithm, Reward, maximum iterations, save interval,
checkpoint lineage, and domain-randomization semantics. The command overrides only the operational
environment count and device/log identities.

For a genuinely clean comparison baseline, select
`task=sac/g1_walk_flat/mujoco_clean_baseline`, use a distinct `RUN_DIR`, and omit
`ICE_CAL_ORACLE_LINEAGE_ID`. That profile has no gait reward, privileged Actor input,
actuator-strength randomization, grouped physical randomization, or curriculum.

## Resource adjustment rule

Keep the CPU reservation and thread limits fixed. Adjust only `algo.num_envs` first:

| Condition | Action |
|---|---|
| SSH is still delayed or memory pressure is high | reduce to `algo.num_envs=512` |
| SSH is responsive and the GPU is materially underutilized | try `algo.num_envs=1536` |
| The server remains responsive at 1536 and the operator accepts the risk | use the profile default `2048` by removing the override |

Change one resource control at a time. A higher environment count is throughput tuning, not a
method change, but it changes the effective runtime configuration and should be recorded with the
run.

## Stop and cleanup

Because the process is foreground, use `Ctrl-C` once and wait for the collector child to exit. In
another SSH session, verify cleanup with:

```bash
pgrep -af 'scripts/train_offpolicy.py'
nvidia-smi
```

Do not use an unconditional recursive kill. If a child remains, identify its exact PID and parent
before terminating it.

## Evidence boundary

A responsive launch proves only that the selected process can start inside this resource envelope.
It does not prove convergence, policy quality, checkpoint superiority, Planner-IDM readiness, or
deployment safety.

## Oracle FADA native failure capture

`scripts/deploy/run_fada_failure_capture.py` arms both the training main process and
collectors for fatal signals only (`SIGSEGV`, `SIGBUS`, `SIGILL`, `SIGABRT`, `SIGFPE`).
It does not install Python exception-formatting or opcode breakpoints: ordinary
handled exceptions must not pause collection. `SIGTERM`, `SIGINT`, and `SIGHUP` pass
through for normal cancellation. Python exceptions retain their traceback/fallback
report but no longer trigger a pre-unwind core capture.

On the Linux x86-64/CPython 3.11 training server, `--self-test` runs a synthetic
collector cancellation followed by a main-process segmentation fault. It loads no
model or simulator. PASS requires clean debugger termination for the cancelled
collector and a complete main-process core with registers and native backtrace.
Formal training arguments and collection request timeouts are unchanged.
