# ICE-Cal engineering documentation

This directory contains stable engineering explanations and repeatable operator procedures.
It does not define research semantics and does not authorize training or deployment.

## Documentation boundary

| Question | Owner |
|---|---|
| What is ICE-Cal and what should a developer read first? | [`README.md`](../README.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| What must an Agent preserve? | [`AGENTS.md`](../AGENTS.md) |
| How should a repeatable engineering operation be performed? | [`runbooks/`](runbooks/) |
| How is the implementation organized? | [`engineering/`](engineering/) |
| What is the current research method or training Contract? | [`note/`](../note/) |
| What happened in one dated implementation or audit unit? | `note/fada/plans`, `note/fada/evidence`, and `note/fada/reviews` |

## Stable engineering documents

- [Server training resource control](runbooks/server-training.md)
- [Engineering document scope](engineering/README.md)

## Offline OAT recording evaluation

Use `uv run --frozen --no-sync python scripts/evaluate_oat_experiment.py` with
explicit `run_dir`, a new `output_dir`, `linear_frame`, `sample_phase`,
`relative_to`, `start_s`, and `end_s`. Available choices are documented in
[`experiment_evaluate.yaml`](../conf/distill/experiment_evaluate.yaml).
The entrypoint reads committed recording chunks without simulation or policy loading
and writes `summary.json` and `errors.csv`. It reports episodes separately and retains
terminal samples. Partial-window errors are observed-segment diagnostics, not full-trial scores.
Normalization requires an explicitly selected compatible reference run; attitude and
action diagnostics require their conventions to be supplied. Raw done flags do not
define falls or survival. No position/path-success metrics are inferred from these recordings.

## Headless OAT component trials

`scripts/run_oat_experiment.py` runs one explicitly selected group and seed without
a viewer. Video rendering is disabled unless `video.enabled=true`. Inspect the required protocol fields without launching:

```bash
uv run --frozen --no-sync python scripts/run_oat_experiment.py --cfg job
```

[`oat_experiment.yaml`](../conf/distill/oat_experiment.yaml) requires the caller's
`task_owner`, checkpoint, command, duration, seed, payload, shared optimizer settings,
and explicit environment overrides. It does not inherit the `straight_payload` Demo.
Groups are `fada_ttt`, `online_reference`, `action_snapshot`, `no_anchor`, and `full_oat`;
only FADA-TTT and No anchor force the anchor coefficient to zero.
An explicit empty `env_overrides: {}` accepts the selected task's environment defaults;
it does not certify them as an approved formal protocol.
Each new `output_dir` contains `raw/` records and, when `evaluation.enabled=true`
with explicit measurement conventions, `evaluation/` results.
The loop stops on its first termination/truncation without autoreset. Duration is
rounded up to complete control steps and both requested and actual times are recorded.
Ctrl-C, SIGTERM and errors use the recorder's existing flush/close path.
No evaluation is silently inferred when collection alone is requested.

## OAT sensitivity batches

Fill a copy of `conf/distill/oat_experiment.yaml` with the shared protocol,
including the checkpoint, payload condition, optimizer and evaluation settings.
The batch runner sets `group=full_oat`, the selected online parameter, `seed`,
and `output_dir`; it enables evaluation and offscreen recording automatically.
Use separate invocations for the [unloaded](../conf/distill/oat_sensitivity_unloaded.yaml)
and [loaded](../conf/distill/oat_sensitivity_loaded.yaml) presets. Both disable
environment domain randomization, observation noise, external pushes, and random
reset pose and velocity. They preserve the checkpoint's gait and control settings;
use the same seeds across conditions. Only the loaded preset adds the runtime payload.
For example:

```bash
uv run --frozen --no-sync python scripts/run_oat_sensitivity.py \
  --config conf/distill/oat_sensitivity_loaded.yaml --output-dir logs/oat_sensitivity/gamma_loaded \
  --param gamma --values 0.05 0.1 --repeats 10
```

`--values` accepts one or more explicit values, and `--repeats` defaults to 10.
Add `--dry-run` to list the planned trial directories. Each trial has a fresh
session and writes `trial.mp4`, `trial.json`, `raw/`, and `evaluation/` when the
evaluation window is available. The batch `summary.json` is refreshed after each
trial and reports the mean, sample standard deviation, minimum, maximum and
sample count per metric and value. Errors from incomplete windows are kept in
the trial records but excluded from full-window means. Termination reasons are
reported separately; they are not interpreted as falls. For normalized errors,
the optional reference must match the trial seed (use `${seed}` in the reference
path if needed) and the evaluator's existing compatibility checks.

Timing overrides are optional: `--duration-s 130 --attach-time-s 10` records
130 seconds total, attaches the payload at 10 seconds, and evaluates the following
120 seconds. `--attach-time-s` is for loaded protocols and also moves the online
warm-up boundary to after attachment, using the protocol's explicit
`timing_ctrl_dt_s`. The trial checks that this period matches the checkpoint's
environment before stepping. `--duration-s` extends the evaluation endpoint and
episode timeout together; unloaded evaluation retains its configured start time.
Without either option, the YAML timing is unchanged. Dry-run output shows the
resolved duration, attachment, warm-up and evaluation interval.

The staged-load protocol `conf/distill/oat_sensitivity_staged_loaded.yaml` keeps
the same OAT settings, attaches 6 kg at episode time 10 s, then adds 2 kg at
20 s (8 kg cumulative). It runs for 70 s and evaluates the 60 s after the first
attachment. `payload.additional_mass_events` contains `[absolute_time_s,
incremental_kg]` pairs at the same attachment position and box dimensions;
recorded `payload_applied.added_mass_kg` is the cumulative added mass.
Reset restores the original body mass, COM and inertia and rearms every event.
Compare `--param sigma --values 0 1e-4` without changing the original loaded preset.

## Table 3 component ablation batches

`conf/distill/oat_component_ablation.yaml` defines four methods, 2/4/6/8 kg
left-shoulder payloads, and seeds 0--9: 160 trials. Each trial runs in a fresh
Python process, walks unloaded without updates for 10 s, attaches its payload,
then adapts for 20 s. OAT variants retain T=10, gamma=0.05 and sigma=1e-4;
FADA-TTT uses its inverse-dynamics objective with sigma=0.

```bash
uv run --frozen --no-sync python scripts/run_oat_ablation.py \
  --output-dir logs/oat_ablation/table3_v1
```

Add `--dry-run` to inspect the plan without creating outputs or starting a
simulator. Add `--resume` with the same configuration to skip verified finalized
trials and run missing ones. Changed protocols, partial trial directories and
abandoned worker logs are rejected rather than overwritten or automatically
retried. Use a new batch directory if a failed attempt left partial artifacts.
`--methods`, `--payloads-kg`, `--repeats` and `--first-seed` can select a smaller
new batch, including the exact missing method/mass/seed combination.

Results live at `<output>/<method>/payload=<kg>/seed_<seed>/` with a video,
`trial.json`, raw measurements and evaluation. Per-trial logs are in `logs/`;
`summary.json` and `table3.csv` refresh after each trial. Fixed-window error
means exclude incomplete windows but keep their counts and termination records.
Survival uses the environment's explicit fall provenance; a completed time limit
is not a fall, and other/unknown early terminations remain unclassified.

Normalized velocity error requires an explicit common reference. Supply
`--reference-run-dir '/path/to/reference/payload={payload_kg}/seed_{seed}/raw'`
to use the existing evaluator's strict compatibility checks. Without it,
normalization is marked `not_configured`, normalized CSV fields remain blank,
and raw linear/angular velocity errors plus Roll RMSE are still collected.

## Isolated MuJoCo native lifecycle repair

`patches/mujoco-uni-3.8.0-lifecycle.patch` targets the exact source archive locked
in `uv.lock`. It rejects overlapping operations on one BatchEnvPool, retires
submitted tasks before unwinding, and propagates worker errors after sibling
tasks finish. MuJoCo fatal-error jump contexts are installed on the worker
that calls MuJoCo, rather than on the dispatching thread.

Build and check an opt-in overlay with the existing MuJoCo 3.8.0 environment,
CMake, a C++17 compiler, and `patch`. Linux additionally requires the wheel's
Clang 20/libc++ toolchain (`CXX` and `--libcxx-root`); the builder refuses a
GCC/libstdc++ fallback before downloading or building:

```bash
uv run --frozen --no-sync python scripts/build_mujoco_lifecycle_overlay.py \
  --work-dir /tmp/unilab-mujoco-lifecycle
```

The builder verifies the locked source SHA256 and rebuilds the Python binding
family with a compatible C++ ABI. It checks native lifecycle contracts, XML
sensor export, and single-environment Oracle initialization/reset/close before
printing `VERIFIED_OVERLAY`. The installed extension is unchanged. Only processes
explicitly given the verified directory through `PYTHONPATH` use the patched
bindings; removing that override returns to the installed package. No training
or environment stepping is started. The earlier GCC `overlay-coherent` artifact
fails XML export and must not be used for training.
For offline builds, supply `--source-archive`, `--abseil-source`, `--eigen-source`,
and `--pybind11-source` using the fixed source versions in the builder.
Passing these checks verifies the repaired boundary, not elimination of an
intermittent training crash. The sibling standalone `mujoco-uni-runtime` package
is not substituted for the locked production package.

## Process material

`superpowers/` contains dated implementation plans and design working records produced by the
Superpowers workflow. It is not the current research authority or the canonical operator manual.
When one of those records conflicts with `AGENTS.md`, an active Contract, or a current runbook,
the current authority wins.

