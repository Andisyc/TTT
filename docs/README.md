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
Use separate invocations for unloaded and loaded protocols. For example:
The [loaded preset](../conf/distill/oat_sensitivity_loaded.yaml) disables base-mass,
body-mass, and center-of-mass randomization as required by deterministic runtime payloads.

```bash
uv run --frozen --no-sync python scripts/run_oat_sensitivity.py \
  --config /path/to/loaded.yaml --output-dir logs/oat_sensitivity/gamma_loaded \
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

