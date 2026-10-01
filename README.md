# DevTech Newton Examples

[![CPU tests and quality](https://github.com/NVIDIA/devtech-newton-examples/actions/workflows/quality.yml/badge.svg?branch=main&event=push)](https://github.com/NVIDIA/devtech-newton-examples/actions/workflows/quality.yml?query=branch%3Amain+event%3Apush)
[![GPU tests](https://github.com/NVIDIA/devtech-newton-examples/actions/workflows/tests.yml/badge.svg?branch=main&event=push)](https://github.com/NVIDIA/devtech-newton-examples/actions/workflows/tests.yml?query=branch%3Amain+event%3Apush)

Newton simulation examples from NVIDIA DevTech. This repository currently contains
the Franka pulley mechanical advantage example.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone
this repository and install the locked dependencies:

```bash
git clone https://github.com/NVIDIA/devtech-newton-examples.git
cd devtech-newton-examples
uv sync --locked --python 3.12
```

The project supports Python 3.10 or newer and requires `newton[examples]>=1.6.0`.
Use the included lockfile to install the tested dependency versions, including
Warp, MuJoCo, MuJoCo Warp, and the OpenGL viewer. The simulation uses Newton's
experimental coupled-solver APIs.
An NVIDIA GPU with a compatible driver is recommended for CUDA execution;
the OpenGL viewer also requires a working graphics display.

The Franka URDF and meshes are downloaded by `newton.utils.download_asset`
from Newton's public asset collection on first use, so the first run requires
internet access. The pulley, cable, and weight are generated in Python; no local
asset files or Git LFS setup are required.

## Run

List the examples and launch the default 4:1 mechanism with a 5 kg load:

```bash
uv run -m examples --list
uv run -m examples franka_pulley_mechanical_advantage --viewer gl
```

A MuJoCo-simulated Franka robot follows an inverse-kinematics trajectory,
grasps a box attached to the free end of a VBD cable, and pulls downward 0.4 m to
raise a guided load. Fixed and moving multi-sheave pulley blocks provide the
mechanical advantage. ADMM coupling transfers contact forces between the robot
and mechanism; the robot has no fixed attachment to the cable. The cable uses
Newton 1.6's `Rod` API with zero bend/twist stiffness, and the VBD solver uses
compliant ALM. Collision detection runs every substep.

Configure the supporting-strand count (an even integer of at least 2) and load:

```bash
uv run -m examples franka_pulley_mechanical_advantage --mechanical-advantage 6 --weight-mass 5
uv run -m examples franka_pulley_mechanical_advantage --help
```

The implementation stays in one file: `add_pulley_mechanism()` builds the
block and tackle, `gripper_target()` describes the approach/grasp/pull/hold
trajectory relative to the box, and `Example` connects the solvers and IK.
Control measurements update on every step, independently of rendering.

Use `--substeps` to change integration resolution (default: 16) and
`--vbd-iterations` to change VBD iterations per coupled substep (default: 10).
CUDA graph capture is always enabled on CUDA devices. Other solver tuning is
kept in `_build_solver()`.

`--weight-mass` is the intended net lifted mass. The load includes an additional
`0.2 kg * mechanical_advantage` to offset the cable-end box's gravitational
assistance. The viewer shows pulling force, measured force advantage, and load
lift relative to its initial height. Force advantage is the intended load's
weight divided by the measured pulling force; it is a diagnostic estimate
affected by friction, guide damping, and coupling.

## Validate

Run the full trajectory headlessly with the example's built-in checks:

```bash
uv run -m examples franka_pulley_mechanical_advantage --viewer null --test
```

The default run lasts 780 frames at 60 simulation frames per second. Validation
checks for finite body state and verifies that the final load lift is within 20%
of the commanded 0.4 m pull divided by the mechanical advantage (0.1 m for 4:1).
The final check measures displacement; the logged force advantage is diagnostic.
Shorter runs can initialize and exercise the simulation, but should omit `--test`
because its final displacement check requires the complete pull:

```bash
uv run -m examples franka_pulley_mechanical_advantage --viewer null --num-frames 10
```

## Tests and CI

Development dependencies are included in the lockfile:

```bash
uv sync --locked --dev
uv run --locked pytest                         # all tests; skips CUDA tests without a GPU
uv run --locked pytest -m "not cuda"           # CPU geometry and configuration checks
uv run --locked pytest -m cuda --require-cuda  # GPU simulation regressions
uv run --locked ruff check tests examples/multiphysics/example_franka_pulley_mechanical_advantage.py
uv run --locked ruff format --check tests examples/multiphysics/example_franka_pulley_mechanical_advantage.py
```

CUDA regression tests cover 2:1, 4:1, and 6:1 lifts, pull distance, grasp drift,
and final holding stability. They compare captured versus uncaptured frames
with odd and even substep counts. CPU tests cover the cable route, trajectory
phases, diagnostic plots, and invalid configuration.

The [CPU workflow](.github/workflows/quality.yml) runs CPU checks and lint on
pull requests and `main` pushes using Python 3.11 and 3.12, which have prebuilt
wheels for the locked GUI dependencies. The [GPU workflow](.github/workflows/tests.yml)
runs the full suite with `--require-cuda` on NVIDIA's
`linux-amd64-gpu-rtxpro6000-latest-1` runner. Both workflows support manual runs
and retain JUnit test results for 14 days, including results from failed tests.

GPU tests run on `main` and vetted `pull-request/<N>` branches created by NVIDIA's
copy-pr-bot. The configuration is in [.github/copy-pr-bot.yaml](.github/copy-pr-bot.yaml):
ready PRs sync automatically; draft PRs require manual `/ok to test`. This requires
the repository to have access to the NVIDIA runner pool and copy-pr-bot app, and
the configuration must be on `main` to take effect. GPU jobs are never triggered
directly by `pull_request` events. Missing CUDA fails the full-suite job rather
than silently skipping the simulations.

## Build

The project uses the `uv_build` backend and packages the `examples` module:

```bash
uv build
```

## License

The example code is licensed under [Apache 2.0](LICENSE.md). Dependencies and
downloaded robot assets retain their respective licenses.
