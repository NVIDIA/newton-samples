# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import Mock, call, patch

import newton.viewer
import numpy as np
import pytest
import warp as wp

from examples.multiphysics import example_franka_pulley_mechanical_advantage as pulley


@pytest.mark.parametrize("force", [0.0, pulley.DEFAULT_WEIGHT_MASS * pulley.GRAVITY / 4])
def test_diagnostic_plots(force):
    example = pulley.Example.__new__(pulley.Example)
    example.viewer = Mock()
    example._measure_robot_downward_force = Mock(return_value=force)
    example.weight_mass = pulley.DEFAULT_WEIGHT_MASS
    example.initial_load_z = 0.5
    example.load_z = 0.625
    example._log_diagnostics()
    assert example.viewer.log_scalar.call_args_list == [
        call("Robot downward force [N]", force, smoothing=10),
        call("Measured force advantage", 4.0 if force else 0.0, smoothing=10),
        call("Load lift [m]", 0.125, smoothing=10),
    ]


@pytest.mark.parametrize(
    ("time", "waypoint", "grip"),
    [
        (0.0, "start", pulley.GRIP_OPEN),
        (pulley.APPROACH_DURATION, "approach", pulley.GRIP_OPEN),
        (pulley.GRASP_START_TIME, "box", pulley.GRIP_OPEN),
        ((pulley.GRASP_START_TIME + pulley.GRASP_END_TIME) / 2, "box", (pulley.GRIP_OPEN + pulley.GRIP_CLOSE) / 2),
        (pulley.GRASP_END_TIME, "box", pulley.GRIP_CLOSE),
        ((pulley.GRASP_END_TIME + pulley.PULL_END_TIME) / 2, "half_pull", pulley.GRIP_CLOSE),
        (pulley.PULL_END_TIME, "pull", pulley.GRIP_CLOSE),
        (pulley.EXAMPLE_DURATION + 1.0, "pull", pulley.GRIP_CLOSE),
    ],
)
def test_gripper_trajectory(time, waypoint, grip):
    start = np.array([1.0, 0.0, 0.8])
    rotation = np.array([0.0, 0.0, 0.0, 1.0])
    box = np.array([0.7, 0.02, 0.45])
    waypoints = {
        "start": start,
        "approach": box + [0.14, 0.0, 0.14 * math.tan(pulley.GRIPPER_APPROACH_ANGLE)],
        "box": box,
        "half_pull": box - [0.0, 0.0, pulley.PULL_DISTANCE / 2],
        "pull": box - [0.0, 0.0, pulley.PULL_DISTANCE],
    }
    position, orientation, width = pulley.gripper_target(time, start, rotation, box)
    np.testing.assert_allclose(position, waypoints[waypoint])
    np.testing.assert_allclose(orientation, rotation if time == 0.0 else pulley.GRIPPER_APPROACH_ORIENTATION)
    assert width == pytest.approx(grip)


def test_gripper_approach_interpolation():
    start = np.array([1.0, 0.0, 0.8])
    # Equivalent quaternion signs must produce the same shortest-path blend.
    rotation = np.array([0.0, 0.0, 0.0, -1.0])
    box = np.array([0.7, 0.02, 0.45])
    position, orientation, width = pulley.gripper_target(pulley.APPROACH_DURATION / 2, start, rotation, box)
    approach = box + [0.14, 0.0, 0.14 * math.tan(pulley.GRIPPER_APPROACH_ANGLE)]
    np.testing.assert_allclose(position, (start + approach) / 2)
    np.testing.assert_allclose(orientation, (-rotation + pulley.GRIPPER_APPROACH_ORIENTATION) / 2)
    assert width == pulley.GRIP_OPEN


@pytest.mark.parametrize("advantage", [2, 4, 6])
def test_cable_route(advantage):
    count = advantage // 2
    ys = [(i - 0.5 * (count - 1)) * pulley.PULLEY_SHEAVE_SPACING for i in range(count)]
    end = wp.vec3(pulley.PULL_X, ys[-1], pulley.PULL_START_Z + pulley.END_WEIGHT_INITIAL_OFFSET)
    points = pulley.create_block_and_tackle_cable_points(
        [wp.vec3(pulley.PULLEY_BLOCK_X, y, pulley.PULLEY_MOVING_Z) for y in ys],
        [wp.vec3(pulley.PULLEY_BLOCK_X, y, pulley.PULLEY_FIXED_Z) for y in ys],
        wp.vec3(pulley.PULLEY_LEAD_X, ys[-1], pulley.PULLEY_FIXED_Z),
        end,
        pulley.CABLE_SEGMENT_LENGTH,
    )
    np.testing.assert_allclose(points[-1], end, atol=1.0e-7)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.all(lengths > 0.0)
    assert np.max(lengths) <= pulley.CABLE_SEGMENT_LENGTH + 1.0e-7
    # The free end must be vertical for the grasp box and initial approach.
    np.testing.assert_allclose(points[-2, :2], points[-1, :2], atol=1.0e-7)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--mechanical-advantage", "1", "even integer"),
        ("--mechanical-advantage", "3", "even integer"),
        ("--substeps", "0", "Substeps"),
        ("--vbd-iterations", "0", "VBD iterations"),
        ("--vbd-iterations", "-1", "VBD iterations"),
        ("--weight-mass", "0", "finite and positive"),
        ("--weight-mass", "nan", "finite and positive"),
        ("--weight-mass", "inf", "finite and positive"),
    ],
)
def test_invalid_configuration(option, value, message):
    args = pulley.Example.create_parser().parse_args([option, value])
    with pytest.raises(ValueError, match=message):
        pulley.Example(None, args)


@pytest.mark.cuda
@pytest.mark.parametrize("substeps", [15, 16])
def test_graph_matches_uncaptured_frames(cuda_device, substeps):
    args = pulley.Example.create_parser().parse_args(["--substeps", str(substeps), "--vbd-iterations", "3"])
    # Leave the reference solver uncaptured without adding a production toggle.
    with patch.object(pulley.Example, "simulate"):
        eager = pulley.Example(newton.viewer.ViewerNull(), args)
    captured = pulley.Example(newton.viewer.ViewerNull(), args)
    assert captured.graph is not None
    assert captured.solver.solver("vbd").iterations == 3
    initial_buffers = (captured.state_0, captured.state_1)
    for _ in range(8):
        # Run the reference frame directly; production CUDA steps always use the graph.
        eager.update_ik_target()
        eager.simulate()
        eager.sim_time += eager.frame_dt
        eager._read_measurements()
        eager.render()
        captured.step()
        assert (captured.state_0, captured.state_1) == initial_buffers
        # Contact ordering perturbs free cable rotations; compare physical
        # positions within 1 mm, well below the motion lost to stale buffers.
        np.testing.assert_allclose(
            captured.state_0.body_q.numpy()[:, :3], eager.state_0.body_q.numpy()[:, :3], atol=1.0e-3
        )
        captured.test_post_step()


@pytest.mark.cuda
@pytest.mark.parametrize("advantage", [2, 4, 6])
def test_mechanical_advantage(cuda_device, advantage):
    args = pulley.Example.create_parser().parse_args(["--mechanical-advantage", str(advantage)])
    example = pulley.Example(newton.viewer.ViewerNull(), args)
    assert example.graph is not None
    assert example.solver.solver("vbd").rigid_compliant_alm
    assert example.solver.solver("vbd").iterations == 10
    # FK must not rotate the anchored segment away from the authored route.
    cable_bodies = example.mechanism.cable_bodies
    np.testing.assert_allclose(
        example.state_0.body_q.numpy()[cable_bodies], example.model.body_q.numpy()[cable_bodies], atol=1.0e-6
    )
    max_grasp_error = 0.0
    hold_heights = []
    pull_origin = None
    for _ in range(math.ceil(pulley.EXAMPLE_DURATION * example.fps)):
        example.step()
        example.test_post_step()
        if example.pull_target_origin is not None:
            if pull_origin is None:
                pull_origin = example.pull_target_origin.copy()
                pull_correction = example.gripper_correction.copy()
            np.testing.assert_array_equal(example.pull_target_origin, pull_origin)
            np.testing.assert_array_equal(example.gripper_correction, pull_correction)
        if example.sim_time > pulley.GRASP_END_TIME:
            max_grasp_error = max(
                max_grasp_error, float(np.linalg.norm(example.box_target_center - example.gripper_pad_midpoint))
            )
        if example.sim_time > pulley.PULL_END_TIME + 1.0:
            hold_heights.append(example.load_z)
    example.test_final()
    lift = example.load_z - example.initial_load_z
    print(f"{advantage}:1 lift={lift:.6f} m, expected={pulley.PULL_DISTANCE / advantage:.6f} m")
    pulled = float(example.pull_target_origin[2] - example.box_target_center[2])
    hold_drift = max(hold_heights) - min(hold_heights)
    print(f"pull={pulled:.6f} m, grasp error={max_grasp_error:.6f} m, hold drift={hold_drift:.6f} m")
    assert pulled == pytest.approx(pulley.PULL_DISTANCE, rel=0.1)
    assert max_grasp_error < 0.02
    assert hold_drift < 0.005
