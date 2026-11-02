# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Franka Pulley Mechanical Advantage
#
# A MuJoCo-simulated Franka follows an IK trajectory that pulls the free end
# of a VBD cable. Upper and lower multi-sheave crane blocks lift a guided
# weight with a configurable mechanical advantage. The hand approaches and
# closes on a thin box attached directly to the cable end. SolverCoupledADMM
# transfers contact force between the MuJoCo gripper and the VBD mechanism
# without a fixed robot-cable attachment.
#
# The freely bending cable is built directly on a pre-wrapped route.
# Validation checks that its load motion matches the mechanical advantage.
#
# Command: uv run -m examples franka_pulley_mechanical_advantage
#
###########################################################################

from __future__ import annotations

import math
from dataclasses import dataclass

import newton
import newton.examples
import newton.utils
import numpy as np
import warp as wp
from newton import ik
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM

import examples

DEFAULT_MECHANICAL_ADVANTAGE = 4
DEFAULT_WEIGHT_MASS = 5.0
END_WEIGHT_MASS = 0.2
GRAVITY = 9.81
FRANKA_BASE_X = 1.3

CABLE_RADIUS = 0.004
CABLE_SEGMENT_LENGTH = 0.020
CABLE_PULLEY_FRICTION = 0.1
CABLE_CONTACT_GAP = 0.5 * CABLE_RADIUS
END_WEIGHT_INITIAL_OFFSET = 0.020
END_BOX_HALF_X = 0.025
END_BOX_HALF_Y = 0.025
END_BOX_HALF_Z = 0.030
GRIPPER_FRICTION = 8.0
GRIPPER_CONTACT_STIFFNESS = 2.0e4
GRIPPER_CONTACT_DAMPING = 50.0
GRIPPER_CORRECTION_FILTER = 0.2
GRIPPER_LATERAL_CORRECTION_MAX = 0.03
GRIPPER_HEIGHT_CORRECTION_MAX = 0.06

PULLEY_RADIUS = 0.045
PULLEY_WRAP_CLEARANCE = 1.25 * CABLE_RADIUS
PULLEY_WRAP_RADIUS = PULLEY_RADIUS + PULLEY_WRAP_CLEARANCE
PULLEY_FIXED_Z = 1.10
PULLEY_MOVING_Z = 0.72
PULLEY_BLOCK_X = 0.50
PULLEY_LEAD_X = PULLEY_BLOCK_X + 3.0 * PULLEY_WRAP_RADIUS
PULL_X = PULLEY_LEAD_X + PULLEY_WRAP_RADIUS
PULLEY_SHEAVE_SPACING = 5.5 * CABLE_RADIUS

PULL_START_Z = 0.48
PULL_DISTANCE = 0.4
GRIPPER_APPROACH_ANGLE = math.pi / 6.0
APPROACH_DURATION = 1.2
SIDE_APPROACH_DURATION = 0.8
GRASP_DURATION = 1.0
PULL_DURATION = 8.0
FINAL_HOLD_DURATION = 2.0
GRASP_START_TIME = APPROACH_DURATION + SIDE_APPROACH_DURATION
GRASP_END_TIME = GRASP_START_TIME + GRASP_DURATION
PULL_END_TIME = GRASP_END_TIME + PULL_DURATION
EXAMPLE_DURATION = PULL_END_TIME + FINAL_HOLD_DURATION

GRIPPER_APPROACH_ORIENTATION = (
    0.0,
    -math.cos(GRIPPER_APPROACH_ANGLE),
    0.0,
    math.sin(GRIPPER_APPROACH_ANGLE),
)
GRIP_OPEN = 0.040
GRIP_CLOSE = 0.007

# Raised-arm starting point for the IK solve (7 arm + 2 finger coordinates).
FRANKA_Q = [
    0.0,
    -0.569,
    0.0,
    -2.810,
    0.0,
    3.037,
    0.741,
    GRIP_OPEN,
    GRIP_OPEN,
]


def gripper_target(
    time: float, start_position: np.ndarray, start_rotation: np.ndarray, box_center: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Approach the box, close the fingers, pull down, then hold the final pose."""
    approach = box_center + np.array([0.14, 0.0, 0.14 * math.tan(GRIPPER_APPROACH_ANGLE)])
    pull_end = box_center - np.array([0.0, 0.0, PULL_DISTANCE])
    times = [0.0, APPROACH_DURATION, GRASP_START_TIME, GRASP_END_TIME, PULL_END_TIME]
    positions = np.array([start_position, approach, box_center, box_center, pull_end])
    position = np.array([np.interp(time, times, positions[:, axis]) for axis in range(3)])
    approach_rotation = np.asarray(GRIPPER_APPROACH_ORIENTATION)
    if np.dot(start_rotation, approach_rotation) < 0.0:
        start_rotation = -start_rotation
    alpha = np.clip(time / APPROACH_DURATION, 0.0, 1.0)
    rotation = (1.0 - alpha) * start_rotation + alpha * approach_rotation
    grip = float(np.interp(time, [GRASP_START_TIME, GRASP_END_TIME], [GRIP_OPEN, GRIP_CLOSE]))
    return position, rotation, grip


@wp.kernel
def set_task_target(
    target_positions: wp.array[wp.vec3],
    target_rotations: wp.array[wp.vec4],
    finger_position: wp.array[float],
    position: wp.vec3,
    rotation: wp.vec4,
    grip_width: float,
):
    """Set the single-world IK target without reallocating device arrays."""
    target_positions[0] = position
    target_rotations[0] = rotation
    finger_position[0] = grip_width


@wp.kernel
def set_gripper_target(joint_q: wp.array2d[float], finger_position: wp.array[float], index_0: int, index_1: int):
    """Set both Franka finger coordinates to the commanded half-width."""
    joint_q[0, index_0] = finger_position[0]
    joint_q[0, index_1] = finger_position[0]


def _find_label_index(labels: list[str], suffix: str) -> int:
    for index, label in enumerate(labels):
        if label.endswith(suffix):
            return index
    raise ValueError(f"Could not find label ending in {suffix!r}")


def _append_route_point(points: list[wp.vec3], point: wp.vec3) -> None:
    if not points or float(wp.length(point - points[-1])) > 1.0e-8:
        points.append(point)


def _append_arc_xz(
    points: list[wp.vec3],
    center: wp.vec3,
    radius: float,
    start_angle: float,
    end_angle: float,
    segment_length: float,
) -> None:
    """Append an arc in the vertical XZ plane, following the signed angle interval."""
    delta = end_angle - start_angle
    count = max(3, math.ceil(abs(delta) * radius / segment_length))
    for i in range(count + 1):
        angle = start_angle + delta * float(i) / float(count)
        _append_route_point(
            points,
            wp.vec3(
                float(center[0]) + radius * math.cos(angle),
                float(center[1]),
                float(center[2]) + radius * math.sin(angle),
            ),
        )


def _resample_route(route_points: list[wp.vec3], segment_length: float) -> np.ndarray:
    """Sample uniformly in polyline arclength; chords at bends are slightly shorter."""
    points = np.asarray(route_points, dtype=np.float64)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    keep = np.concatenate(([True], np.diff(distances) > 1.0e-8))
    points, distances = points[keep], distances[keep]
    segment_count = max(2, math.ceil(distances[-1] / segment_length))
    samples = np.linspace(0.0, distances[-1], segment_count + 1)
    return np.column_stack([np.interp(samples, distances, points[:, axis]) for axis in range(3)])


def create_block_and_tackle_cable_points(
    moving_centers: list[wp.vec3],
    fixed_centers: list[wp.vec3],
    lead_center: wp.vec3,
    pull_end: wp.vec3,
    segment_length: float,
) -> np.ndarray:
    """Create the pre-wrapped route through upper and lower crane blocks."""
    # Keep the dead end below the upper block so it does not touch an unused sheave.
    start = wp.vec3(
        float(moving_centers[0][0]) + PULLEY_WRAP_RADIUS,
        float(moving_centers[0][1]),
        float(fixed_centers[0][2]) - 2.0 * PULLEY_WRAP_RADIUS,
    )
    points = [start]

    for index, (moving_center, fixed_center) in enumerate(zip(moving_centers, fixed_centers, strict=True)):
        _append_arc_xz(
            points,
            moving_center,
            PULLEY_WRAP_RADIUS,
            0.0,
            -math.pi,
            segment_length,
        )
        _append_arc_xz(
            points,
            fixed_center,
            PULLEY_WRAP_RADIUS,
            math.pi,
            0.0 if index < len(fixed_centers) - 1 else 0.5 * math.pi,
            segment_length,
        )

    _append_arc_xz(
        points,
        lead_center,
        PULLEY_WRAP_RADIUS,
        0.5 * math.pi,
        0.0,
        segment_length,
    )
    _append_route_point(points, pull_end)
    return _resample_route(points, segment_length)


def _filter_body_group_collisions(builder: newton.ModelBuilder, bodies: list[int]) -> None:
    """Disable collisions between distinct bodies in a group."""
    for i, body_a in enumerate(bodies):
        for body_b in bodies[i + 1 :]:
            for shape_a in builder.body_shapes.get(body_a, []):
                for shape_b in builder.body_shapes.get(body_b, []):
                    builder.add_shape_collision_filter_pair(int(shape_a), int(shape_b))


def _add_box(
    builder: newton.ModelBuilder,
    *,
    body: int,
    center: wp.vec3,
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float],
    label: str,
    density: float = 0.0,
    collision: bool = False,
) -> int:
    cfg = newton.ModelBuilder.ShapeConfig(
        density=density,
        ke=1.0e5,
        kd=20.0,
        mu=0.8,
        has_shape_collision=collision,
        has_particle_collision=collision,
    )
    return builder.add_shape_box(
        body=body,
        xform=wp.transform(center, wp.quat_identity()),
        hx=half_extents[0],
        hy=half_extents[1],
        hz=half_extents[2],
        cfg=cfg,
        color=color,
        label=label,
    )


def _add_pulley_stack_mount(
    builder: newton.ModelBuilder,
    *,
    body: int,
    axle_center: wp.vec3,
    mount_z: float,
    stack_half_y: float,
    support_color: tuple[float, float, float],
    label: str,
) -> None:
    """Add a shared axle and two bearing supports to a pulley stack."""
    axle_radius = 2.0 * CABLE_RADIUS
    bearing_half_y = 2.0 * CABLE_RADIUS
    bearing_center_y = stack_half_y + bearing_half_y + 0.002
    axle_half_height = stack_half_y + 2.0 * bearing_half_y
    axle_x = float(axle_center[0])
    axle_y = float(axle_center[1])
    axle_z = float(axle_center[2])
    align_cylinder_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
    axle_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
    )
    builder.add_shape_cylinder(
        body=body,
        xform=wp.transform(axle_center, align_cylinder_to_y),
        radius=axle_radius,
        half_height=axle_half_height + 0.003,
        cfg=axle_cfg,
        color=(0.42, 0.45, 0.48),
        label=f"{label}_axle",
    )

    support_center_z = 0.5 * (axle_z + mount_z)
    support_half_z = 0.5 * abs(mount_z - axle_z) + axle_radius
    for side, suffix in ((-1.0, "neg"), (1.0, "pos")):
        _add_box(
            builder,
            body=body,
            center=wp.vec3(axle_x, axle_y + side * bearing_center_y, support_center_z),
            half_extents=(0.025, bearing_half_y, support_half_z),
            color=support_color,
            label=f"{label}_bearing_{suffix}",
        )


def add_pulley(
    builder: newton.ModelBuilder,
    *,
    center: wp.vec3,
    parent: int,
    parent_origin: wp.vec3,
    color: tuple[float, float, float],
    label: str,
    density: float,
) -> tuple[int, int]:
    """Add a passive grooved pulley adapted from cable_cross_slide_table."""
    body = builder.add_link(
        xform=wp.transform(center, wp.quat_identity()),
        label=f"{label}_body",
    )
    joint = builder.add_joint_revolute(
        parent=parent,
        child=body,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(center - parent_origin, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        armature=1.0e-4,
        friction=0.0,
        label=f"{label}_axle",
    )

    # Match the cross-slide sample's close-fitting groove so the cable cannot
    # shuttle along the pulley axle before contacting a flange.
    groove_half_width = 1.55 * CABLE_RADIUS
    flange_half_thickness = 0.6 * CABLE_RADIUS
    flange_radius = PULLEY_RADIUS + 3.2 * CABLE_RADIUS
    align_cylinder_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
    sheave_cfg = newton.ModelBuilder.ShapeConfig(
        density=density,
        ke=1.0e5,
        kd=0.0,
        mu=CABLE_PULLEY_FRICTION,
    )
    flange_cfg = newton.ModelBuilder.ShapeConfig(
        density=density,
        ke=1.0e5,
        kd=0.0,
        mu=0.5 * CABLE_PULLEY_FRICTION,
    )

    for suffix, y, radius, half_height, cfg, shade in (
        ("sheave", 0.0, PULLEY_RADIUS, groove_half_width, sheave_cfg, color),
        (
            "flange_neg",
            -(groove_half_width + flange_half_thickness),
            flange_radius,
            flange_half_thickness,
            flange_cfg,
            tuple(0.68 * component for component in color),
        ),
        (
            "flange_pos",
            groove_half_width + flange_half_thickness,
            flange_radius,
            flange_half_thickness,
            flange_cfg,
            tuple(0.68 * component for component in color),
        ),
    ):
        builder.add_shape_cylinder(
            body=body,
            xform=wp.transform(wp.vec3(0.0, y, 0.0), align_cylinder_to_y),
            radius=radius,
            half_height=half_height,
            cfg=cfg,
            color=shade,
            label=f"{label}_{suffix}",
        )

    return body, joint


@dataclass
class PulleyMechanism:
    """Solver ownership and attachment geometry for the block and tackle."""

    bodies: range
    joints: list[int]
    shapes: range
    weight_body: int
    cable_bodies: list[int]
    end_box_center_offset: float
    pull_line_xy: np.ndarray


def _add_cable(builder: newton.ModelBuilder, cable_points: np.ndarray) -> tuple[list[int], list[int], float]:
    """Add the anchored rod and grasp box; return bodies, joints, and box offset."""
    cable = newton.Rod(cable_points, radius=CABLE_RADIUS)
    cable_cfg = newton.ModelBuilder.ShapeConfig(
        density=10.0,
        ke=1.0e5,
        kd=0.0,
        mu=CABLE_PULLEY_FRICTION,
        gap=CABLE_CONTACT_GAP,
    )
    cable_bodies, cable_joints = builder.add_rod(
        rod=cable,
        body_frame_origin="com",
        cfg=cable_cfg,
        # A freely bending cable needs no separate straight rest geometry.
        stretch_stiffness=1.0e5,
        bend_stiffness=0.0,
        twist_stiffness=0.0,
        wrap_in_articulation=False,  # articulation is created manually later with anchor ball joint
        color=(0.82, 0.72, 0.46),
        label="block_and_tackle_cable",
    )
    _filter_body_group_collisions(builder, cable_bodies)

    segment_lengths = cable.segment_lengths
    end_box_center_offset = 0.5 * float(segment_lengths[-1]) + END_BOX_HALF_Z
    first_endpoint_xform = wp.transform(wp.vec3(0.0, 0.0, -0.5 * float(segment_lengths[0])), wp.quat_identity())
    anchor_joint = builder.add_joint_ball(
        parent=-1,
        child=cable_bodies[0],
        # At joint_q=identity, FK must preserve the first segment's frame.
        parent_xform=wp.transform(cable_points[0], wp.quat(*cable.quaternions[0])),
        child_xform=first_endpoint_xform,
        label="fixed_cable_anchor",
    )

    end_weight_volume = 8.0 * END_BOX_HALF_X * END_BOX_HALF_Y * END_BOX_HALF_Z
    end_weight_cfg = newton.ModelBuilder.ShapeConfig(
        density=END_WEIGHT_MASS / end_weight_volume,
        ke=GRIPPER_CONTACT_STIFFNESS,
        kd=GRIPPER_CONTACT_DAMPING,
        mu=GRIPPER_FRICTION,
        gap=0.002,
    )
    builder.add_shape_box(
        body=cable_bodies[-1],
        xform=wp.transform(
            wp.vec3(0.0, 0.0, end_box_center_offset),
            wp.quat_identity(),
        ),
        hx=END_BOX_HALF_X,
        hy=END_BOX_HALF_Y,
        hz=END_BOX_HALF_Z,
        cfg=end_weight_cfg,
        color=(0.96, 0.78, 0.26),
        label="graspable_end_weight",
    )
    builder.add_articulation(
        [*cable_joints, anchor_joint],
        label="block_and_tackle_cable",
    )

    anchor_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
    )
    builder.add_shape_sphere(
        body=-1,
        xform=wp.transform(cable_points[0], wp.quat_identity()),
        radius=2.0 * CABLE_RADIUS,
        cfg=anchor_cfg,
        color=(0.86, 0.82, 0.70),
        label="fixed_cable_anchor_marker",
    )
    return cable_bodies, [*cable_joints, anchor_joint], end_box_center_offset


def add_pulley_mechanism(
    builder: newton.ModelBuilder, mechanical_advantage: int, weight_mass: float
) -> PulleyMechanism:
    """Build the guided load, passive sheaves, and cable owned by VBD."""
    vbd_body_start = builder.body_count
    vbd_shape_start = builder.shape_count
    vbd_joints: list[int] = []

    pulley_pair_count = mechanical_advantage // 2
    pulley_sheave_y = tuple(
        (index - 0.5 * (pulley_pair_count - 1)) * PULLEY_SHEAVE_SPACING for index in range(pulley_pair_count)
    )
    pulley_block_half_y = 0.5 * pulley_pair_count * PULLEY_SHEAVE_SPACING
    pull_y = pulley_sheave_y[-1]
    pull_line_xy = np.array([PULL_X, pull_y], dtype=np.float64)

    frame_color = (0.16, 0.22, 0.30)
    top_beam_center_z = PULLEY_FIXED_Z + 0.09
    top_beam_half_z = 0.025
    _add_box(
        builder,
        body=-1,
        center=wp.vec3(0.5 * (PULLEY_BLOCK_X + PULL_X), 0.0, top_beam_center_z),
        half_extents=(
            0.5 * (PULL_X - PULLEY_BLOCK_X) + 0.08,
            pulley_block_half_y + 0.03,
            top_beam_half_z,
        ),
        color=frame_color,
        label="pulley_frame_top_beam",
    )

    load_center = wp.vec3(PULLEY_BLOCK_X, 0.0, 0.49)
    load_half_x = 0.085
    weight_half_y = max(0.06, pulley_block_half_y + 0.02)
    _add_box(
        builder,
        body=-1,
        center=wp.vec3(float(load_center[0]), 0.0, 0.39),
        half_extents=(0.14, weight_half_y + 0.06, 0.01),
        color=(0.24, 0.27, 0.30),
        label="weight_floor",
        collision=True,
    )
    weight_body = builder.add_link(
        xform=wp.transform(load_center, wp.quat_identity()),
        label="load_weight_body",
    )
    weight_half_extents = (load_half_x, weight_half_y, 0.09)
    weight_volume = 8.0 * math.prod(weight_half_extents)
    _add_box(
        builder,
        body=weight_body,
        center=wp.vec3(0.0),
        half_extents=weight_half_extents,
        density=(weight_mass + END_WEIGHT_MASS * mechanical_advantage) / weight_volume,
        collision=True,
        color=(0.72, 0.16, 0.12),
        label="load_weight",
    )
    load_slide = builder.add_joint_prismatic(
        parent=-1,
        child=weight_body,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(load_center, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        target_kd=20.0,
        label="weight_vertical_guide",
    )
    vbd_joints.append(load_slide)

    moving_color = (0.88, 0.54, 0.12)
    fixed_color = (0.12, 0.38, 0.78)
    moving_centers = [wp.vec3(PULLEY_BLOCK_X, y, PULLEY_MOVING_Z) for y in pulley_sheave_y]
    fixed_centers = [wp.vec3(PULLEY_BLOCK_X, y, PULLEY_FIXED_Z) for y in pulley_sheave_y]
    lead_center = wp.vec3(PULLEY_LEAD_X, pull_y, PULLEY_FIXED_Z)

    _add_pulley_stack_mount(
        builder,
        body=-1,
        axle_center=wp.vec3(PULLEY_BLOCK_X, 0.0, PULLEY_FIXED_Z),
        mount_z=top_beam_center_z - top_beam_half_z,
        stack_half_y=pulley_block_half_y,
        support_color=frame_color,
        label="fixed_pulley_stack",
    )
    _add_pulley_stack_mount(
        builder,
        body=-1,
        axle_center=lead_center,
        mount_z=top_beam_center_z - top_beam_half_z,
        stack_half_y=0.5 * PULLEY_SHEAVE_SPACING,
        support_color=frame_color,
        label="lead_pulley_mount",
    )

    _add_pulley_stack_mount(
        builder,
        body=weight_body,
        axle_center=wp.vec3(0.0, 0.0, PULLEY_MOVING_Z - float(load_center[2])),
        mount_z=weight_half_extents[2],
        stack_half_y=pulley_block_half_y,
        support_color=(0.50, 0.18, 0.12),
        label="moving_pulley_stack",
    )

    moving_bodies = []
    moving_joints = []
    for index, center in enumerate(moving_centers):
        body, joint = add_pulley(
            builder,
            center=center,
            parent=weight_body,
            parent_origin=load_center,
            color=moving_color,
            label=f"moving_pulley_{index}",
            density=1.0e-1,
        )
        moving_bodies.append(body)
        moving_joints.append(joint)
        vbd_joints.append(joint)

    fixed_bodies = []
    fixed_joints = []
    for index, center in enumerate(fixed_centers):
        body, joint = add_pulley(
            builder,
            center=center,
            parent=-1,
            parent_origin=wp.vec3(0.0),
            color=fixed_color,
            label=f"fixed_pulley_{index}",
            density=1.0e-1,
        )
        fixed_bodies.append(body)
        fixed_joints.append(joint)
        vbd_joints.append(joint)

    lead_body, lead_joint = add_pulley(
        builder,
        center=lead_center,
        parent=-1,
        parent_origin=wp.vec3(0.0),
        color=fixed_color,
        label="lead_pulley",
        density=1.0e-1,
    )
    fixed_bodies.append(lead_body)
    vbd_joints.append(lead_joint)

    _filter_body_group_collisions(builder, moving_bodies)
    _filter_body_group_collisions(builder, fixed_bodies)
    builder.add_articulation([load_slide, *moving_joints], label="moving_pulley_block")
    for index, joint in enumerate(fixed_joints):
        builder.add_articulation([joint], label=f"fixed_pulley_{index}_articulation")
    builder.add_articulation([lead_joint], label="lead_pulley_articulation")

    pull_end = wp.vec3(PULL_X, pull_y, PULL_START_Z + END_WEIGHT_INITIAL_OFFSET)
    cable_points = create_block_and_tackle_cable_points(
        moving_centers,
        fixed_centers,
        lead_center,
        pull_end,
        CABLE_SEGMENT_LENGTH,
    )
    cable_bodies, cable_joints, end_box_center_offset = _add_cable(builder, cable_points)
    vbd_joints.extend(cable_joints)

    return PulleyMechanism(
        bodies=range(vbd_body_start, builder.body_count),
        joints=vbd_joints,
        shapes=range(vbd_shape_start, builder.shape_count),
        weight_body=weight_body,
        cable_bodies=cable_bodies,
        end_box_center_offset=end_box_center_offset,
        pull_line_xy=pull_line_xy,
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(args.substeps)
        if self.sim_substeps < 1:
            raise ValueError("Substeps must be positive")
        self.sim_dt = self.frame_dt / self.sim_substeps
        vbd_iterations = int(args.vbd_iterations)
        if vbd_iterations < 1:
            raise ValueError("VBD iterations must be positive")
        self.mechanical_advantage = int(args.mechanical_advantage)
        if self.mechanical_advantage < 2 or self.mechanical_advantage % 2 != 0:
            raise ValueError("Mechanical advantage must be an even integer of at least 2")
        self.weight_mass = float(args.weight_mass)
        if not math.isfinite(self.weight_mass) or self.weight_mass <= 0.0:
            raise ValueError("Weight mass must be finite and positive")

        self.pull_target_origin: np.ndarray | None = None
        self.gripper_correction = np.zeros(3)

        self._build_scene()
        self.control = self.model.control()
        self._build_solver(vbd_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        for state in (self.state_0, self.state_1):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        initial_body_q = self._read_measurements()
        if not np.allclose(self.box_target_center[:2], self.mechanism.pull_line_xy, atol=1.0e-5):
            raise ValueError("Graspable box is not initialized on the vertical free-rope line")
        self.start_position = self.gripper_tcp_position.copy()
        self.start_rotation = initial_body_q[self.hand_body, 3:7].copy()
        self.last_commanded_tcp_z = float(self.start_position[2])
        self.initial_load_z = self.load_z
        self._build_ik()
        self.solver.sync_entry_states(self.state_0)

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)

        newton.examples.configure_coupled_view(self, args)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(pos=wp.vec3(1.35, -2.1, 1.25), pitch=-10.0, yaw=145.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.45, 0.0, 0.62))

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    @staticmethod
    def _add_franka(builder: newton.ModelBuilder) -> None:
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(
                wp.vec3(FRANKA_BASE_X, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi),
            ),
            floating=False,
            joint_damping_mode="drive",
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q

        for finger_label in ("fr3_leftfinger", "fr3_rightfinger"):
            finger_body = _find_label_index(builder.body_label, finger_label)
            rubber_pad_shape = int(builder.body_shapes[finger_body][-1])
            builder.shape_label[rubber_pad_shape] = f"{finger_label}_rubber_contact_pad"

    def _build_scene(self) -> None:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
        builder.rigid_gap = CABLE_CONTACT_GAP
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder)

        franka_body_start = builder.body_count
        franka_joint_start = builder.joint_count
        franka_shape_start = builder.shape_count
        self._add_franka(builder)
        builder.joint_target_ke[:7] = [900.0] * 7
        builder.joint_target_kd[:7] = [90.0] * 7
        builder.joint_target_ke[7:9] = [5000.0, 5000.0]
        builder.joint_target_kd[7:9] = [200.0, 200.0]
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [1000.0, 1000.0]
        builder.joint_armature[:7] = [0.05] * 7
        self.franka_bodies = list(range(franka_body_start, builder.body_count))
        self.franka_joints = list(range(franka_joint_start, builder.joint_count))
        self.franka_shapes = list(range(franka_shape_start, builder.shape_count))
        self.hand_body = _find_label_index(builder.body_label, "fr3_hand")

        self.mechanism = add_pulley_mechanism(builder, self.mechanical_advantage, self.weight_mass)

        self.gripper_bodies = [body for body in self.franka_bodies if "finger" in builder.body_label[body]]
        self.gripper_contact_shapes = {
            shape for shape in self.franka_shapes if builder.shape_label[shape].endswith("_contact_pad")
        }
        for franka_shape in self.franka_shapes:
            if franka_shape in self.gripper_contact_shapes:
                continue
            for vbd_shape in self.mechanism.shapes:
                builder.add_shape_collision_filter_pair(franka_shape, vbd_shape)

        for shape in self.gripper_contact_shapes:
            builder.shape_material_ke[shape] = GRIPPER_CONTACT_STIFFNESS
            builder.shape_material_kd[shape] = GRIPPER_CONTACT_DAMPING
            builder.shape_material_mu[shape] = GRIPPER_FRICTION

        builder.color(balance_colors=False)
        self.model = builder.finalize()
        shape_body = self.model.shape_body.numpy()
        shape_transform = self.model.shape_transform.numpy()
        self.gripper_pad_local_centers = tuple(
            (
                int(shape_body[shape]),
                np.asarray(shape_transform[shape, :3], dtype=np.float64),
            )
            for shape in sorted(self.gripper_contact_shapes)
        )
        self.device = self.model.device

    def _build_solver(self, vbd_iterations: int) -> None:
        entries = [
            SolverCoupled.Entry(
                name="mjc",
                solver=lambda view: SolverMuJoCo(
                    model=view,
                    solver="newton",
                    integrator="implicitfast",
                    iterations=12,
                    ls_iterations=25,
                    use_mujoco_contacts=False,
                    njmax=256,
                    nconmax=64,
                ),
                bodies=self.franka_bodies,
                joints=self.franka_joints,
                shapes=self.franka_shapes,
            ),
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda view: SolverVBD(
                    model=view,
                    iterations=vbd_iterations,
                    rigid_body_contact_buffer_size=512,
                    rigid_contact_history=False,
                    rigid_compliant_alm=True,
                ),
                bodies=self.mechanism.bodies,
                joints=self.mechanism.joints,
                shapes=self.mechanism.shapes,
            ),
        ]
        self.solver = SolverCoupledADMM(
            model=self.model,
            entries=entries,
            coupling=SolverCoupledADMM.Config(
                iterations=1,
                rho=200.0,
                gamma=0.001,
                baumgarte=0.5,
                rigid_contact_matching="sticky",
                contact_matching_force_scale=1.0,
                contact_pairs=[SolverCoupledADMM.ContactPair(source="mjc", destination="vbd")],
            ),
        )
        self._configure_force_measurement()

    def _configure_force_measurement(self) -> None:
        """Map finger wrenches and remove ADMM's added-mass gravity contribution."""
        self.mujoco_solver = self.solver.solver("mjc")
        mujoco_view = self.solver.view("mjc")
        finger_local_bodies = {i for i, label in enumerate(mujoco_view.body_label) if "finger" in label}
        mujoco_body_to_newton = self.mujoco_solver.mjc_body_to_newton.numpy()[0]
        self.mujoco_finger_bodies = [
            i for i, body in enumerate(mujoco_body_to_newton) if int(body) in finger_local_bodies
        ]
        if len(self.mujoco_finger_bodies) != 2:
            raise ValueError("Could not map both Franka finger bodies into MuJoCo")

        self.robot_proximal_gravity_compensation = 0.0
        global_finger_by_label = {self.model.body_label[body]: body for body in self.gripper_bodies}
        local_mass = mujoco_view.body_mass.numpy()
        global_mass = self.model.body_mass.numpy()
        local_gravity = wp.empty(mujoco_view.body_count, dtype=wp.vec3, device=self.device)
        self.mujoco_solver.coupling_eval_gravity_acceleration(local_gravity, None)
        local_gravity = local_gravity.numpy()

        for local_body in finger_local_bodies:
            label = mujoco_view.body_label[local_body]
            if label not in global_finger_by_label:
                raise ValueError(f"Could not map MuJoCo finger body {label!r} into the full model")
            global_body = global_finger_by_label[label]
            proximal_mass = max(0.0, float(local_mass[local_body] - global_mass[global_body]))
            self.robot_proximal_gravity_compensation -= proximal_mass * float(local_gravity[local_body, 2])

    def _build_ik(self) -> None:
        ik_builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
        self._add_franka(ik_builder)
        self.ik_model = ik_builder.finalize(device=self.device)
        self.n_coords = self.ik_model.joint_coord_count
        self.ik_joint_q = wp.clone(self.model.joint_q.reshape((1, -1))[:, : self.n_coords])
        self.control_joint_target_q = self.control.joint_target_q.reshape((1, -1))
        self.finger_index_0 = self.n_coords - 2
        self.finger_index_1 = self.n_coords - 1
        self.finger_position = wp.full(1, GRIP_OPEN, dtype=float, device=self.device)

        self.ik_target_positions = wp.array([self.start_position], dtype=wp.vec3, device=self.device)
        self.ik_target_rotations = wp.array([self.start_rotation], dtype=wp.vec4, device=self.device)
        ik_hand_body = _find_label_index(self.ik_model.body_label, "fr3_hand")
        position_objective = ik.IKObjectivePosition(
            link_index=ik_hand_body,
            link_offset=wp.vec3(0.0, 0.0, 0.107),
            target_positions=self.ik_target_positions,
        )
        rotation_objective = ik.IKObjectiveRotation(
            link_index=ik_hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.ik_target_rotations,
        )
        joint_limits_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.clone(self.model.joint_limit_lower[: self.n_coords]),
            joint_limit_upper=wp.clone(self.model.joint_limit_upper[: self.n_coords]),
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[position_objective, rotation_objective, joint_limits_objective],
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def update_ik_target(self) -> None:
        if self.sim_time > GRASP_END_TIME and self.pull_target_origin is None:
            # Track the box during approach, then freeze the pull origin and correction.
            self.pull_target_origin = self.box_target_center.copy()
        center = self.box_target_center.copy() if self.pull_target_origin is None else self.pull_target_origin.copy()
        center[:2] = self.mechanism.pull_line_xy
        position, rotation, grip = gripper_target(self.sim_time, self.start_position, self.start_rotation, center)

        if 0.0 < self.sim_time <= GRASP_END_TIME:
            measured_offset = self.gripper_tcp_position - self.gripper_pad_midpoint
            measured_offset[2] = self.last_commanded_tcp_z - self.gripper_pad_midpoint[2]
            lateral = GRIPPER_LATERAL_CORRECTION_MAX
            self.gripper_correction = np.clip(
                (1.0 - GRIPPER_CORRECTION_FILTER) * self.gripper_correction
                + GRIPPER_CORRECTION_FILTER * measured_offset,
                [-lateral, -lateral, 0.0],
                [lateral, lateral, GRIPPER_HEIGHT_CORRECTION_MAX],
            )
        if self.sim_time > 0.0:
            position += self.gripper_correction
        self.last_commanded_tcp_z = float(position[2])

        wp.launch(
            set_task_target,
            dim=1,
            inputs=[
                self.ik_target_positions,
                self.ik_target_rotations,
                self.finger_position,
                wp.vec3(*position),
                wp.vec4(*rotation),
                grip,
            ],
            device=self.device,
        )

    def simulate(self) -> None:
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_target,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_position, self.finger_index_0, self.finger_index_1],
            device=self.device,
        )
        wp.copy(dest=self.control_joint_target_q[:, : self.n_coords], src=self.ik_joint_q)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # A captured frame must read/write the same buffers on every replay.
        if self.sim_substeps % 2:
            self.state_1.assign(self.state_0)
            self.state_0, self.state_1 = self.state_1, self.state_0

    @staticmethod
    def _transform_point(pose: np.ndarray, local_point: np.ndarray) -> np.ndarray:
        return np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*local_point)), dtype=np.float64)

    def _end_box_center(self, body_q: np.ndarray) -> np.ndarray:
        return self._transform_point(
            body_q[self.mechanism.cable_bodies[-1]],
            np.array([0.0, 0.0, self.mechanism.end_box_center_offset], dtype=np.float64),
        )

    def _gripper_pad_midpoint(self, body_q: np.ndarray) -> np.ndarray:
        return np.mean(
            [
                self._transform_point(body_q[body], local_center)
                for body, local_center in self.gripper_pad_local_centers
            ],
            axis=0,
        )

    def _gripper_tcp_position(self, body_q: np.ndarray) -> np.ndarray:
        return self._transform_point(body_q[self.hand_body], np.array([0.0, 0.0, 0.107], dtype=np.float64))

    def _measure_robot_downward_force(self) -> float:
        if self.mujoco_solver.use_mujoco_cpu:
            applied_wrenches = np.asarray(self.mujoco_solver.mj_data.xfrc_applied)
        else:
            applied_wrenches = self.mujoco_solver.mjw_data.xfrc_applied.numpy()[0]
        upward_reaction = float(np.sum(applied_wrenches[self.mujoco_finger_bodies, 2]))
        physical_reaction = upward_reaction - self.robot_proximal_gravity_compensation
        return max(0.0, physical_reaction)

    def _read_measurements(self) -> np.ndarray:
        """Refresh control feedback independently of rendering and diagnostic plots."""
        body_q = self.state_0.body_q.numpy()
        self.load_z = float(body_q[self.mechanism.weight_body, 2])
        self.box_target_center = self._end_box_center(body_q)
        self.gripper_pad_midpoint = self._gripper_pad_midpoint(body_q)
        self.gripper_tcp_position = self._gripper_tcp_position(body_q)
        return body_q

    def _log_diagnostics(self) -> None:
        force = self._measure_robot_downward_force()
        force_ratio = self.weight_mass * GRAVITY / force if force > 1.0e-3 else 0.0
        self.viewer.log_scalar("Robot downward force [N]", force, smoothing=10)
        self.viewer.log_scalar("Measured force advantage", force_ratio, smoothing=10)
        self.viewer.log_scalar("Load lift [m]", self.load_z - self.initial_load_z, smoothing=10)

    def step(self) -> None:
        self.update_ik_target()
        if self.device.is_cuda:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._read_measurements()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self._log_diagnostics()
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(body_qd)):
            raise ValueError("Pulley mechanism contains NaN or inf body state")

    def test_final(self) -> None:
        if self.sim_time + 0.5 * self.frame_dt < EXAMPLE_DURATION:
            raise ValueError(f"Lift validation requires the full {EXAMPLE_DURATION:g} s trajectory")
        body_q = self.state_0.body_q.numpy()
        final_load_z = float(body_q[self.mechanism.weight_body, 2])
        load_lift = final_load_z - self.initial_load_z
        expected_load_lift = PULL_DISTANCE / self.mechanical_advantage

        if not math.isclose(load_lift, expected_load_lift, rel_tol=0.2):
            raise ValueError(
                f"The {self.weight_mass:.0f} kg weight lifted {load_lift:.3f} m; "
                f"expected approximately {expected_load_lift:.3f} m for a "
                f"{self.mechanical_advantage}:1 mechanical advantage"
            )

    @staticmethod
    def create_parser():
        parser = examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.set_defaults(num_frames=math.ceil(EXAMPLE_DURATION * 60))
        parser.add_argument(
            "--mechanical-advantage",
            type=int,
            default=DEFAULT_MECHANICAL_ADVANTAGE,
            help="Even supporting-strand count (minimum 2).",
        )
        parser.add_argument(
            "--weight-mass",
            type=float,
            default=DEFAULT_WEIGHT_MASS,
            help="Intended lifted mass before cable-end compensation [kg].",
        )
        parser.add_argument("--substeps", type=int, default=16, help="Coupled substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=10, help="VBD iterations per coupled substep.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser)
    example = Example(viewer, args)
    examples.run(example, args)
