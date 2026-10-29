"""UR5 + Tesollo DG5F-S (left) geometric fabric.

Variant A (``mode="cspace"``, default): full joint-space forcing attractor —
policy issues ``q*`` (26-D) consumed by ``forcing_cspace_attractor``.

Variant B (``mode="pose"``): palm pose + raw hand joint attractor. Palm control
frames are not yet in ``ur5_dg5fs_left_fabric.urdf``, so ``mode="pose"`` raises
until those frames are added (Phase 1 CP1 only needs cspace).

Mirrors ``fabrics_sim.fabrics.kuka_allegro_pose_fabric.KukaAllegroPoseFabric``.
"""

from __future__ import annotations

from typing import Literal

import torch

from fabrics_sim.energy.euclidean_energy import EuclideanEnergy
from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import (
    BaseFabricRepulsion,
    BodySphereRepulsion,
)
from fabrics_sim.fabric_terms.joint_limit_repulsion import (
    JointLimitBarrier,
    JointLimitRepulsion,
)
from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.taskmaps.identity import IdentityMap
from fabrics_sim.taskmaps.lower_joint_limit import LowerJointLimitMap
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.taskmaps.slice_taskmap import SliceMap
from fabrics_sim.taskmaps.upper_joint_limit import UpperJointLimitMap
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.utils.rotation_utils import (
    euler_to_matrix,
    matrix_to_euler,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

FabricMode = Literal["cspace", "pose"]

# Match RobotCfg.arm_default_joint_pos (simtoolreal_tesollo_env_cfg.py).
_ARM_DEFAULT = (
    0.1654,  # shoulder_pan
    -0.7681,  # shoulder_lift
    1.4818,  # elbow
    0.7414,  # wrist_1
    1.5830,  # wrist_2
    -1.4057,  # wrist_3
)

# Lightly curled fingers (DextrAH-G B.4 posture bias). Thumb mid/distal are
# negative-flexion in our baked limits; other fingers positive.
# left_joint_5_1 upper is 0 — keep a small interior margin so HD2 / limit
# diagnostics are not trivially zero at the default posture.
_HAND_DEFAULT = tuple(
    v
    for finger in range(1, 6)
    for v in (
        (-0.2 if finger == 5 else 0.0),  # spread / yaw
        0.4,  # proximal flex
        (-0.5 if finger == 1 else 0.5),
        (-0.5 if finger == 1 else 0.5),
    )
)

NUM_ARM = 6
NUM_HAND = 20
NUM_JOINTS = NUM_ARM + NUM_HAND


class Ur5TesolloFabric(BaseFabric):
    """Geometric fabric for UR5 + Tesollo DG5F-S left hand."""

    def __init__(
        self,
        batch_size: int,
        device: str,
        timestep: float,
        graph_capturable: bool = True,
        mode: FabricMode = "cspace",
        fabric_params_filename: str = "ur5_tesollo_params.yaml",
    ):
        if mode not in ("cspace", "pose"):
            raise ValueError(f"mode must be 'cspace' or 'pose', got {mode!r}")
        self.mode: FabricMode = mode
        self.num_arm = NUM_ARM
        self.num_hand = NUM_HAND

        super().__init__(
            device,
            batch_size,
            timestep,
            fabric_params_filename,
            graph_capturable=graph_capturable,
        )

        robot_dir_name = "ur5_tesollo"
        # Each sphere model has its own fabric URDF (only its frames); the params yaml names it.
        robot_name = self.fabric_params.get("robot_urdf", "ur5_dg5fs_left_fabric")
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        self.load_robot(robot_dir_name, robot_name, batch_size)

        if self._num_joints != NUM_JOINTS:
            raise RuntimeError(
                f"expected {NUM_JOINTS} revolute joints, got {self._num_joints}. "
                f"Check {self.urdf_path}"
            )
        if len(self.accel_limits) != NUM_JOINTS:
            raise RuntimeError(
                f"joint_limits.acceleration length {len(self.accel_limits)} "
                f"!= {NUM_JOINTS} (params yaml)"
            )

        default_config = torch.tensor(
            list(_ARM_DEFAULT) + list(_HAND_DEFAULT), device=self.device
        )
        self.default_config = default_config.unsqueeze(0).repeat(self.batch_size, 1)

        self._palm_pose_target = torch.zeros(batch_size, 12, device=device)
        self._native_palm_pose_target = None

        self.construct_fabric()

    # ------------------------------------------------------------------
    # Fabric terms
    # ------------------------------------------------------------------

    def add_joint_limit_repulsion(self) -> None:
        """Joint-limit term per side. ``joint_limit_repulsion.type``: ``adept`` = App. B.4
        barrier on range-normalised clearance (tanh gate); ``kuka`` = upstream term in rad."""
        joints = self.urdfpy_robot.joints
        upper_joint_limits = [
            j.limit.upper for j in joints if j.joint_type == "revolute"
        ]
        lower_joint_limits = [
            j.limit.lower for j in joints if j.joint_type == "revolute"
        ]
        params = self.fabric_params["joint_limit_repulsion"]
        if params.get("type", "kuka") == "adept":
            joint_range = [u - l for u, l in zip(upper_joint_limits, lower_joint_limits)]
            term_cls = JointLimitBarrier
        else:
            joint_range = None
            term_cls = JointLimitRepulsion

        for taskmap_name, taskmap in (
            ("upper_joint_limit", UpperJointLimitMap(
                upper_joint_limits, self.batch_size, self.device, joint_range)),
            ("lower_joint_limit", LowerJointLimitMap(
                lower_joint_limits, self.batch_size, self.device, joint_range)),
        ):
            self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
            self.add_fabric(
                taskmap_name,
                "joint_limit_repulsion",
                term_cls(True, params, self.device, graph_capturable=self.graph_capturable),
            )

    def add_cspace_attractor(self, is_forcing: bool) -> None:
        taskmap_name = "identity"
        self.add_taskmap(
            taskmap_name, IdentityMap(self.device), graph_capturable=self.graph_capturable
        )

        if is_forcing:
            fabric_name = "forcing_cspace_attractor"
            params = self.fabric_params["forcing_cspace_attractor"]
        else:
            fabric_name = "cspace_attractor"
            params = self.fabric_params["cspace_attractor"]

        self.add_fabric(
            taskmap_name,
            fabric_name,
            Attractor(
                is_forcing, params, self.device, graph_capturable=self.graph_capturable
            ),
        )

    def add_forcing_attractors(self) -> None:
        """ADEPT App. B.2 forcing cspace attractor(s) carrying the policy target q*.

        ``forcing_partition: true`` splits q* = [q*_arm, q*_hand] into two attractors on joint
        slices (``forcing_arm_attractor`` / ``forcing_hand_attractor`` yaml blocks, defaulting to
        ``forcing_cspace_attractor``). Each has its own error norm, so a large finger error does
        not use up the arm's conical pull, and each damping radius engages on its own group.
        ``false`` keeps the single 26-D attractor on ``identity``."""
        if not self.fabric_params.get("forcing_partition", False):
            self.add_cspace_attractor(True)
            return
        groups = (("forcing_arm", list(range(self.num_arm)), "forcing_arm_attractor"),
                  ("forcing_hand", list(range(self.num_arm, self.num_arm + self.num_hand)),
                   "forcing_hand_attractor"))
        for taskmap_name, indices, block in groups:
            params = self.fabric_params.get(block, self.fabric_params["forcing_cspace_attractor"])
            self.add_taskmap(taskmap_name, SliceMap(indices, self.device),
                             graph_capturable=self.graph_capturable)
            self.add_fabric(taskmap_name, block,
                            Attractor(True, params, self.device, graph_capturable=self.graph_capturable))

    def add_hand_joint_attractor(self) -> None:
        """Forcing attractor on the 20 hand joints (Variant B raw hand)."""
        hand_indices = list(range(self.num_arm, self.num_arm + self.num_hand))
        taskmap_name = "hand_joints"
        self.add_taskmap(
            taskmap_name,
            SliceMap(hand_indices, self.device),
            graph_capturable=self.graph_capturable,
        )
        self.add_fabric(
            taskmap_name,
            "hand_attractor",
            Attractor(
                True,
                self.fabric_params["hand_attractor"],
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )

    def add_palm_points_attractor(self) -> None:
        """7-point palm pose attractor (Variant B). Requires palm_* URDF frames."""
        taskmap_name = "palm"
        control_point_frames = [
            "palm",
            "palm_x",
            "palm_x_neg",
            "palm_y",
            "palm_y_neg",
            "palm_z",
            "palm_z_neg",
        ]
        taskmap = RobotFrameOriginsTaskMap(
            self.urdf_path, control_point_frames, self.batch_size, self.device
        )
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        self.add_fabric(
            taskmap_name,
            "palm_attractor",
            Attractor(
                True,
                self.fabric_params["palm_attractor"],
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )

    def add_body_repulsion(self) -> None:
        collision_sphere_frames = self.fabric_params["body_repulsion"][
            "collision_sphere_frames"
        ]
        self.collision_sphere_radii = self.fabric_params["body_repulsion"][
            "collision_sphere_radii"
        ]
        assert len(collision_sphere_frames) == len(self.collision_sphere_radii), (
            "collision_sphere_frames and collision_sphere_radii length mismatch"
        )

        collision_sphere_pairs = list(
            self.fabric_params["body_repulsion"]["collision_sphere_pairs"]
        )
        # Must be int64: Warp kernel expects wp.int64 (Kuka uses dtype=int → int64).
        collision_matrix = torch.zeros(
            len(collision_sphere_frames),
            len(collision_sphere_frames),
            dtype=torch.int64,
            device=self.device,
        )

        if len(collision_sphere_pairs) == 0:
            for prefix1, prefix2 in self.fabric_params["body_repulsion"][
                "collision_link_prefix_pairs"
            ]:
                frames1 = [s for s in collision_sphere_frames if prefix1 in s]
                frames2 = [s for s in collision_sphere_frames if prefix2 in s]
                for s1 in frames1:
                    for s2 in frames2:
                        collision_sphere_pairs.append([s1, s2])

        for sphere1, sphere2 in collision_sphere_pairs:
            i = collision_sphere_frames.index(sphere1)
            j = collision_sphere_frames.index(sphere2)
            collision_matrix[i, j] = 1

        taskmap_name = "body_points"
        self.add_taskmap(
            taskmap_name,
            RobotFrameOriginsTaskMap(
                self.urdf_path, collision_sphere_frames, self.batch_size, self.device
            ),
            graph_capturable=self.graph_capturable,
        )

        sphere_radius = torch.tensor(
            self.collision_sphere_radii, device=self.device
        ).repeat(self.batch_size, 1)

        self.add_fabric(
            taskmap_name,
            "repulsion",
            BodySphereRepulsion(
                True,
                self.fabric_params["body_repulsion"],
                self.batch_size,
                sphere_radius,
                collision_matrix,
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )
        self.add_fabric(
            taskmap_name,
            "geom_repulsion",
            BodySphereRepulsion(
                False,
                self.fabric_params["body_repulsion"],
                self.batch_size,
                sphere_radius,
                collision_matrix,
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )

        self.base_fabric_repulsion = BaseFabricRepulsion(
            self.fabric_params["body_repulsion"],
            self.batch_size,
            sphere_radius,
            collision_matrix,
            self.device,
        )

    def add_cspace_energy(self) -> None:
        """Energization always uses the Euclidean energy on all 26 joints (upstream).
        ``speed_control.energy``: ``cspace`` = speed control also measures that energy;
        ``arm_palm`` = ADEPT App. B.5, speed control measures an arm-joint slice (beta
        ``arm_energy_scale``) + palm-frame origin (beta ``palm_energy_scale``), hand excluded.
        ADEPT also energizes with the arm/palm energy; here that made alpha singular whenever
        the arm stops while the fingers move (thumb run away to 8 rad/s), so it is not used."""
        sc = self.fabric_params["speed_control"]
        arm_palm = sc.get("energy", "cspace") == "arm_palm"
        self.add_energy(
            "identity",
            "euclidean",
            EuclideanEnergy(self.batch_size, self._num_joints, self.device),
            role="energize" if arm_palm else "both",
        )
        if not arm_palm:
            return
        self.add_taskmap(
            "arm_joints",
            SliceMap(list(range(self.num_arm)), self.device),
            graph_capturable=self.graph_capturable,
        )
        self.add_energy(
            "arm_joints",
            "euclidean",
            EuclideanEnergy(self.batch_size, self.num_arm, self.device, sc["arm_energy_scale"]),
            role="speed",
        )
        self.add_taskmap(
            "palm_origin",
            RobotFrameOriginsTaskMap(
                self.urdf_path, [sc["palm_frame"]], self.batch_size, self.device
            ),
            graph_capturable=self.graph_capturable,
        )
        self.add_energy(
            "palm_origin",
            "euclidean",
            EuclideanEnergy(self.batch_size, 3, self.device, sc["palm_energy_scale"]),
            role="speed",
        )

    def construct_fabric(self) -> None:
        self.add_joint_limit_repulsion()
        self.add_cspace_attractor(False)
        self.add_body_repulsion()
        self.add_cspace_energy()

        if self.mode == "cspace":
            self.add_forcing_attractors()
        else:
            # Variant B: needs palm_* frames in the fabric URDF (not present yet).
            raise NotImplementedError(
                "mode='pose' needs palm / palm_±{x,y,z} frames in "
                "ur5_dg5fs_left_fabric.urdf. Use mode='cspace' for Phase 1 / Variant A."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_sphere_radii(self):
        return self.collision_sphere_radii

    @property
    def collision_status(self):
        return self.base_fabric_repulsion.collision_status

    def get_palm_pose(self, cspace_position, orientation_convention: str):
        if "palm" not in self.taskmap_containers:
            raise RuntimeError(
                "palm taskmap not built (cspace mode or missing palm frames)"
            )

        palm_points, _ = self.get_taskmap("palm")(cspace_position, None)
        palm_origin = palm_points[:, :3]
        x_point = palm_points[:, 3:6]
        y_point = palm_points[:, 9:12]
        z_point = palm_points[:, 15:18]

        x_axis = torch.nn.functional.normalize(x_point - palm_origin, dim=1)
        y_axis = torch.nn.functional.normalize(y_point - palm_origin, dim=1)
        z_axis = torch.nn.functional.normalize(z_point - palm_origin, dim=1)

        rotation_matrix = torch.zeros(self.batch_size, 3, 3, device=self.device)
        rotation_matrix[:, :, 0] = x_axis
        rotation_matrix[:, :, 1] = y_axis
        rotation_matrix[:, :, 2] = z_axis

        if orientation_convention == "euler_zyx":
            orientation = matrix_to_euler(rotation_matrix)
        elif orientation_convention == "quaternion":
            orientation = matrix_to_quaternion(rotation_matrix)[:, [1, 2, 3, 0]]
        else:
            raise ValueError(
                'orientation_convention must be "euler_zyx" or "quaternion"'
            )
        return torch.cat([palm_origin, orientation], dim=-1)

    def convert_transform_to_points(self):
        palm_transform = torch.zeros(self.batch_size, 4, 4, device=self.device)
        palm_transform[:, 3, 3] = 1.0
        palm_transform[:, :3, :3] = torch.transpose(
            self._palm_pose_target[:, 3:].reshape(self.batch_size, 3, 3), 1, 2
        )
        palm_transform[:, :3, 3] = self._palm_pose_target[:, :3]

        def _axis_point(axis: int, sign: float):
            p = torch.zeros(self.batch_size, 4, device=self.device)
            p[:, 3] = 1.0
            p[:, axis] = sign * 0.25
            return p

        palm_targets = torch.zeros(self.batch_size, 7 * 3, device=self.device)
        palm_targets[:, :3] = self._palm_pose_target[:, :3]
        palm_targets[:, 3:6] = torch.bmm(
            palm_transform, _axis_point(0, 1.0).unsqueeze(2)
        ).squeeze(2)[:, :3]
        palm_targets[:, 6:9] = torch.bmm(
            palm_transform, _axis_point(0, -1.0).unsqueeze(2)
        ).squeeze(2)[:, :3]
        palm_targets[:, 9:12] = torch.bmm(
            palm_transform, _axis_point(1, 1.0).unsqueeze(2)
        ).squeeze(2)[:, :3]
        palm_targets[:, 12:15] = torch.bmm(
            palm_transform, _axis_point(1, -1.0).unsqueeze(2)
        ).squeeze(2)[:, :3]
        palm_targets[:, 15:18] = torch.bmm(
            palm_transform, _axis_point(2, 1.0).unsqueeze(2)
        ).squeeze(2)[:, :3]
        palm_targets[:, 18:21] = torch.bmm(
            palm_transform, _axis_point(2, -1.0).unsqueeze(2)
        ).squeeze(2)[:, :3]
        return palm_targets

    # ------------------------------------------------------------------
    # Features / actions
    # ------------------------------------------------------------------

    def _update_body_repulsion(
        self, q: torch.Tensor, qd: torch.Tensor, object_ids, object_indicator
    ) -> None:
        body_point_pos, jac = self.get_taskmap("body_points")(q, None)
        body_point_vel = torch.bmm(jac, qd.unsqueeze(2)).squeeze(2)
        self.base_fabric_repulsion.calculate_response(
            body_point_pos, body_point_vel, object_ids, object_indicator
        )
        self.fabrics_features["body_points"]["repulsion"] = self.base_fabric_repulsion
        self.fabrics_features["body_points"]["geom_repulsion"] = (
            self.base_fabric_repulsion
        )

    def set_features(self, *args, **kwargs):
        """Dispatch to cspace or pose feature setters.

        Cspace (Variant A)::

            set_features(q_target, q_f, qd_f, object_ids, object_indicator,
                         cspace_damping_gain=None, speed_energy_target=None)

        Pose (Variant B, once palm frames exist)::

            set_features(hand_target, palm_pose_target, orientation_convention,
                         q_f, qd_f, object_ids, object_indicator,
                         cspace_damping_gain=None)
        """
        if self.mode == "cspace":
            return self.set_features_cspace(*args, **kwargs)
        return self.set_features_pose(*args, **kwargs)

    def set_features_cspace(
        self,
        q_target: torch.Tensor,
        q_f: torch.Tensor,
        qd_f: torch.Tensor,
        object_ids,
        object_indicator,
        cspace_damping_gain=None,
        speed_energy_target=None,
    ) -> None:
        """Variant A: ``q_target`` is bx26 C-space target for the forcing attractor(s).

        ``cspace_damping_gain`` / ``speed_energy_target``: optional per-env (N, 1) tensors
        (ADR-annealed, ADEPT App. B.8). They are the captured graph inputs, so writing them in
        place changes the fabric without a re-capture."""
        assert q_target.shape == (self.batch_size, NUM_JOINTS), (
            f"q_target shape {tuple(q_target.shape)} != "
            f"({self.batch_size}, {NUM_JOINTS})"
        )
        if self.fabric_params.get("forcing_partition", False):
            self.fabrics_features["forcing_arm"]["forcing_arm_attractor"] = q_target[:, : self.num_arm]
            self.fabrics_features["forcing_hand"]["forcing_hand_attractor"] = q_target[:, self.num_arm :]
        else:
            self.fabrics_features["identity"]["forcing_cspace_attractor"] = q_target
        self.fabrics_features["identity"]["cspace_attractor"] = self.default_config
        self._update_body_repulsion(q_f, qd_f, object_ids, object_indicator)
        if cspace_damping_gain is not None:
            self.fabric_params["cspace_damping"]["gain"] = cspace_damping_gain
        if speed_energy_target is not None:
            self.fabric_params["speed_control"]["energy_target"] = speed_energy_target

    def set_features_pose(
        self,
        hand_target: torch.Tensor,
        palm_pose_target: torch.Tensor,
        orientation_convention: str,
        q_f: torch.Tensor,
        qd_f: torch.Tensor,
        object_ids,
        object_indicator,
        cspace_damping_gain=None,
    ) -> None:
        """Variant B: palm pose + hand joint targets (requires palm taskmap)."""
        if "palm" not in self.taskmap_containers:
            raise RuntimeError("palm taskmap missing; construct with mode='pose'")

        self.fabrics_features["hand_joints"]["hand_attractor"] = hand_target
        self.fabrics_features["identity"]["cspace_attractor"] = self.default_config

        self._palm_pose_target[:, :3] = palm_pose_target[:, :3]
        if orientation_convention == "euler_zyx":
            assert palm_pose_target.shape[1] == 6
            self._palm_pose_target[:, 3:] = torch.transpose(
                euler_to_matrix(palm_pose_target[:, 3:]), 1, 2
            ).reshape(self.batch_size, 9)
        elif orientation_convention == "quaternion":
            assert palm_pose_target.shape[1] == 7
            self._palm_pose_target[:, 3:] = torch.transpose(
                quaternion_to_matrix(palm_pose_target[:, [6, 3, 4, 5]]), 1, 2
            ).reshape(self.batch_size, 9)
        else:
            raise ValueError(
                'orientation_convention must be "euler_zyx" or "quaternion"'
            )

        native = self.convert_transform_to_points()
        if self._native_palm_pose_target is None:
            self._native_palm_pose_target = torch.clone(native)
        else:
            self._native_palm_pose_target.copy_(native)

        self.fabrics_features["palm"]["palm_attractor"] = self._native_palm_pose_target
        self.get_fabric_term("palm", "palm_attractor").damping_position = (
            self._native_palm_pose_target
        )

        self._update_body_repulsion(q_f, qd_f, object_ids, object_indicator)
        if cspace_damping_gain is not None:
            self.fabric_params["cspace_damping"]["gain"] = cspace_damping_gain


__all__ = ["Ur5TesolloFabric", "NUM_ARM", "NUM_HAND", "NUM_JOINTS"]
