# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Robot Racecar
#
# Loads a URDF racecar, drops it on a flat plane, and allows simple
# keyboard control:
#   I / K : increase / decrease desired drive velocity (wheel spin)
#   J / L : steering left / right
#
# Wheels are controlled with target POSITION (integrated wheel angle) and a
# low-level PD controller (joint_target_ke/kd).
# Steering is also controlled with target POSITION and PD.
#
# Command: python -m newton.examples robot_racecar
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
import newton.examples


# Hardcoded actuated joints for racecar.urdf (exact URDF joint names).
DRIVE_JOINTS = [
    "left_rear_wheel_joint",
    "right_rear_wheel_joint",
    # Uncomment for AWD:
    # "left_front_wheel_joint",
    # "right_front_wheel_joint",
]

STEER_JOINTS = [
    "left_steering_hinge_joint",
    "right_steering_hinge_joint",
]


class Example:
    def __init__(self, viewer, args=None):
        # simulation timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # --- Contact tuning for the ground (MuJoCo uses ke/kd too) ---
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=2.0e3,     # contact stiffness (normal “spring”)
            kd=2.0e4,     # contact damping   (normal “damper”)
            mu=1.0,       # friction coefficient
            thickness=1e-4,  # small collision “skin” (optional)
        )

        builder.add_ground_plane(cfg=ground_cfg)

        builder.add_urdf(
            newton.examples.get_asset("racecar/racecar.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
        )

        # Resolve joint-name -> DOF indices on the BUILDER (pre-finalize)
        self.drive_dofs: list[int] = []
        self.steer_dofs: list[int] = []
        self._resolve_actuated_dofs_from_names(builder)

        # --- Low-level PD gains (same pattern as humanoid example) ---
        # Steering PD (target position)
        self.steer_kp = 50.0
        self.steer_kd = 5.0
        for dof in self.steer_dofs:
            builder.joint_target_ke[dof] = self.steer_kp
            builder.joint_target_kd[dof] = self.steer_kd

        # Drive PD (target position for wheel spin)
        # Keep these fairly low to avoid chatter; tune as needed.
        self.drive_kp =  0.0 
        self.drive_kd = 10.0
        for dof in self.drive_dofs:
            builder.joint_target_ke[dof] = self.drive_kp
            builder.joint_target_kd[dof] = self.drive_kd

        # Finalize and create solver
        self.model = builder.finalize()

        # todo much simpler, assign an array here that sets all values to 1
        condim_np = self.model.mujoco.condim.numpy()
        condim_np[:] = 1
        self.model.mujoco.condim = wp.array(
            condim_np,
            dtype=self.model.mujoco.condim.dtype,
            device=self.model.mujoco.condim.device,
        )
        print(self.model.mujoco.condim.numpy()) # prints [3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=200, nconmax=200)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        # --- High-level commands ---
        # Desired wheel spin velocity (rad/s). Modified by I/K.
        self.desired_drive_vel = 0.0
        self.drive_accel = 20.0          # rad/s^2 per second of key-hold
        self.max_drive_vel = 50.0        # clamp rad/s

        # Steering command (-1..1) from J/L
        self.steer_cmd = 0.0
        self.max_steer_angle = math.radians(30.0)

        # We integrate wheel angle targets ourselves (position commands).
        self.drive_target_pos = [0.0 for _ in self.drive_dofs]

        self.graph = None

        print("drive_dofs:", self.drive_dofs)
        print("steer_dofs:", self.steer_dofs)

    def _resolve_actuated_dofs_from_names(self, builder: newton.ModelBuilder):
        """
        Map explicit URDF joint names to flattened DOF indices using builder.joint_qd_start.
        """
        joint_qd_start = list(builder.joint_qd_start)
        joint_keys = list(builder.joint_key)

        key_to_joint_id = {key: i for i, key in enumerate(joint_keys)}
        lower_to_key = {key.lower(): key for key in joint_keys}

        def add_joint_dofs(joint_name: str, out: list[int]):
            if joint_name in key_to_joint_id:
                jid = key_to_joint_id[joint_name]
            else:
                key = lower_to_key.get(joint_name.lower())
                if key is None:
                    available = ", ".join(joint_keys)
                    raise KeyError(
                        f"Joint '{joint_name}' not found in builder.joint_key.\n"
                        f"Available joints: {available}"
                    )
                jid = key_to_joint_id[key]

            dof_start = int(joint_qd_start[jid])
            dof_end = int(joint_qd_start[jid + 1])
            out.extend(range(dof_start, dof_end))

        for j in DRIVE_JOINTS:
            add_joint_dofs(j, self.drive_dofs)

        for j in STEER_JOINTS:
            add_joint_dofs(j, self.steer_dofs)

    def _update_keyboard_inputs(self):
        """
        Update high-level commands once per FRAME (not per substep):
          - desired_drive_vel integrates with I/K
          - steer_cmd is instantaneous from J/L
        """
        inc = 0.0
        dec = 0.0
        left = 0.0
        right = 0.0

        if hasattr(self.viewer, "is_key_down"):
            inc = 1.0 if self.viewer.is_key_down("i") else 0.0
            dec = 1.0 if self.viewer.is_key_down("k") else 0.0
            left = 1.0 if self.viewer.is_key_down("j") else 0.0
            right = 1.0 if self.viewer.is_key_down("l") else 0.0

        # ramp desired velocity
        self.desired_drive_vel += (inc - dec) * self.drive_accel * self.frame_dt
        self.desired_drive_vel = float(wp.clamp(self.desired_drive_vel, -self.max_drive_vel, self.max_drive_vel))

        # instantaneous steering command
        self.steer_cmd = float(wp.clamp(left - right, -1.0, 1.0))

    def _apply_control_substep(self):
        """
        Apply target position/velocity commands for BOTH drive + steering.

        Drive: we integrate wheel angle targets from desired_drive_vel:
          theta_target += desired_drive_vel * sim_dt
          q_target_pos = theta_target
          q_target_vel = desired_drive_vel

        Steering: position target = +/- max_steer_angle, vel target = 0
        """
        if self.control.joint_f is not None:
            self.control.joint_f.zero_()
        if self.control.joint_target_pos is not None:
            self.control.joint_target_pos.zero_()
        if self.control.joint_target_vel is not None:
            self.control.joint_target_vel.zero_()

        if self.control.joint_target_pos is None or self.control.joint_target_vel is None:
            # If these are None, target-position control isn't available in this build.
            return

        # --- Drive wheel targets (position-based) ---
        for i, dof in enumerate(self.drive_dofs):
            self.drive_target_pos[i] += self.desired_drive_vel * self.sim_dt

        target_pos_np = self.control.joint_target_pos.numpy()
        target_vel_np = self.control.joint_target_vel.numpy()

        for i, dof in enumerate(self.drive_dofs):
            target_pos_np[dof] = self.drive_target_pos[i]
            target_vel_np[dof] = self.desired_drive_vel

        # --- Steering targets (position-based) ---
        target_angle = self.max_steer_angle * self.steer_cmd
        for dof in self.steer_dofs:
            target_pos_np[dof] = target_angle
            target_vel_np[dof] = 0.0

        self.control.joint_target_pos = wp.array(
            target_pos_np,
            dtype=self.control.joint_target_pos.dtype,
            device=self.control.joint_target_pos.device,
        )
        self.control.joint_target_vel = wp.array(
            target_vel_np,
            dtype=self.control.joint_target_vel.dtype,
            device=self.control.joint_target_vel.device,
        )

    def _simulate_once(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self._apply_control_substep()

            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Update commands once per frame
        self._update_keyboard_inputs()

        self._simulate_once()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
