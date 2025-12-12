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
# keyboard control. Use I/K for throttle / brake and J/L for steering.
# MuJoCo Warp is used for dynamics integration.
#
# Command: python -m newton.examples robot_racecar
#
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
import newton.examples


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

        builder.add_ground_plane()

        # load the URDF and place it just above the plane
        builder.add_urdf(
            newton.examples.get_asset("racecar/racecar.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
        )

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=200, nconmax=200)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        self.drive_dofs: list[int] = []
        self.steer_dofs: list[int] = []
        self._identify_dofs()
        self.drive_torque = 25.0
        self.brake_damping = 5.0
        self.max_steer_angle = math.radians(30.0)

        # For simplicity, disable CUDA graph capture in this example since
        # we modify control arrays on the host each frame.
        self.graph = None

    def _identify_dofs(self):
        joint_qd_start = self.model.joint_qd_start.numpy()
        for joint_index, key in enumerate(self.model.joint_key):
            name = key.lower()
            dof_start = joint_qd_start[joint_index]
            dof_end = joint_qd_start[joint_index + 1]
            if "wheel" in name:
                self.drive_dofs.extend(range(dof_start, dof_end))
            elif "steering" in name:
                self.steer_dofs.extend(range(dof_start, dof_end))

    def _apply_keyboard_control(self):
        if self.control.joint_f is not None:
            self.control.joint_f.zero_()
        if self.control.joint_target_pos is not None:
            self.control.joint_target_pos.zero_()
        if self.control.joint_target_vel is not None:
            self.control.joint_target_vel.zero_()

        throttle = 0.0
        steering = 0.0
        braking = False

        if hasattr(self.viewer, "is_key_down"):
            throttle += 1.0 if self.viewer.is_key_down("i") else 0.0
            throttle -= 1.0 if self.viewer.is_key_down("k") else 0.0
            steering += 1.0 if self.viewer.is_key_down("j") else 0.0
            steering -= 1.0 if self.viewer.is_key_down("l") else 0.0
            braking = bool(self.viewer.is_key_down("space"))

        torque = self.drive_torque * throttle
        if braking:
            torque -= self.brake_damping

        if self.control.joint_f is not None:
            # write via NumPy view since wp.array does not support item assignment
            joint_f_np = self.control.joint_f.numpy()
            for dof in self.drive_dofs:
                joint_f_np[dof] = torque
            self.control.joint_f = wp.array(joint_f_np, dtype=self.control.joint_f.dtype, device=self.control.joint_f.device)

        if self.control.joint_target_pos is not None and self.control.joint_target_vel is not None:
            target_angle = self.max_steer_angle * wp.clamp(steering, -1.0, 1.0)
            joint_target_pos_np = self.control.joint_target_pos.numpy()
            joint_target_vel_np = self.control.joint_target_vel.numpy()
            for dof in self.steer_dofs:
                joint_target_pos_np[dof] = target_angle
                joint_target_vel_np[dof] = 0.0
            self.control.joint_target_pos = wp.array(
                joint_target_pos_np,
                dtype=self.control.joint_target_pos.dtype,
                device=self.control.joint_target_pos.device,
            )
            self.control.joint_target_vel = wp.array(
                joint_target_vel_np,
                dtype=self.control.joint_target_vel.dtype,
                device=self.control.joint_target_vel.device,
            )

    def _simulate_once(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self._apply_keyboard_control()

            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
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
