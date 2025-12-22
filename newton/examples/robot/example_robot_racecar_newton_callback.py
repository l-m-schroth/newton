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
# Example Robot Racecar (newton + mjcb_control)
#
# This is the "one-step-lag" baseline, equivalent to `example_robot_racecar_newton.py`,
# but applies the tire forces via MuJoCo-Warp's `mjcb_control` callback instead of
# writing them into `state.body_f`.
#
# With `integrator="euler"` this should behave qualitatively the same as the Newton
# version (given identical inputs), and is useful to sanity-check mappings and
# MuJoCo conventions for external wrenches (`xfrc_applied`).
#
# Command: python -m newton.examples robot_racecar_newton_callback
###########################################################################

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.sensors import populate_contacts


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

WHEEL_SPIN_JOINTS = [
    "left_rear_wheel_joint",
    "right_rear_wheel_joint",
    "left_front_wheel_joint",
    "right_front_wheel_joint",
]

WHEEL_UNLOADED_RADIUS_M = 0.05  # racecar.urdf wheel collision cylinder radius
FIALA_REF_FZ_N = 4000.0  # scale stiffnesses relative to a nominal passenger-car wheel load
FIALA_REF_RADIUS_M = 0.3099  # report's unloaded radius (TR-2015-13)


@dataclass(frozen=True)
class FialaTireParams:
    c_slip: float
    c_alpha: float
    u_max: float
    u_min: float
    rolling_resistance: float
    width: float
    v_eps: float = 0.1  # m/s, avoid division explosion at low speed
    fz_min: float = 1e-3  # N, ignore near-zero load


@dataclass(frozen=True)
class WheelInfo:
    joint_name: str
    joint_id: int
    dof_id: int
    body_id: int
    axis_child: np.ndarray  # (3,), unit axis in child frame
    shape_ids: np.ndarray  # (k,), collision shapes belonging to this wheel body
    radius: float


def _sign(x: float, eps: float = 1e-8) -> float:
    if x > eps:
        return 1.0
    if x < -eps:
        return -1.0
    return 0.0


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q_xyz = q[:3]
    q_w = float(q[3])
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _extract_xform7(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row)
    if row.shape[-1] == 7:
        return row
    if hasattr(row, "p") and hasattr(row, "q"):
        return np.concatenate([np.asarray(row.p), np.asarray(row.q)], axis=0)
    raise ValueError(f"Unsupported transform representation with shape={getattr(row, 'shape', None)}")


def _extract_spatial6(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row)
    if row.shape[-1] == 6:
        return row
    if hasattr(row, "v") and hasattr(row, "w"):
        return np.concatenate([np.asarray(row.v), np.asarray(row.w)], axis=0)
    raise ValueError(f"Unsupported spatial_vector representation with shape={getattr(row, 'shape', None)}")


def _fiala_forces(
    params: FialaTireParams, *, fz: float, v_x: float, v_y: float, omega: float, r_eff: float
) -> tuple[float, float, float, float]:
    if fz < params.fz_min:
        return 0.0, 0.0, 0.0, 0.0

    denom_vx = max(abs(v_x), params.v_eps)
    ss = (omega * r_eff - v_x) / denom_vx

    tan_alpha = v_y / denom_vx
    alpha = math.atan(tan_alpha)

    ss_alpha = math.sqrt(ss * ss + tan_alpha * tan_alpha)
    ss_alpha = min(ss_alpha, 1.0)

    u = params.u_max - (params.u_max - params.u_min) * ss_alpha

    # Longitudinal force (report eqs. 22-24, with |Ss| in the 1/S term for physical symmetry)
    s_crit = (u * fz) / (2.0 * params.c_slip) if params.c_slip > 0.0 else 0.0
    if abs(ss) <= s_crit:
        f_x = params.c_slip * ss
    else:
        fx1 = u * fz
        fx2 = (u * fz) ** 2 / (4.0 * max(abs(ss), 1e-6) * params.c_slip)
        f_x = _sign(ss) * (fx1 - fx2)

    # Lateral force (report eqs. 25-27)
    alpha_crit = math.atan((3.0 * u * fz) / params.c_alpha) if params.c_alpha > 0.0 else 0.0
    if abs(alpha) <= alpha_crit:
        h = 1.0 - (params.c_alpha * abs(tan_alpha)) / (3.0 * u * fz)
        h = float(np.clip(h, 0.0, 1.0))
        f_y = -u * fz * (1.0 - h**3) * _sign(alpha)
        m_z = u * fz * params.width * (1.0 - h) * (h**3) * _sign(alpha)
    else:
        f_y = -u * fz * _sign(alpha)
        m_z = 0.0

    # Rolling resistance (report eq. 28)
    m_y = -params.rolling_resistance * fz * _sign(omega)

    return f_x, f_y, m_y, m_z


@wp.kernel
def _add_xfrc_applied_kernel(
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    """Add Cartesian wrenches into mujoco_warp Data.xfrc_applied (world 0)."""
    tid = wp.tid()
    body = body_ids[tid]
    if body < 0:
        return

    x = xfrc_applied[0, body]
    f = wp.spatial_top(x) + forces[tid]
    t = wp.spatial_bottom(x) + torques[tid]
    xfrc_applied[0, body] = wp.spatial_vector(f, t)


@wp.kernel
def _set_xfrc_applied_kernel(
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    """Overwrite Cartesian wrenches in mujoco_warp Data.xfrc_applied (world 0)."""
    tid = wp.tid()
    body = body_ids[tid]
    if body < 0:
        return

    xfrc_applied[0, body] = wp.spatial_vector(forces[tid], torques[tid])


class Example:
    def __init__(self, viewer, args=None):
        # simulation timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # --- Contact tuning for the ground (MuJoCo uses ke/kd too) ---
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=2.0e3,     # contact stiffness (normal “spring”)
            kd=2.0e4,     # contact damping   (normal “damper”)
            mu=1.0,       # friction coefficient (condim=1 means normal-only anyway)
            thickness=1e-4,  # small collision “skin” (optional)
        )

        builder.add_ground_plane(cfg=ground_cfg)

        builder.add_urdf(
            newton.examples.get_asset("racecar/racecar.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
        )

        # Resolve joint-name -> DOF indices on the BUILDER (pre-finalize)
        self.drive_dofs: list[int] = []
        self.steer_dofs: list[int] = []
        self._resolve_actuated_dofs_from_names(builder)

        # --- Low-level PD gains ---
        self.steer_kp = 50.0
        self.steer_kd = 5.0
        for dof in self.steer_dofs:
            builder.joint_target_ke[dof] = self.steer_kp
            builder.joint_target_kd[dof] = self.steer_kd

        self.drive_kp = 0.0
        self.drive_kd = 10.0
        for dof in self.drive_dofs:
            builder.joint_target_ke[dof] = self.drive_kp
            builder.joint_target_kd[dof] = self.drive_kd

        # Finalize and create solver
        self.model = builder.finalize()

        # Normal-only contacts
        condim_np = self.model.mujoco.condim.numpy()
        condim_np[:] = 1
        self.model.mujoco.condim = wp.array(
            condim_np,
            dtype=self.model.mujoco.condim.dtype,
            device=self.model.mujoco.condim.device,
        )

        # Euler integrator for 1:1 comparison
        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=200, nconmax=200, integrator="rk4")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.mj_contacts = newton.Contacts(0, 0)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        # --- High-level commands ---
        self.desired_drive_vel = 0.0
        self.drive_accel = 20.0
        self.max_drive_vel = 50.0

        self.steer_cmd = 0.0
        self.max_steer_angle = math.radians(30.0)

        self.drive_target_pos = [0.0 for _ in self.drive_dofs]

        self.graph = None

        print("drive_dofs:", self.drive_dofs)
        print("steer_dofs:", self.steer_dofs)

        self._init_tire_model()

        # Install MuJoCo-Warp control callback to apply tire wrenches.
        import mujoco_warp  # noqa: PLC0415

        mujoco_warp.mjcb_control = self._tire_mjcb_control
        mujoco_warp.mjcb_postsolve = None  # ensure Picard loop is disabled

        self._maybe_init_pos_speed(args)

    def _maybe_init_pos_speed(self, args) -> None:
        if args is None or not getattr(args, "init_pos_speed", False):
            return

        v0 = 2.0  # m/s, nominal tire model range and within max_drive_vel
        omega0 = v0 / WHEEL_UNLOADED_RADIUS_M  # rad/s

        joint_qd_np = np.asarray(self.state_0.joint_qd.numpy(), dtype=np.float64)

        # Find the floating base (free) joint dof range and set a forward (world +X) speed.
        joint_qd_start = np.asarray(self.model.joint_qd_start.numpy(), dtype=np.int32)
        joint_q_start = np.asarray(self.model.joint_q_start.numpy(), dtype=np.int32)
        base_dof_start = None
        for jid in range(len(joint_qd_start) - 1):
            dof_start = int(joint_qd_start[jid])
            dof_end = int(joint_qd_start[jid + 1])
            q_start = int(joint_q_start[jid])
            q_end = int(joint_q_start[jid + 1])
            if (dof_end - dof_start) == 6 and (q_end - q_start) == 7:
                base_dof_start = dof_start
                break

        if base_dof_start is not None and (base_dof_start + 2) < joint_qd_np.shape[0]:
            joint_qd_np[base_dof_start + 0] = v0
            joint_qd_np[base_dof_start + 1] = 0.0
            joint_qd_np[base_dof_start + 2] = 0.0
            if (base_dof_start + 5) < joint_qd_np.shape[0]:
                joint_qd_np[base_dof_start + 3 : base_dof_start + 6] = 0.0

        # Initialize all wheel spin joints to match pure rolling at v0.
        body_q = np.asarray(self.state_0.body_q.numpy(), dtype=np.float64)
        chassis_tf = _extract_xform7(body_q[self._chassis_body_id])
        chassis_q = chassis_tf[3:7]
        chassis_forward = _normalize(_quat_rotate_xyzw(chassis_q, np.array([1.0, 0.0, 0.0], dtype=np.float64)))
        chassis_up = _normalize(_quat_rotate_xyzw(chassis_q, np.array([0.0, 0.0, 1.0], dtype=np.float64)))

        for wheel in self.wheels:
            wheel_tf = _extract_xform7(body_q[wheel.body_id])
            wheel_q = wheel_tf[3:7]

            axis_world = _normalize(_quat_rotate_xyzw(wheel_q, wheel.axis_child))
            z = chassis_up
            x = _normalize(np.cross(axis_world, z))
            if np.linalg.norm(x) < 1e-8:
                x = _normalize(chassis_forward - np.dot(chassis_forward, z) * z)
            if np.dot(x, chassis_forward) < 0.0:
                axis_world = -axis_world
                x = -x
            y = _normalize(np.cross(z, x))

            spin_proj = float(np.dot(y, axis_world))
            if abs(spin_proj) < 1e-8:
                joint_qd_np[wheel.dof_id] = omega0
            else:
                joint_qd_np[wheel.dof_id] = omega0 / spin_proj

        wp.copy(
            self.state_0.joint_qd,
            wp.array(joint_qd_np, dtype=self.state_0.joint_qd.dtype, device=self.model.device),
        )
        wp.copy(self.state_1.joint_q, self.state_0.joint_q)
        wp.copy(self.state_1.joint_qd, self.state_0.joint_qd)

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

        if len(self.drive_dofs) > 0:
            self.desired_drive_vel = float(np.mean([joint_qd_np[dof] for dof in self.drive_dofs]))
        else:
            self.desired_drive_vel = float(omega0)

    def _resolve_actuated_dofs_from_names(self, builder: newton.ModelBuilder):
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

    def _init_tire_model(self) -> None:
        body_mass = self.model.body_mass.numpy()
        gravity = np.asarray(self.model.gravity.numpy()[0], dtype=np.float64)
        g_mag = float(np.linalg.norm(gravity))
        mass_total = float(np.sum(body_mass))
        fz_nom = (mass_total * g_mag) / 4.0
        scale = fz_nom / FIALA_REF_FZ_N

        params_report = dict(
            c_slip=1_000_000.0,
            c_alpha=45_836.6236,
            u_max=1.0,
            u_min=0.9,
            rolling_resistance=0.001,
            width=0.235,
        )

        rolling_resistance = params_report["rolling_resistance"] * (WHEEL_UNLOADED_RADIUS_M / FIALA_REF_RADIUS_M)
        width = params_report["width"] * (WHEEL_UNLOADED_RADIUS_M / FIALA_REF_RADIUS_M)
        self.tire_params = FialaTireParams(
            c_slip=params_report["c_slip"] * scale,
            c_alpha=params_report["c_alpha"] * scale,
            u_max=params_report["u_max"],
            u_min=params_report["u_min"],
            rolling_resistance=rolling_resistance,
            width=width,
        )

        self._body_com_np = np.asarray(self.model.body_com.numpy(), dtype=np.float64)

        joint_keys = list(self.model.joint_key)
        lower_to_key = {key.lower(): key for key in joint_keys}
        joint_child = np.asarray(self.model.joint_child.numpy(), dtype=np.int32)
        joint_qd_start = np.asarray(self.model.joint_qd_start.numpy(), dtype=np.int32)
        joint_axis = np.asarray(self.model.joint_axis.numpy(), dtype=np.float64)
        shape_flags = np.asarray(self.model.shape_flags.numpy(), dtype=np.int32)

        wheels: list[WheelInfo] = []
        for joint_name in WHEEL_SPIN_JOINTS:
            key = joint_name if joint_name in joint_keys else lower_to_key.get(joint_name.lower())
            if key is None:
                available = ", ".join(joint_keys)
                raise KeyError(f"Wheel joint '{joint_name}' not found. Available joints: {available}")
            joint_id = joint_keys.index(key)
            dof_start = int(joint_qd_start[joint_id])
            dof_end = int(joint_qd_start[joint_id + 1])
            if dof_end - dof_start != 1:
                raise ValueError(f"Expected 1 DOF for wheel joint '{joint_name}', got {dof_end - dof_start}")

            body_id = int(joint_child[joint_id])
            if body_id not in self.model.body_shapes:
                raise KeyError(f"Wheel joint '{joint_name}' maps to body {body_id}, but no body_shapes entry exists")

            body_shapes = np.asarray(self.model.body_shapes[body_id], dtype=np.int32)
            collide_mask = (shape_flags[body_shapes] & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
            shape_ids = body_shapes[collide_mask]

            wheels.append(
                WheelInfo(
                    joint_name=joint_name,
                    joint_id=joint_id,
                    dof_id=dof_start,
                    body_id=body_id,
                    axis_child=_normalize(joint_axis[dof_start]),
                    shape_ids=shape_ids,
                    radius=WHEEL_UNLOADED_RADIUS_M,
                )
            )

        self.wheels = wheels

        # Map Newton wheel bodies -> MuJoCo bodies (world 0) for xfrc_applied.
        mjc_body_to_newton = np.asarray(self.solver.mjc_body_to_newton.numpy(), dtype=np.int32)[0]
        newton_body_to_mjc: dict[int, int] = {}
        for mjc_body, newton_body in enumerate(mjc_body_to_newton.tolist()):
            if newton_body >= 0:
                newton_body_to_mjc[newton_body] = mjc_body

        wheel_mjc_body_ids: list[int] = []
        for wheel in self.wheels:
            mjc_body_id = int(newton_body_to_mjc.get(wheel.body_id, -1))
            if mjc_body_id < 0:
                raise RuntimeError(f"Failed to map wheel body {wheel.body_id} ('{wheel.joint_name}') to MuJoCo body id")
            wheel_mjc_body_ids.append(mjc_body_id)

        self._wheel_mjc_body_ids_wp = wp.array(wheel_mjc_body_ids, dtype=wp.int32, device=self.model.device)

        # Tire wrench buffers (world frame), applied in mjcb_control.
        self._tire_forces_wp = wp.zeros(len(wheels), dtype=wp.vec3, device=self.model.device)
        self._tire_torques_wp = wp.zeros(len(wheels), dtype=wp.vec3, device=self.model.device)

        body_keys = list(self.model.body_key)
        self._chassis_body_id = 0
        for preferred in ("chassis", "base_link", "chassis_inertia"):
            if preferred in body_keys:
                self._chassis_body_id = body_keys.index(preferred)
                break

    def _aggregate_wheel_contacts(self, wheel: WheelInfo) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        if not hasattr(self.mj_contacts, "n_contacts"):
            return None
        n = int(np.asarray(self.mj_contacts.n_contacts.numpy(), dtype=np.int32)[0])
        if n <= 0:
            return None

        pairs = np.asarray(self.mj_contacts.pair.numpy(), dtype=np.int32)[:n]
        pos = np.asarray(self.mj_contacts.position.numpy(), dtype=np.float64)[:n]
        normal = np.asarray(self.mj_contacts.normal.numpy(), dtype=np.float64)[:n]
        f_n = np.asarray(self.mj_contacts.force.numpy(), dtype=np.float64)[:n]
        sep = None
        if hasattr(self.mj_contacts, "separation"):
            sep = np.asarray(self.mj_contacts.separation.numpy(), dtype=np.float64)[:n]

        wheel_shapes = set(int(s) for s in wheel.shape_ids.tolist())
        fz = 0.0
        p_sum = np.zeros(3, dtype=np.float64)
        n_sum = np.zeros(3, dtype=np.float64)
        depth_sum = 0.0

        for i in range(n):
            s0, s1 = int(pairs[i, 0]), int(pairs[i, 1])
            if s0 not in wheel_shapes and s1 not in wheel_shapes:
                continue
            fi = float(f_n[i])
            if fi <= 0.0:
                continue
            fz += fi
            p_sum += fi * pos[i]
            n_sum += fi * normal[i]
            if sep is not None:
                depth_sum += fi * max(0.0, -float(sep[i]))

        if fz <= self.tire_params.fz_min:
            return None

        p_c = p_sum / fz
        n_c = _normalize(n_sum)
        depth = (depth_sum / fz) if sep is not None else 0.0
        return p_c, n_c, fz, depth

    def _update_tire_forces_substep(self, state: newton.State) -> None:
        """Compute tire tangential forces using *previous* contact normal load, cache for mjcb_control."""
        if not hasattr(self.mj_contacts, "n_contacts"):
            return

        body_q = np.asarray(state.body_q.numpy(), dtype=np.float64)
        body_qd = np.asarray(state.body_qd.numpy(), dtype=np.float64)
        joint_qd = np.asarray(state.joint_qd.numpy(), dtype=np.float64)

        chassis_tf = _extract_xform7(body_q[self._chassis_body_id])
        chassis_q = chassis_tf[3:7]
        chassis_forward = _normalize(_quat_rotate_xyzw(chassis_q, np.array([1.0, 0.0, 0.0])))
        chassis_up = _normalize(_quat_rotate_xyzw(chassis_q, np.array([0.0, 0.0, 1.0])))

        forces = np.zeros((len(self.wheels), 3), dtype=np.float32)
        torques = np.zeros((len(self.wheels), 3), dtype=np.float32)

        for wi, wheel in enumerate(self.wheels):
            agg = self._aggregate_wheel_contacts(wheel)
            if agg is None:
                continue
            p_c, n_c, fz, depth = agg

            wheel_tf = _extract_xform7(body_q[wheel.body_id])
            wheel_p = wheel_tf[:3]
            wheel_q = wheel_tf[3:7]

            com_offset_w = _quat_rotate_xyzw(wheel_q, self._body_com_np[wheel.body_id])
            com_w = wheel_p + com_offset_w

            v_com = _extract_spatial6(body_qd[wheel.body_id])[:3]
            w_w = _extract_spatial6(body_qd[wheel.body_id])[3:]
            v_origin = v_com + np.cross(w_w, wheel_p - com_w)

            axis_world = _normalize(_quat_rotate_xyzw(wheel_q, wheel.axis_child))
            z = n_c
            if np.dot(z, chassis_up) < 0.0:
                z = -z
            x = _normalize(np.cross(axis_world, z))
            if np.linalg.norm(x) < 1e-8:
                x = _normalize(chassis_forward - np.dot(chassis_forward, z) * z)
            if np.dot(x, chassis_forward) < 0.0:
                axis_world = -axis_world
                x = -x
            y = _normalize(np.cross(z, x))

            omega = float(joint_qd[wheel.dof_id]) * float(np.dot(y, axis_world))

            v_x = float(np.dot(v_origin, x))
            v_y = float(np.dot(v_origin, y))

            r_eff = max(0.2 * wheel.radius, wheel.radius - depth)
            f_x, f_y, m_y, m_z = _fiala_forces(self.tire_params, fz=fz, v_x=v_x, v_y=v_y, omega=omega, r_eff=r_eff)

            f_world = (f_x * x) + (f_y * y)
            tau_world = np.cross(p_c - com_w, f_world) + (m_y * y) + (m_z * z)

            forces[wi, :] = f_world.astype(np.float32)
            torques[wi, :] = tau_world.astype(np.float32)

        wp.copy(self._tire_forces_wp, wp.array(forces, dtype=wp.vec3, device=self.model.device))
        wp.copy(self._tire_torques_wp, wp.array(torques, dtype=wp.vec3, device=self.model.device))

    def _tire_mjcb_control(self, m, d) -> None:
        """MuJoCo-Warp control callback: set cached tire wrenches into `d.xfrc_applied`."""
        if len(self.wheels) == 0:
            return
        wp.launch(
            _set_xfrc_applied_kernel,
            dim=len(self.wheels),
            inputs=[self._wheel_mjc_body_ids_wp, self._tire_forces_wp, self._tire_torques_wp, d.xfrc_applied],
            device=self.model.device,
        )

    def _update_keyboard_inputs(self):
        inc = 0.0
        dec = 0.0
        left = 0.0
        right = 0.0

        if hasattr(self.viewer, "is_key_down"):
            inc = 1.0 if self.viewer.is_key_down("i") else 0.0
            dec = 1.0 if self.viewer.is_key_down("k") else 0.0
            left = 1.0 if self.viewer.is_key_down("j") else 0.0
            right = 1.0 if self.viewer.is_key_down("l") else 0.0

        self.desired_drive_vel += (inc - dec) * self.drive_accel * self.frame_dt
        self.desired_drive_vel = float(wp.clamp(self.desired_drive_vel, -self.max_drive_vel, self.max_drive_vel))
        self.steer_cmd = float(wp.clamp(left - right, -1.0, 1.0))

    def _apply_control_substep(self):
        if self.control.joint_f is not None:
            self.control.joint_f.zero_()
        if self.control.joint_target_pos is not None:
            self.control.joint_target_pos.zero_()
        if self.control.joint_target_vel is not None:
            self.control.joint_target_vel.zero_()

        if self.control.joint_target_pos is None or self.control.joint_target_vel is None:
            return

        for i, dof in enumerate(self.drive_dofs):
            self.drive_target_pos[i] += self.desired_drive_vel * self.sim_dt

        target_pos_np = self.control.joint_target_pos.numpy()
        target_vel_np = self.control.joint_target_vel.numpy()

        for i, dof in enumerate(self.drive_dofs):
            target_pos_np[dof] = self.drive_target_pos[i]
            target_vel_np[dof] = self.desired_drive_vel

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

            # Viewer may apply picking/wind forces into state.body_f; these will be
            # mapped into MuJoCo before the callback runs.
            self.viewer.apply_forces(self.state_0)

            # Compute tire forces using previous-step normal loads and cache them
            # for application in `mjcb_control`.
            self._update_tire_forces_substep(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            populate_contacts(self.mj_contacts, self.solver)

    def step(self):
        self._update_keyboard_inputs()
        self._simulate_once()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--init-pos-speed",
        action="store_true",
        help="Initialize the chassis at a forward speed and wheel spin for no-slip rolling.",
    )
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
