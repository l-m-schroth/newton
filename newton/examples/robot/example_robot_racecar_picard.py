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
# Example Robot Racecar (picard)
#
# Loads a URDF racecar, drops it on a flat plane, and allows simple
# keyboard control:
#   I / K : increase / decrease desired drive velocity (wheel spin)
#   J / L : steering left / right
#
# Wheels are controlled with target POSITION (integrated wheel angle) and a
# low-level PD controller (joint_target_ke/kd), typically ke = 0.
# Steering is also controlled with target POSITION and PD.
#
# This example uses MuJoCo contact constraints for NORMAL forces (frictionless,
# condim=1), and applies tire tangential forces + moments via callbacks:
# - `mjcb_control`: applies the current tire-force guess (warmstart)
# - `mjcb_postsolve`: updates tire forces based on solved normal load
# The MuJoCo forward pass runs multiple Picard iterations per step.
#
# Command: python -m newton.examples robot_racecar_picard
###########################################################################

from __future__ import annotations

import atexit
from dataclasses import dataclass
import math
import os

import numpy as np
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
    mjc_body_id: int
    axis_child: np.ndarray  # (3,), unit axis in child frame
    shape_ids: np.ndarray  # (k,), collision shapes belonging to this wheel body
    mjc_geom_ids: np.ndarray  # (k,), Mujoco geoms belonging to this wheel (world 0)
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
    """ function that rotates a 3D vector around a unit quaternion, for instance mentioned here:
    https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication"""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q_xyz = q[:3]
    q_w = float(q[3])
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _fiala_forces(
    params: FialaTireParams, *, fz: float, v_x: float, v_y: float, omega: float, r_eff: float
) -> tuple[float, float, float, float]:
    """Compute longitudinal/lateral forces, self-aligning moment and rolling resistance (TR-2015-13)."""
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
    s_crit = (u * fz) / (2.0 * params.c_slip) 
    if abs(ss) <= s_crit:
        f_x = params.c_slip * ss
    else:
        fx1 = u * fz
        fx2 = (u * fz) ** 2 / (4.0 * max(abs(ss), 1e-6) * params.c_slip)
        f_x = _sign(ss) * (fx1 - fx2)

    # Lateral force (report eqs. 25-27)
    alpha_crit = math.atan((3.0 * u * fz) / params.c_alpha) 
    if abs(alpha) <= alpha_crit:
        h = 1.0 - (params.c_alpha * abs(tan_alpha)) / (3.0 * u * fz)
        h = float(np.clip(h, 0.0, 1.0))
        f_y = -u * fz * (1.0 - h**3) * _sign(alpha)
        m_z = u * fz * params.width * (1.0 - h) * (h**3) * _sign(alpha)
    else:
        f_y = -u * fz * _sign(alpha)
        m_z = 0.0

    # Rolling resistance (report eq. 28)
    m_roll = -params.rolling_resistance * fz * _sign(omega)

    return f_x, f_y, m_roll, m_z


@wp.kernel
def _set_xfrc_applied_kernel(
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    """Set Cartesian wrenches in mujoco_warp Data.xfrc_applied (world 0)."""
    tid = wp.tid()
    body = body_ids[tid]
    if body < 0:
        return
    xfrc_applied[0, body] = wp.spatial_vector(forces[tid], torques[tid])

# -------- # 
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
            mu=0.0,       # friction coefficient (tangential handled by tire model)
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

        # Use MuJoCo contact constraints for NORMAL forces; tangential friction is
        # disabled via `condim=1` and `friction=0` on wheel/ground geoms.
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=200,
            nconmax=200,
            integrator="rk4",
            disable_contacts=False,
            use_mujoco_contacts=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        # Tire-model debug printing (very verbose).
        # Default: enabled. Disable with NEWTON_TIRE_DEBUG=0.
        self.debug_tire = os.getenv("NEWTON_TIRE_DEBUG", "1").strip() not in ("", "0", "false", "False")
        self._debug_cb_calls = 0
        self._debug_print_first = 80
        self._debug_print_every = 80
        self._debug_warn_every = 10
        g = np.asarray(self.model.gravity.numpy()[0], dtype=np.float64)
        g_mag = float(np.linalg.norm(g))
        self._up_dir = (-g / g_mag) if g_mag > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Picard settings (MJWarp extension, see mujoco_warp.Model.opt).
        self.picard_iterations = int(os.getenv("NEWTON_PICARD_ITERS", "1"))
        self.picard_beta = float(os.getenv("NEWTON_PICARD_BETA", "1.0"))
        self.solver.mjw_model.opt.picard_iterations = max(1, int(self.picard_iterations))
        self.solver.mjw_model.opt.picard_beta = float(np.clip(self.picard_beta, 0.0, 1.0))

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

        self._init_tire_model()
        self._configure_mujoco_tire_contact()
        self._init_picard_logging()

        # Install callbacks:
        # - `mjcb_control` applies the current tire wrench guess (warmstart)
        # - `mjcb_postsolve` updates the tire wrench guess from solved normal loads
        # The Picard loop is executed inside mujoco_warp forward dynamics.
        import mujoco_warp

        mujoco_warp.mjcb_control = self._tire_mjcb_control
        mujoco_warp.mjcb_postsolve = self._tire_mjcb_postsolve

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

    def _init_tire_model(self) -> None:
        body_mass = self.model.body_mass.numpy()
        gravity = np.asarray(self.model.gravity.numpy()[0], dtype=np.float64)
        g_mag = float(np.linalg.norm(gravity))
        mass_total = float(np.sum(body_mass))
        fz_nom = (mass_total * g_mag) / 4.0
        scale = fz_nom / FIALA_REF_FZ_N

        params_report = dict(
            c_slip=3*1_000_000.0, # just added 3* for quick trial
            c_alpha=3*45_836.6236, # just added 3* for quick trial
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

            mjc_body_id = -1
            mjc_geom_ids = np.empty((0,), dtype=np.int32)

            wheels.append(
                WheelInfo(
                    joint_name=joint_name,
                    joint_id=joint_id,
                    dof_id=dof_start,
                    body_id=body_id,
                    mjc_body_id=mjc_body_id,
                    axis_child=_normalize(joint_axis[dof_start]),
                    shape_ids=shape_ids,
                    mjc_geom_ids=mjc_geom_ids,
                    radius=WHEEL_UNLOADED_RADIUS_M,
                )
            )

        # Newton -> MuJoCo body/geom mapping (world 0), used to identify wheel bodies and geoms
        mjc_body_to_newton = np.asarray(self.solver.mjc_body_to_newton.numpy(), dtype=np.int32)[0]
        mjc_geom_to_newton_shape = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy(), dtype=np.int32)[0]

        newton_body_to_mjc: dict[int, int] = {}
        for mjc_body, newton_body in enumerate(mjc_body_to_newton.tolist()):
            if newton_body >= 0:
                newton_body_to_mjc[newton_body] = mjc_body

        newton_shape_to_mjc_geoms: dict[int, list[int]] = {}
        for mjc_geom, newton_shape in enumerate(mjc_geom_to_newton_shape.tolist()):
            if newton_shape >= 0:
                newton_shape_to_mjc_geoms.setdefault(newton_shape, []).append(mjc_geom)

        wheels_mapped: list[WheelInfo] = []
        geom_to_wheel: dict[int, int] = {}
        for wi, wheel in enumerate(wheels):
            mjc_body_id = int(newton_body_to_mjc.get(wheel.body_id, -1))
            if mjc_body_id < 0:
                raise RuntimeError(f"Failed to map wheel body {wheel.body_id} ('{wheel.joint_name}') to MuJoCo body id")

            geom_ids: list[int] = []
            for sid in wheel.shape_ids.tolist():
                geom_ids.extend(newton_shape_to_mjc_geoms.get(int(sid), []))
            mjc_geom_ids = np.asarray(sorted(set(geom_ids)), dtype=np.int32)
            for gid in mjc_geom_ids.tolist():
                geom_to_wheel[gid] = wi

            wheels_mapped.append(
                WheelInfo(
                    joint_name=wheel.joint_name,
                    joint_id=wheel.joint_id,
                    dof_id=wheel.dof_id,
                    body_id=wheel.body_id,
                    mjc_body_id=mjc_body_id,
                    axis_child=wheel.axis_child,
                    shape_ids=wheel.shape_ids,
                    mjc_geom_ids=mjc_geom_ids,
                    radius=wheel.radius,
                )
            )

        self.wheels = wheels_mapped
        self._mjc_geom_to_wheel = geom_to_wheel
        self._wheel_mjc_body_ids_wp = wp.array(
            [w.mjc_body_id for w in self.wheels], dtype=wp.int32, device=self.model.device
        )
        self._tire_forces_wp = wp.zeros(len(wheels), dtype=wp.vec3, device=self.model.device)
        self._tire_torques_wp = wp.zeros(len(wheels), dtype=wp.vec3, device=self.model.device)

        body_keys = list(self.model.body_key)
        self._chassis_body_id = 0
        for preferred in ("chassis", "base_link", "chassis_inertia"):
            if preferred in body_keys:
                self._chassis_body_id = body_keys.index(preferred)
                break

        # Per-geom body ids (static across simulation) for contact processing in the callback
        self._mjc_geom_bodyid_np = np.asarray(self.solver.mjw_model.geom_bodyid.numpy(), dtype=np.int32)

        # Chassis MuJoCo body id (world 0) for forward-direction reference
        self._chassis_mjc_body_id = int(newton_body_to_mjc.get(self._chassis_body_id, 0))

        if getattr(self, "debug_tire", False):
            print("[tire-debug] enabled (set NEWTON_TIRE_DEBUG=0 to disable)", flush=True)
            print("[tire-debug] wheels mapping:", flush=True)
            for w in self.wheels:
                print(
                    f"  joint={w.joint_name} newton_body={w.body_id} mjc_body={w.mjc_body_id} "
                    f"mjc_geoms={w.mjc_geom_ids.tolist()}",
                    flush=True,
                )
            print(
                f"[tire-debug] sim_dt={self.sim_dt:.3g} sim_substeps={self.sim_substeps} "
                f"picard_iterations={self.picard_iterations} picard_beta={self.picard_beta:.3g}",
                flush=True,
            )

    def _configure_mujoco_tire_contact(self) -> None:
        """Configure MuJoCo contact to be normal-only (condim=1, friction=0) for wheel/ground."""
        wheel_geoms: set[int] = set()
        for w in self.wheels:
            wheel_geoms.update(int(g) for g in w.mjc_geom_ids.tolist())

        ground_geoms = np.nonzero(self._mjc_geom_bodyid_np == 0)[0].tolist()
        target_geoms = sorted(set(ground_geoms).union(wheel_geoms))
        if not target_geoms:
            return

        condim = np.asarray(self.solver.mjw_model.geom_condim.numpy(), dtype=np.int32)
        condim[target_geoms] = 1
        self.solver.mjw_model.geom_condim = wp.array(condim, dtype=wp.int32, device=self.model.device)

        friction = np.asarray(self.solver.mjw_model.geom_friction.numpy(), dtype=np.float32)
        if friction.ndim == 3:
            friction[0, target_geoms, :] = 0.0
        elif friction.ndim == 2:
            friction[target_geoms, :] = 0.0
        else:
            raise ValueError(f"Unexpected geom_friction shape: {friction.shape}")
        self.solver.mjw_model.geom_friction = wp.array(friction, dtype=wp.vec3, device=self.model.device)

    def _init_picard_logging(self) -> None:
        self.picard_log_enabled = os.getenv("NEWTON_PICARD_LOG", "1").strip() not in ("", "0", "false", "False")
        self.picard_log_path = os.getenv("NEWTON_PICARD_LOG_PATH", "racecar_picard_log.npz").strip()
        self.picard_log_maxlen = int(os.getenv("NEWTON_PICARD_LOG_MAX", "20000"))

        self._picard_log_last_time: float | None = None
        self._picard_log_iter = 0
        self._picard_log_times: list[float] = []
        self._picard_log_iters: list[int] = []
        self._picard_log_forces: list[np.ndarray] = []
        self._picard_log_torques: list[np.ndarray] = []

        if self.picard_log_enabled:
            atexit.register(self._save_picard_log)

    def _save_picard_log(self) -> None:
        if not getattr(self, "picard_log_enabled", False):
            return

        if not getattr(self, "_picard_log_times", None):
            return

        path = getattr(self, "picard_log_path", "").strip()
        if not path:
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        times = np.asarray(self._picard_log_times, dtype=np.float64)
        iters = np.asarray(self._picard_log_iters, dtype=np.int32)
        forces = np.stack(self._picard_log_forces, axis=0).astype(np.float32)
        torques = np.stack(self._picard_log_torques, axis=0).astype(np.float32)
        wheel_names = np.asarray([w.joint_name for w in self.wheels], dtype=np.str_)

        np.savez(
            path,
            times=times,
            iters=iters,
            forces=forces,
            torques=torques,
            wheel_names=wheel_names,
            picard_iterations=int(self.picard_iterations),
            picard_beta=float(self.picard_beta),
        )
        print(f"[picard-log] saved {len(times)} entries to {path}", flush=True)

    def _record_picard_iteration(self, d, forces: np.ndarray, torques: np.ndarray) -> None:
        if not getattr(self, "picard_log_enabled", False):
            return

        if len(self._picard_log_times) >= self.picard_log_maxlen:
            return

        worldid = 0
        time0 = float(np.asarray(d.time.numpy(), dtype=np.float64)[worldid])
        if self._picard_log_last_time is None or abs(time0 - self._picard_log_last_time) > 1e-12:
            self._picard_log_last_time = time0
            self._picard_log_iter = 0
        else:
            self._picard_log_iter += 1

        self._picard_log_times.append(time0)
        self._picard_log_iters.append(int(self._picard_log_iter))
        self._picard_log_forces.append(np.asarray(forces, dtype=np.float32).copy())
        self._picard_log_torques.append(np.asarray(torques, dtype=np.float32).copy())

    def _apply_tire_wrenches(self, d, forces: np.ndarray, torques: np.ndarray) -> None:
        wp.copy(self._tire_forces_wp, wp.array(forces, dtype=wp.vec3, device=self.model.device))
        wp.copy(self._tire_torques_wp, wp.array(torques, dtype=wp.vec3, device=self.model.device))
        wp.launch(
            _set_xfrc_applied_kernel,
            dim=len(self.wheels),
            inputs=[self._wheel_mjc_body_ids_wp, self._tire_forces_wp, self._tire_torques_wp, d.xfrc_applied],
            device=self.model.device,
        )

    def _tire_mjcb_control(self, m, d) -> None:
        """MuJoCo-Warp control callback: apply warmstart tire forces/moments via `d.xfrc_applied`."""
        self._debug_cb_calls += 1
        if len(self.wheels) == 0:
            return

        wp.launch(
            _set_xfrc_applied_kernel,
            dim=len(self.wheels),
            inputs=[self._wheel_mjc_body_ids_wp, self._tire_forces_wp, self._tire_torques_wp, d.xfrc_applied],
            device=self.model.device,
        )

    def _tire_mjcb_postsolve(self, m, d) -> None:
        """MuJoCo-Warp post-solve callback: update tire forces/moments from solved normal loads."""
        worldid = 0
        ncon = int(np.asarray(d.nacon.numpy(), dtype=np.int32)[0])

        forces = np.zeros((len(self.wheels), 3), dtype=np.float32)
        torques = np.zeros((len(self.wheels), 3), dtype=np.float32)

        if len(self.wheels) == 0:
            return

        if ncon > 0:
            xipos = np.asarray(d.xipos.numpy(), dtype=np.float64)[worldid]  # (nbody,3)
            xmat = np.asarray(d.xmat.numpy(), dtype=np.float64)[worldid]    # (nbody,3,3)
            cvel = np.asarray(d.cvel.numpy(), dtype=np.float64)[worldid]    # (nbody,6) rot:lin

            chassis_R = xmat[self._chassis_mjc_body_id]
            chassis_forward = _normalize(chassis_R @ np.array([1.0, 0.0, 0.0], dtype=np.float64))

            contact_world = np.asarray(d.contact.worldid.numpy(), dtype=np.int32)[:ncon]
            contact_geom = np.asarray(d.contact.geom.numpy(), dtype=np.int32)[:ncon]
            contact_pos = np.asarray(d.contact.pos.numpy(), dtype=np.float64)[:ncon]
            contact_dist = np.asarray(d.contact.dist.numpy(), dtype=np.float64)[:ncon]
            contact_frame = np.asarray(d.contact.frame.numpy(), dtype=np.float64)[:ncon]  # columns: n,t1,t2
            contact_efc_address = np.asarray(d.contact.efc_address.numpy(), dtype=np.int32)[:ncon]

            efc_force = np.asarray(d.efc.force.numpy(), dtype=np.float64)[worldid]  # (njmax,)

            wheel_contacts: list[list[tuple[float, float, np.ndarray, np.ndarray, int]]] = [[] for _ in self.wheels]

            for ci in range(ncon):
                if int(contact_world[ci]) != worldid:
                    continue

                g0 = int(contact_geom[ci, 0])
                g1 = int(contact_geom[ci, 1])
                wi = self._mjc_geom_to_wheel.get(g0)
                wheel_is_geom1 = False
                if wi is None:
                    wi = self._mjc_geom_to_wheel.get(g1)
                    wheel_is_geom1 = True
                if wi is None:
                    continue

                delta = max(0.0, -float(contact_dist[ci]))
                if delta <= 0.0:
                    continue

                adr_raw = contact_efc_address[ci, 0] if contact_efc_address.ndim == 2 else contact_efc_address[ci]
                adr = int(adr_raw)
                if adr < 0:
                    continue

                f_n = float(efc_force[adr])
                if not np.isfinite(f_n) or f_n <= 0.0:
                    continue

                p = contact_pos[ci]

                other_geom = g0 if wheel_is_geom1 else g1
                other_body = int(self._mjc_geom_bodyid_np[other_geom])
                wheel_body = self.wheels[wi].mjc_body_id
                com_w = xipos[wheel_body]
                if other_body == 0:
                    expected_dir = self._up_dir
                else:
                    expected_dir = com_w - xipos[other_body]
                    expected_dir = _normalize(expected_dir) if float(np.linalg.norm(expected_dir)) > 1e-9 else self._up_dir

                frame = contact_frame[ci]
                n_cand0 = np.asarray(frame.T[0], dtype=np.float64)
                n_cand1 = np.asarray(frame[0], dtype=np.float64)

                def _orient_and_score(n: np.ndarray) -> tuple[np.ndarray, float]:
                    n = _normalize(n)
                    s = float(np.dot(n, expected_dir))
                    if s < 0.0:
                        n = -n
                        s = -s
                    return n, s

                n0, s0 = _orient_and_score(n_cand0)
                n1, s1 = _orient_and_score(n_cand1)
                n_on_wheel = n0 if s0 >= s1 else n1

                wheel_contacts[wi].append((f_n, delta, p, n_on_wheel, other_body))

            for wi, wheel in enumerate(self.wheels):
                contacts = wheel_contacts[wi]
                if not contacts:
                    continue

                # Use up to the two strongest contacts (racecar wheel manifold).
                contacts.sort(key=lambda x: (x[0], x[1]), reverse=True)
                contacts = contacts[:2]

                fns = np.array([c[0] for c in contacts], dtype=np.float64)
                deltas = np.array([c[1] for c in contacts], dtype=np.float64)
                positions = np.stack([c[2] for c in contacts], axis=0)
                normals = np.stack([c[3] for c in contacts], axis=0)

                weights = np.maximum(fns, 1e-6)
                p_mid = np.average(positions, axis=0, weights=weights)
                n_mid = _normalize(np.average(normals, axis=0, weights=weights))
                delta = float(np.average(deltas, weights=weights))
                f_n_total = float(np.sum(fns))
                other_body = int(contacts[0][4])

                # Wheel kinematics at midpoint
                wheel_body = wheel.mjc_body_id
                com_w = xipos[wheel_body]
                w_w = cvel[wheel_body, 0:3]
                v_com_w = cvel[wheel_body, 3:6]
                v_mid_wheel = v_com_w + np.cross(w_w, p_mid - com_w)

                if other_body == 0:
                    v_mid_other = np.zeros(3, dtype=np.float64)
                else:
                    com_o = xipos[other_body]
                    w_o = cvel[other_body, 0:3]
                    v_com_o = cvel[other_body, 3:6]
                    v_mid_other = v_com_o + np.cross(w_o, p_mid - com_o)

                v_rel = v_mid_wheel - v_mid_other

                # Ensure the aggregated normal points from the other body toward the wheel.
                if other_body == 0:
                    if float(np.dot(n_mid, self._up_dir)) < 0.0:
                        n_mid = -n_mid
                else:
                    expected_dir = _normalize(com_w - xipos[other_body])
                    if float(np.dot(n_mid, expected_dir)) < 0.0:
                        n_mid = -n_mid

                # Wheel frame (x: rolling, y: lateral in road plane, z: normal into wheel)
                R_wheel = xmat[wheel_body]
                axis_world = _normalize(R_wheel @ wheel.axis_child)
                z = n_mid
                x = _normalize(np.cross(axis_world, z))
                if np.linalg.norm(x) < 1e-8:
                    x = _normalize(chassis_forward - np.dot(chassis_forward, z) * z)
                if np.dot(x, chassis_forward) < 0.0:
                    axis_world = -axis_world
                    x = -x
                y = _normalize(np.cross(z, x))

                omega = float(np.dot(w_w, axis_world))
                v_x = float(np.dot(v_rel, x))
                v_y = float(np.dot(v_rel, y))

                r_eff = max(0.2 * wheel.radius, wheel.radius - delta)
                f_x, f_y, m_roll, m_z = _fiala_forces(
                    self.tire_params, fz=f_n_total, v_x=v_x, v_y=v_y, omega=omega, r_eff=r_eff
                )

                # IMPORTANT: do NOT apply the normal force here (MuJoCo solves it).
                f_world = (f_x * x) + (f_y * y)
                tau_world = np.cross(p_mid - com_w, f_world) + (m_roll * axis_world) + (m_z * z)

                forces[wi, :] = f_world.astype(np.float32)
                torques[wi, :] = tau_world.astype(np.float32)

        if getattr(self, "debug_tire", False):
            if not (np.isfinite(forces).all() and np.isfinite(torques).all()):
                print("[tire-debug][WARN] non-finite tire forces/torques; zeroing", flush=True)
                forces[:, :] = 0.0
                torques[:, :] = 0.0

        self._record_picard_iteration(d, forces, torques)
        self._apply_tire_wrenches(d, forces, torques)

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

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Update commands once per frame
        self._update_keyboard_inputs()

        self._simulate_once()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
