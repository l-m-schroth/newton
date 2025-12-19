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
# low-level PD controller (joint_target_ke/kd), typically ke = 0.
# Steering is also controlled with target POSITION and PD.
#
# Command: python -m newton.examples robot_racecar
###########################################################################

from __future__ import annotations

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

        # Disable MuJoCo contact constraints (mjDSBL_CONTACT) so that no normal/tangential
        # contact forces are generated by the solver. Collision detection still runs so
        # we can read contact points/normals inside the control callback.
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=200,
            nconmax=200,
            integrator="rk4",
            disable_contacts=True,
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

        # Install the tire-force callback (MuJoCo-C: mjcb_control). This callback is
        # invoked during forward dynamics and at each RK4 stage, so forces applied via
        # xfrc_applied are integrated consistently.
        import mujoco_warp

        mujoco_warp.mjcb_control = self._tire_mjcb_control

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

        # Normal-force spring/damper defaults (per-wheel). Choose stiffness from a target
        # static deflection (fraction of unloaded radius), and critical damping.
        static_deflection = max(1e-4, 0.04 * WHEEL_UNLOADED_RADIUS_M)  # ~2mm for 5cm wheel
        self.normal_k = fz_nom / static_deflection *2   # L: added *10 for a quick trial
        m_per_wheel = mass_total / 4.0
        self.normal_d = 2.0 * math.sqrt(max(self.normal_k * m_per_wheel, 0.0)) * 2

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
                f"[tire-debug] normal_k={self.normal_k:.3g} normal_d={self.normal_d:.3g} "
                f"sim_dt={self.sim_dt:.3g} sim_substeps={self.sim_substeps}",
                flush=True,
            )

    def _tire_mjcb_control(self, m, d) -> None:
        """MuJoCo-Warp control callback: compute and apply tire forces via d.xfrc_applied."""
        self._debug_cb_calls += 1
        ncon = int(np.asarray(d.nacon.numpy(), dtype=np.int32)[0])
        worldid = 0

        forces = np.zeros((len(self.wheels), 3), dtype=np.float32)
        torques = np.zeros((len(self.wheels), 3), dtype=np.float32)

        if getattr(self, "debug_tire", False):
            do_print = (
                self._debug_cb_calls <= self._debug_print_first
                or (self._debug_cb_calls % self._debug_print_every) == 0
                or ncon == 0
            )
            if do_print:
                try:
                    time0 = float(np.asarray(d.time.numpy(), dtype=np.float64)[worldid])
                except Exception:
                    time0 = float("nan")
                print(
                    f"[tire-debug] cb_call={self._debug_cb_calls} time={time0:.3f} ncon={ncon} "
                    f"disableflags={getattr(m.opt, 'disableflags', None)} "
                    f"run_collision={getattr(m.opt, 'run_collision_detection', None)} "
                    f"integrator={getattr(m.opt, 'integrator', None)}",
                    flush=True,
                )

        # If there are no contacts anywhere in the world, make sure we still clear the
        # previous RK stage's tire wrenches (MuJoCo does not auto-clear xfrc_applied).
        if ncon <= 0:
            wp.copy(self._tire_forces_wp, wp.array(forces, dtype=wp.vec3, device=self.model.device))
            wp.copy(self._tire_torques_wp, wp.array(torques, dtype=wp.vec3, device=self.model.device))
            wp.launch(
                _set_xfrc_applied_kernel,
                dim=len(self.wheels),
                inputs=[self._wheel_mjc_body_ids_wp, self._tire_forces_wp, self._tire_torques_wp, d.xfrc_applied],
                device=self.model.device,
            )
            return

        # Fetch state needed for all wheels (small model, OK to pull to host for now).
        xipos = np.asarray(d.xipos.numpy(), dtype=np.float64)[worldid]  # (nbody,3)
        xmat = np.asarray(d.xmat.numpy(), dtype=np.float64)[worldid]    # (nbody,3,3)
        cvel = np.asarray(d.cvel.numpy(), dtype=np.float64)[worldid]    # (nbody,6) rot:lin
        if getattr(self, "debug_tire", False):
            qpos0 = np.asarray(d.qpos.numpy(), dtype=np.float64)[worldid]
            qvel0 = np.asarray(d.qvel.numpy(), dtype=np.float64)[worldid]
            if not (np.isfinite(qpos0).all() and np.isfinite(qvel0).all()):
                print(
                    f"[tire-debug][WARN] non-finite state: "
                    f"finite_qpos={bool(np.isfinite(qpos0).all())} finite_qvel={bool(np.isfinite(qvel0).all())} "
                    f"max|qpos|={float(np.nanmax(np.abs(qpos0))):.3g} max|qvel|={float(np.nanmax(np.abs(qvel0))):.3g}",
                    flush=True,
                )

        chassis_R = xmat[self._chassis_mjc_body_id]
        chassis_forward = _normalize(chassis_R @ np.array([1.0, 0.0, 0.0], dtype=np.float64))

        # Contacts (global arrays across worlds)
        contact_world = np.asarray(d.contact.worldid.numpy(), dtype=np.int32)[:ncon]
        contact_geom = np.asarray(d.contact.geom.numpy(), dtype=np.int32)[:ncon]
        contact_pos = np.asarray(d.contact.pos.numpy(), dtype=np.float64)[:ncon]
        contact_dist = np.asarray(d.contact.dist.numpy(), dtype=np.float64)[:ncon]
        contact_frame = np.asarray(d.contact.frame.numpy(), dtype=np.float64)[:ncon]  # columns: n,t1,t2
        contact_type = None
        if hasattr(d.contact, "type"):
            contact_type = np.asarray(d.contact.type.numpy(), dtype=np.int32)[:ncon]

        if getattr(self, "debug_tire", False) and (
            self._debug_cb_calls <= self._debug_print_first or (self._debug_cb_calls % self._debug_print_every) == 0
        ):
            for ci in range(min(ncon, 6)):
                g0, g1 = int(contact_geom[ci, 0]), int(contact_geom[ci, 1])
                dist = float(contact_dist[ci])
                ctype = int(contact_type[ci]) if contact_type is not None else -1
                print(f"[tire-debug]  contact[{ci}] geoms=({g0},{g1}) dist={dist:.4g} type={ctype}", flush=True)

        # Gather candidate contacts per wheel (by Mujoco geom id)
        wheel_contacts: list[list[tuple[float, np.ndarray, np.ndarray, int, int]]] = [[] for _ in self.wheels]
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

            # Robust normal extraction: depending on convention the normal may be stored
            # as a row or a column. Try both and pick the candidate that best matches
            # the expected direction (from other body toward the wheel body).
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

            wheel_contacts[wi].append((delta, p, n_on_wheel, other_body, other_geom))

        for wi, wheel in enumerate(self.wheels):
            contacts = wheel_contacts[wi]
            if not contacts:
                continue

            # Use up to the two deepest contacts (racecar wheel manifold)
            contacts.sort(key=lambda x: x[0], reverse=True)
            contacts = contacts[:2]

            deltas = np.array([c[0] for c in contacts], dtype=np.float64)
            positions = np.stack([c[1] for c in contacts], axis=0)
            normals = np.stack([c[2] for c in contacts], axis=0)

            weights = np.maximum(deltas, 1e-6)
            p_mid = np.average(positions, axis=0, weights=weights)
            n_mid = _normalize(np.average(normals, axis=0, weights=weights))
            delta = float(np.average(deltas, weights=weights))
            other_body = int(contacts[0][3])

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

            # Penetration rate: positive when compressing. Only damp compression to avoid
            # turning off the spring due to a negative damper term during rebound.
            delta_dot = -float(np.dot(v_rel, n_mid))
            delta_dot = max(0.0, delta_dot)
            f_n = self.normal_k * delta + self.normal_d * delta_dot
            f_n = max(0.0, f_n)

            # Wheel frame (x: rolling, y: lateral in road plane, n: normal into wheel)
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
                self.tire_params, fz=f_n, v_x=v_x, v_y=v_y, omega=omega, r_eff=r_eff
            )

            f_world = (f_n * z) + (f_x * x) + (f_y * y)
            tau_world = np.cross(p_mid - com_w, f_world) + (m_roll * axis_world) + (m_z * z)

            if getattr(self, "debug_tire", False) and (
                self._debug_cb_calls <= self._debug_print_first or (self._debug_cb_calls % self._debug_warn_every) == 0
            ):
                n_up = float(np.dot(n_mid, self._up_dir))
                if n_up < 0.0:
                    print(
                        f"[tire-debug][WARN] wheel={wheel.joint_name} downward normal: dot(n,up)={n_up:.3f} n={n_mid}"
                        ,
                        flush=True,
                    )
                if delta > 0.5 * wheel.radius:
                    print(
                        f"[tire-debug][WARN] wheel={wheel.joint_name} huge penetration: delta={delta:.4g} "
                        f"radius={wheel.radius:.4g}"
                        ,
                        flush=True,
                    )
                if not (np.isfinite(f_world).all() and np.isfinite(tau_world).all()):
                    print(
                        f"[tire-debug][WARN] wheel={wheel.joint_name} non-finite wrench: F={f_world} tau={tau_world}",
                        flush=True,
                    )
                print(
                    f"[tire-debug]  wheel={wheel.joint_name} delta={delta:.4g} delta_dot={delta_dot:.4g} "
                    f"fN={f_n:.4g} dot(n,up)={n_up:.3f} v_rel_xy=({v_x:.3g},{v_y:.3g}) omega={omega:.3g} "
                    f"Fx={f_x:.3g} Fy={f_y:.3g} M_roll={m_roll:.3g} Mz={m_z:.3g}",
                    flush=True,
                )

            forces[wi, :] = f_world.astype(np.float32)
            torques[wi, :] = tau_world.astype(np.float32)

        if getattr(self, "debug_tire", False):
            if not (np.isfinite(forces).all() and np.isfinite(torques).all()):
                print(
                    "[tire-debug][WARN] non-finite forces/torques arrays; zeroing tire forces for this callback",
                    flush=True,
                )
                forces[:, :] = 0.0
                torques[:, :] = 0.0

        # Push forces to device and write them into xfrc_applied for this RK stage.
        wp.copy(self._tire_forces_wp, wp.array(forces, dtype=wp.vec3, device=self.model.device))
        wp.copy(self._tire_torques_wp, wp.array(torques, dtype=wp.vec3, device=self.model.device))
        wp.launch(
            _set_xfrc_applied_kernel,
            dim=len(self.wheels),
            inputs=[self._wheel_mjc_body_ids_wp, self._tire_forces_wp, self._tire_torques_wp, d.xfrc_applied],
            device=self.model.device,
        )

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
