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

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence

import warp as wp

from .disc_terrain_collision import ChFunctionInterp, CollisionType, ConstructAreaDepthTable, _mujoco_disc_terrain_collision_kernel
from .fiala_tire import FialaTire, fiala_tire_advance_kernel


@wp.kernel
def _mujoco_tire_contact_kinematics_kernel(
    xmat: wp.array2d(dtype=wp.mat33),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    wheel_body_ids: wp.array(dtype=int),
    nwheels: int,
    contact_x: wp.array(dtype=wp.vec3),
    contact_y: wp.array(dtype=wp.vec3),
    contact_z: wp.array(dtype=wp.vec3),
    in_contact: wp.array(dtype=int),
    vx_out: wp.array(dtype=float),
    vy_out: wp.array(dtype=float),
    velz_out: wp.array(dtype=float),
    omega_out: wp.array(dtype=float),
):
    tid = wp.tid()

    worldid = tid // nwheels
    wheel_idx = tid - worldid * nwheels
    bodyid = wheel_body_ids[wheel_idx]

    if in_contact[tid] == 0:
        vx_out[tid] = 0.0
        vy_out[tid] = 0.0
        velz_out[tid] = 0.0
        omega_out[tid] = 0.0
        return

    R = xmat[worldid, bodyid]
    disc_normal = wp.normalize(wp.vec3(R[0, 1], R[1, 1], R[2, 1]))  # wheel axis (Chrono: Y)

    w = cvel[worldid, bodyid]  # (ang:lin) in world
    omega_world = wp.vec3(w[0], w[1], w[2])
    vel_world = wp.vec3(w[3], w[4], w[5])

    cx = contact_x[tid]
    cy = contact_y[tid]
    cz = contact_z[tid]

    # Wheel velocity in ISO-C contact frame.
    vx_out[tid] = wp.dot(vel_world, cx)
    vy_out[tid] = wp.dot(vel_world, cy)
    velz_out[tid] = wp.dot(vel_world, cz)

    # Wheel spin about disc normal (Chrono wheel_state.omega).
    omega_out[tid] = wp.dot(omega_world, disc_normal)


@wp.kernel
def _mujoco_apply_tire_wrenches_kernel(
    xipos: wp.array2d(dtype=wp.vec3),
    wheel_body_ids: wp.array(dtype=int),
    nwheels: int,
    in_contact: wp.array(dtype=int),
    depth: wp.array(dtype=float),
    contact_pos: wp.array(dtype=wp.vec3),
    contact_x: wp.array(dtype=wp.vec3),
    contact_y: wp.array(dtype=wp.vec3),
    contact_z: wp.array(dtype=wp.vec3),
    fx: wp.array(dtype=float),
    fy: wp.array(dtype=float),
    fz: wp.array(dtype=float),
    my: wp.array(dtype=float),
    mz: wp.array(dtype=float),
    xfrc_applied_out: wp.array2d(dtype=wp.spatial_vector),
):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChForceElementTire.cpp::GetTireForce
    tid = wp.tid()

    worldid = tid // nwheels
    wheel_idx = tid - worldid * nwheels
    bodyid = wheel_body_ids[wheel_idx]

    if in_contact[tid] == 0:
        xfrc_applied_out[worldid, bodyid] = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0))
        return

    cx = contact_x[tid]
    cy = contact_y[tid]
    cz = contact_z[tid]

    force_world = fx[tid] * cx + fy[tid] * cy + fz[tid] * cz
    moment_world = my[tid] * cy + mz[tid] * cz

    # Shift tire forces from the contact patch to the wheel center.
    disc_center = xipos[worldid, bodyid]
    r = (contact_pos[tid] + depth[tid] * cz) - disc_center
    moment_world = moment_world + wp.cross(r, force_world)

    xfrc_applied_out[worldid, bodyid] = wp.spatial_vector(force_world, moment_world)


@dataclasses.dataclass(slots=True)
class MujocoFialaTireModule:
    """Chrono-like Fiala tire model applied via MuJoCo-Warp `mjcb_control`.

    This module computes disc-terrain contact with Chrono's `DiscTerrainCollision*` logic and applies the resulting
    tire force/moment as external forces (xfrc_applied) each time MuJoCo-Warp calls `forward()` (and thus at every RK4
    stage).
    """

    tire: FialaTire
    wheel_body_ids: Sequence[int]
    terrain_geom_id: int
    collision_type: CollisionType = CollisionType.SINGLE_POINT

    # Warp device-side buffers (initialized lazily).
    _wheel_body_ids_wp: wp.array | None = None
    _vert_xs: wp.array | None = None
    _vert_ys: wp.array | None = None
    _area_xs: wp.array | None = None
    _area_ys: wp.array | None = None
    _area_n: int = 0

    # Scratch buffers (flat, length = nworld*nwheels).
    _scratch_n: int = 0
    _in_contact: wp.array | None = None
    _depth: wp.array | None = None
    _mu: wp.array | None = None
    _contact_pos: wp.array | None = None
    _contact_x: wp.array | None = None
    _contact_y: wp.array | None = None
    _contact_z: wp.array | None = None

    _vx: wp.array | None = None
    _vy: wp.array | None = None
    _velz: wp.array | None = None
    _omega: wp.array | None = None

    _fx: wp.array | None = None
    _fy: wp.array | None = None
    _fz: wp.array | None = None
    _my: wp.array | None = None
    _mz: wp.array | None = None
    _kappa: wp.array | None = None
    _alpha: wp.array | None = None

    _checked_terrain_geom_type: bool = False

    @classmethod
    def from_json(
        cls,
        tire_json: str | Path,
        *,
        wheel_body_ids: Sequence[int],
        terrain_geom_id: int,
        collision_type: CollisionType = CollisionType.SINGLE_POINT,
    ) -> "MujocoFialaTireModule":
        tire = FialaTire.from_json(tire_json)
        return cls(
            tire=tire,
            wheel_body_ids=tuple(int(x) for x in wheel_body_ids),
            terrain_geom_id=int(terrain_geom_id),
            collision_type=collision_type,
        )

    @classmethod
    def from_mujoco_names(
        cls,
        mj_model,
        tire_json: str | Path,
        *,
        wheel_body_names: Sequence[str],
        terrain_geom_name: str,
        collision_type: CollisionType = CollisionType.SINGLE_POINT,
    ) -> "MujocoFialaTireModule":
        """Create a module by resolving MuJoCo body/geom names to ids."""
        import mujoco  # noqa: PLC0415

        wheel_body_ids: list[int] = []
        for name in wheel_body_names:
            body_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, str(name)))
            if body_id < 0:
                raise ValueError(f"Unknown MuJoCo body name: {name!r}")
            wheel_body_ids.append(body_id)

        terrain_geom_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, str(terrain_geom_name)))
        if terrain_geom_id < 0:
            raise ValueError(f"Unknown MuJoCo geom name: {terrain_geom_name!r}")

        gtype = int(mj_model.geom_type[terrain_geom_id])
        if gtype not in (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD)):
            raise ValueError(
                f"Unsupported terrain geom type {gtype} for {terrain_geom_name!r}; only plane and hfield are supported."
            )

        return cls.from_json(
            tire_json,
            wheel_body_ids=wheel_body_ids,
            terrain_geom_id=terrain_geom_id,
            collision_type=collision_type,
        )

    def _ensure_device_arrays(self, device: wp.context.Device) -> None:
        if self._wheel_body_ids_wp is None:
            self._wheel_body_ids_wp = wp.array(list(self.wheel_body_ids), dtype=wp.int32, device=device)

        if self._vert_xs is None or self._vert_ys is None:
            vert_pairs = self.tire.m_vert_map.table if self.tire.m_has_vert_table else []
            self._vert_xs = wp.array([p[0] for p in vert_pairs], dtype=float, device=device)
            self._vert_ys = wp.array([p[1] for p in vert_pairs], dtype=float, device=device)

        if self._area_xs is None or self._area_ys is None:
            area_dep = ChFunctionInterp()
            ConstructAreaDepthTable(float(self.tire.m_unloaded_radius), area_dep)
            pairs = area_dep.table
            self._area_xs = wp.array([p[0] for p in pairs], dtype=float, device=device)
            self._area_ys = wp.array([p[1] for p in pairs], dtype=float, device=device)
            self._area_n = int(len(pairs))

    def _ensure_scratch(self, device: wp.context.Device, n: int) -> None:
        if self._scratch_n == n and self._in_contact is not None:
            return

        self._scratch_n = n

        self._in_contact = wp.empty(n, dtype=wp.int32, device=device)
        self._depth = wp.empty(n, dtype=float, device=device)
        self._mu = wp.empty(n, dtype=float, device=device)
        self._contact_pos = wp.empty(n, dtype=wp.vec3, device=device)
        self._contact_x = wp.empty(n, dtype=wp.vec3, device=device)
        self._contact_y = wp.empty(n, dtype=wp.vec3, device=device)
        self._contact_z = wp.empty(n, dtype=wp.vec3, device=device)

        self._vx = wp.empty(n, dtype=float, device=device)
        self._vy = wp.empty(n, dtype=float, device=device)
        self._velz = wp.empty(n, dtype=float, device=device)
        self._omega = wp.empty(n, dtype=float, device=device)

        self._fx = wp.empty(n, dtype=float, device=device)
        self._fy = wp.empty(n, dtype=float, device=device)
        self._fz = wp.empty(n, dtype=float, device=device)
        self._my = wp.empty(n, dtype=float, device=device)
        self._mz = wp.empty(n, dtype=float, device=device)
        self._kappa = wp.empty(n, dtype=float, device=device)
        self._alpha = wp.empty(n, dtype=float, device=device)

    def _ensure_supported_terrain_geom(self, m) -> None:
        if self._checked_terrain_geom_type:
            return
        gtype = int(m.geom_type.numpy()[int(self.terrain_geom_id)])
        if gtype not in (0, 1):  # mjGEOM_PLANE, mjGEOM_HFIELD
            raise ValueError(f"Unsupported terrain geom type {gtype} for geom id {self.terrain_geom_id}.")
        self._checked_terrain_geom_type = True

    def apply(self, m, d) -> None:
        """Compute and write wheel `xfrc_applied` for the current MuJoCo-Warp state."""
        wp.init()
        device = d.xfrc_applied.device
        self._ensure_device_arrays(device)
        self._ensure_supported_terrain_geom(m)

        nwheels = len(self.wheel_body_ids)
        if nwheels == 0:
            return
        n = int(d.nworld) * int(nwheels)
        self._ensure_scratch(device, n)

        assert self._wheel_body_ids_wp is not None
        assert self._vert_xs is not None and self._vert_ys is not None
        assert self._area_xs is not None and self._area_ys is not None

        assert self._in_contact is not None
        assert self._depth is not None and self._mu is not None
        assert self._contact_pos is not None
        assert self._contact_x is not None and self._contact_y is not None and self._contact_z is not None

        assert self._vx is not None and self._vy is not None and self._velz is not None and self._omega is not None
        assert self._fx is not None and self._fy is not None and self._fz is not None
        assert self._my is not None and self._mz is not None
        assert self._kappa is not None and self._alpha is not None

        wp.launch(
            _mujoco_disc_terrain_collision_kernel,
            dim=n,
            inputs=[
                m.geom_type,
                m.geom_dataid,
                m.geom_friction,
                m.hfield_size,
                m.hfield_nrow,
                m.hfield_ncol,
                m.hfield_adr,
                m.hfield_data,
                d.xipos,
                d.xmat,
                d.geom_xpos,
                d.geom_xmat,
                self._wheel_body_ids_wp,
                int(nwheels),
                int(self.terrain_geom_id),
                int(self.collision_type),
                float(self.tire.m_unloaded_radius),
                float(self.tire.m_width),
                self._area_xs,
                self._area_ys,
                int(self._area_n),
            ],
            outputs=[
                self._in_contact,
                self._depth,
                self._mu,
                self._contact_pos,
                self._contact_x,
                self._contact_y,
                self._contact_z,
            ],
            device=device,
        )

        wp.launch(
            _mujoco_tire_contact_kinematics_kernel,
            dim=n,
            inputs=[
                d.xmat,
                d.cvel,
                self._wheel_body_ids_wp,
                int(nwheels),
                self._contact_x,
                self._contact_y,
                self._contact_z,
                self._in_contact,
            ],
            outputs=[self._vx, self._vy, self._velz, self._omega],
            device=device,
        )

        wp.launch(
            fiala_tire_advance_kernel,
            dim=n,
            inputs=[
                self._vx,
                self._vy,
                self._velz,
                self._omega,
                self._depth,
                self._mu,
                float(self.tire.m_unloaded_radius),
                float(self.tire.m_width),
                float(self.tire.m_rolling_resistance),
                float(self.tire.m_mu_0),
                float(self.tire.m_c_slip),
                float(self.tire.m_c_alpha),
                float(self.tire.m_u_min),
                float(self.tire.m_u_max),
                float(self.tire.m_normalStiffness),
                float(self.tire.m_normalDamping),
                int(self.tire.m_has_vert_table),
                self._vert_xs,
                self._vert_ys,
                int(self._vert_xs.shape[0]),
                float(self.tire.m_max_depth),
                float(self.tire.m_max_val),
                float(self.tire.m_slope),
            ],
            outputs=[
                self._fx,
                self._fy,
                self._fz,
                self._my,
                self._mz,
                self._kappa,
                self._alpha,
            ],
            device=device,
        )

        wp.launch(
            _mujoco_apply_tire_wrenches_kernel,
            dim=n,
            inputs=[
                d.xipos,
                self._wheel_body_ids_wp,
                int(nwheels),
                self._in_contact,
                self._depth,
                self._contact_pos,
                self._contact_x,
                self._contact_y,
                self._contact_z,
                self._fx,
                self._fy,
                self._fz,
                self._my,
                self._mz,
            ],
            outputs=[d.xfrc_applied],
            device=device,
        )
