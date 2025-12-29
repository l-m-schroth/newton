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

from .disc_terrain_collision import ChFunctionInterp, CollisionType, ConstructAreaDepthTable
from .fiala_tire import FialaTire


# NOTE: Warp does not currently support calling `@wp.func` across Python modules in this repository layout.
# Keep all Warp device functions used by `_mujoco_fiala_tire_forces_kernel` in this file.


@wp.func
def _wp_cross(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@wp.func
def _wp_dot(a: wp.vec3, b: wp.vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@wp.func
def _wp_len2(a: wp.vec3) -> float:
    return _wp_dot(a, a)


@wp.func
def _wp_normalize(a: wp.vec3) -> wp.vec3:
    n2 = _wp_len2(a)
    if n2 == 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    inv = 1.0 / wp.sqrt(n2)
    return wp.vec3(a[0] * inv, a[1] * inv, a[2] * inv)


@wp.func
def _wp_height(p: wp.vec3) -> float:
    return p[2]


# Note: We intentionally only support MuJoCo `plane` and `hfield` geoms here.
_MJ_GEOM_PLANE = 0  # mujoco.mjtGeom.mjGEOM_PLANE
_MJ_GEOM_HFIELD = 1  # mujoco.mjtGeom.mjGEOM_HFIELD


@wp.func
def _wp_hfield_height_normal_local(
    x: float,
    y: float,
    size_x: float,
    size_y: float,
    size_z_top: float,
    nrow: int,
    ncol: int,
    adr: int,
    hfield_data: wp.array(dtype=float),
) -> wp.vec4:
    # MuJoCo reference: mujoco/src/engine/engine_collision_convex.c::mjc_ConvexHField (grid layout)
    # MuJoCo reference: mujoco/src/engine/engine_ray.c::mj_rayHfieldNormal (triangulation)
    #
    # Heightfield local frame:
    #   x in [-size_x, size_x], y in [-size_y, size_y], z in [0, size_z_top]
    # Vertex heights are `hfield_data * size_z_top` (data range [0, 1]).
    #
    # Triangulation per cell uses the diagonal v00 -> v11:
    #   tri1: (v00, v11, v10) for (tx >= ty)
    #   tri2: (v00, v01, v11) for (tx <  ty)

    height = float(0.0)
    normal = wp.vec3(0.0, 0.0, 1.0)

    if nrow < 2 or ncol < 2:
        return wp.vec4(height, normal[0], normal[1], normal[2])

    dx = (2.0 * size_x) / float(ncol - 1)
    dy = (2.0 * size_y) / float(nrow - 1)
    if dx == 0.0 or dy == 0.0:
        return wp.vec4(height, normal[0], normal[1], normal[2])

    u = (x + size_x) / dx
    v = (y + size_y) / dy

    u = wp.min(wp.max(u, 0.0), float(ncol - 1))
    v = wp.min(wp.max(v, 0.0), float(nrow - 1))

    c = int(wp.floor(u))
    r = int(wp.floor(v))
    if c > ncol - 2:
        c = ncol - 2
    if r > nrow - 2:
        r = nrow - 2

    tx = u - float(c)
    ty = v - float(r)

    idx00 = adr + r * ncol + c
    idx10 = adr + r * ncol + (c + 1)
    idx01 = adr + (r + 1) * ncol + c
    idx11 = adr + (r + 1) * ncol + (c + 1)

    h00 = hfield_data[idx00] * size_z_top
    h10 = hfield_data[idx10] * size_z_top
    h01 = hfield_data[idx01] * size_z_top
    h11 = hfield_data[idx11] * size_z_top

    if tx >= ty:
        height = (1.0 - tx) * h00 + (tx - ty) * h10 + ty * h11

        dz10 = h10 - h00
        dz11 = h11 - h00
        normal = _wp_normalize(wp.vec3(-dz10 * dy, dx * (dz10 - dz11), dx * dy))
    else:
        height = (1.0 - ty) * h00 + (ty - tx) * h01 + tx * h11

        dz01 = h01 - h00
        dz11 = h11 - h00
        normal = _wp_normalize(wp.vec3(dy * (dz01 - dz11), -dx * dz01, dx * dy))

    return wp.vec4(height, normal[0], normal[1], normal[2])


@wp.func
def _wp_mujoco_terrain_height_normal(
    loc: wp.vec3,
    worldid: int,
    terrain_geom_id: int,
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    hfield_size: wp.array(dtype=wp.vec4),
    hfield_nrow: wp.array(dtype=int),
    hfield_ncol: wp.array(dtype=int),
    hfield_adr: wp.array(dtype=int),
    hfield_data: wp.array(dtype=float),
) -> wp.vec4:
    gtype = geom_type[terrain_geom_id]

    pos = geom_xpos[worldid, terrain_geom_id]
    mat = geom_xmat[worldid, terrain_geom_id]
    az = wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])

    if gtype == _MJ_GEOM_PLANE:
        n = _wp_normalize(az)
        nz = n[2]
        if wp.abs(nz) < 1e-12:
            return wp.vec4(pos[2], n[0], n[1], n[2])
        dx = loc[0] - pos[0]
        dy = loc[1] - pos[1]
        dz = -(n[0] * dx + n[1] * dy) / nz
        h = pos[2] + dz
        return wp.vec4(h, n[0], n[1], n[2])

    if gtype == _MJ_GEOM_HFIELD:
        hid = geom_dataid[terrain_geom_id]
        size = hfield_size[hid]
        nrow = hfield_nrow[hid]
        ncol = hfield_ncol[hid]
        adr = hfield_adr[hid]

        ax = wp.vec3(mat[0, 0], mat[1, 0], mat[2, 0])
        ay = wp.vec3(mat[0, 1], mat[1, 1], mat[2, 1])

        delta = wp.vec3(loc[0] - pos[0], loc[1] - pos[1], 0.0)
        x_local = wp.dot(delta, ax)
        y_local = wp.dot(delta, ay)

        hn = _wp_hfield_height_normal_local(x_local, y_local, size[0], size[1], size[2], nrow, ncol, adr, hfield_data)
        h_local = hn[0]
        n_local = wp.vec3(hn[1], hn[2], hn[3])

        h_world = pos[2] + ax[2] * x_local + ay[2] * y_local + az[2] * h_local

        n_world = _wp_normalize(ax * n_local[0] + ay * n_local[1] + az * n_local[2])
        return wp.vec4(h_world, n_world[0], n_world[1], n_world[2])

    return wp.vec4(pos[2], 0.0, 0.0, 1.0)


@wp.func
def _wp_signum(x: float) -> float:
    return wp.where(x > 0.0, 1.0, wp.where(x < 0.0, -1.0, 0.0))


@wp.func
def _wp_clamp(x: float, a: float, b: float) -> float:
    return wp.min(wp.max(x, a), b)


@wp.func
def _wp_sine_step_eval(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    # Chrono reference: chrono/src/chrono/functions/ChFunctionSineStep.cpp::Eval
    if x <= x1:
        return y1
    if x >= x2:
        return y2
    xx = (x - x1) / (x2 - x1)
    return y1 + (y2 - y1) * (xx - wp.sin(2.0 * wp.pi * xx) / (2.0 * wp.pi))


@wp.func
def _wp_interp_get_val(x: float, xs: wp.array(dtype=float), ys: wp.array(dtype=float), n: int) -> float:
    # Chrono reference: chrono/src/chrono/functions/ChFunctionInterp.cpp::GetVal
    if n <= 0:
        return 0.0

    x0 = xs[0]
    y0 = ys[0]
    if x <= x0:
        return y0

    x_last = xs[n - 1]
    y_last = ys[n - 1]
    if x >= x_last:
        return y_last

    idx = int(1)
    while idx < n and xs[idx] <= x:
        idx += 1

    x_prev = xs[idx - 1]
    y_prev = ys[idx - 1]
    x_next = xs[idx]
    y_next = ys[idx]
    return y_prev + (y_next - y_prev) * (x - x_prev) / (x_next - x_prev)


@wp.func
def _wp_normal_stiffness_force(
    depth: float,
    has_table: int,
    normal_stiffness: float,
    xs: wp.array(dtype=float),
    ys: wp.array(dtype=float),
    n: int,
    max_depth: float,
    max_val: float,
    slope: float,
) -> float:
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/FialaTire.cpp::GetNormalStiffnessForce
    if has_table != 0:
        if depth > max_depth:
            return max_val + slope * (depth - max_depth)
        return _wp_interp_get_val(depth, xs, ys, n)
    return normal_stiffness * depth


@wp.func
def _wp_fiala_patch_forces(
    kappa: float,
    alpha: float,
    fz: float,
    mu: float,
    mu0: float,
    c_slip: float,
    c_alpha: float,
    u_min: float,
    u_max: float,
    width: float,
) -> wp.vec3:
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp::FialaPatchForces
    tan_alpha = wp.tan(alpha)
    SsA = wp.min(1.0, wp.sqrt(kappa * kappa + tan_alpha * tan_alpha))
    U = u_max - (u_max - u_min) * SsA
    S_critical = wp.abs(U * fz / (2.0 * c_slip))
    Alpha_critical = wp.atan(3.0 * U * fz / c_alpha)

    U = U * (mu / mu0)

    if wp.abs(kappa) < S_critical:
        fx = c_slip * kappa
    else:
        Fx1 = U * fz
        Fx2 = wp.abs((U * fz) * (U * fz) / (4.0 * kappa * c_slip))
        fx = _wp_signum(kappa) * (Fx1 - Fx2)

    if wp.abs(alpha) <= Alpha_critical:
        H = 1.0 - c_alpha * wp.abs(tan_alpha) / (3.0 * U * fz)
        H3 = H * H * H
        fy = -U * fz * (1.0 - H3) * _wp_signum(alpha)
        mz = U * fz * width * (1.0 - H) * H3 * _wp_signum(alpha)
    else:
        fy = -U * fz * _wp_signum(alpha)
        mz = 0.0

    return wp.vec3(fx, fy, mz)


@wp.func
def _wp_axes_via_chrono_quat_roundtrip(x_axis: wp.vec3, y_axis: wp.vec3, z_axis: wp.vec3) -> wp.mat33:
    # Chrono reference: chrono/src/chrono/core/ChMatrix33.h::{SetFromDirectionAxes,GetQuaternion,SetFromQuaternion}
    #
    # Chrono stores the contact frame rotation in a quaternion (ChCoordsys::rot). When `ChTire.cpp` creates the
    # direction-axes matrix and converts it to quaternion, the effective frame used downstream is the quaternion
    # round-trip of those axes (notably relevant for the 4pt collision method).

    # Matrix elements from SetFromDirectionAxes(X,Y,Z): columns are the axes.
    m00 = x_axis[0]
    m10 = x_axis[1]
    m20 = x_axis[2]
    m01 = y_axis[0]
    m11 = y_axis[1]
    m21 = y_axis[2]
    m02 = z_axis[0]
    m12 = z_axis[1]
    m22 = z_axis[2]

    half = 0.5
    tr = m00 + m11 + m22

    e0 = 0.0
    e1 = 0.0
    e2 = 0.0
    e3 = 0.0

    if tr >= 0.0:
        s = wp.sqrt(tr + 1.0)
        e0 = half * s
        s = half / s
        e1 = (m21 - m12) * s
        e2 = (m02 - m20) * s
        e3 = (m10 - m01) * s
    else:
        i = int(0)
        if m11 > m00:
            i = int(1)
            if m22 > m11:
                i = int(2)
        else:
            if m22 > m00:
                i = int(2)

        if i == 0:
            s = wp.sqrt(m00 - m11 - m22 + 1.0)
            e1 = half * s
            s = half / s
            e2 = (m01 + m10) * s
            e3 = (m20 + m02) * s
            e0 = (m21 - m12) * s
        elif i == 1:
            s = wp.sqrt(m11 - m22 - m00 + 1.0)
            e2 = half * s
            s = half / s
            e3 = (m12 + m21) * s
            e1 = (m01 + m10) * s
            e0 = (m02 - m20) * s
        else:
            s = wp.sqrt(m22 - m00 - m11 + 1.0)
            e3 = half * s
            s = half / s
            e1 = (m20 + m02) * s
            e2 = (m12 + m21) * s
            e0 = (m10 - m01) * s

    # SetFromQuaternion(q)
    e0e0 = e0 * e0
    e1e1 = e1 * e1
    e2e2 = e2 * e2
    e3e3 = e3 * e3
    e0e1 = e0 * e1
    e0e2 = e0 * e2
    e0e3 = e0 * e3
    e1e2 = e1 * e2
    e1e3 = e1 * e3
    e2e3 = e2 * e3

    rm00 = (e0e0 + e1e1) * 2.0 - 1.0
    rm01 = (e1e2 - e0e3) * 2.0
    rm02 = (e1e3 + e0e2) * 2.0
    rm10 = (e1e2 + e0e3) * 2.0
    rm11 = (e0e0 + e2e2) * 2.0 - 1.0
    rm12 = (e2e3 - e0e1) * 2.0
    rm20 = (e1e3 - e0e2) * 2.0
    rm21 = (e2e3 + e0e1) * 2.0
    rm22 = (e0e0 + e3e3) * 2.0 - 1.0

    # Rows of the rotation matrix (columns are axes).
    return wp.mat33(
        rm00,
        rm01,
        rm02,
        rm10,
        rm11,
        rm12,
        rm20,
        rm21,
        rm22,
    )


@wp.kernel
def _mujoco_fiala_tire_forces_kernel(
    # MuJoCo model (subset):
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_friction: wp.array2d(dtype=wp.vec3),
    hfield_size: wp.array(dtype=wp.vec4),
    hfield_nrow: wp.array(dtype=int),
    hfield_ncol: wp.array(dtype=int),
    hfield_adr: wp.array(dtype=int),
    hfield_data: wp.array(dtype=float),
    # MuJoCo data (subset):
    xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    # Tire config:
    wheel_body_ids: wp.array(dtype=int),
    terrain_geom_id: int,
    collision_type: int,
    # Tire parameters (Chrono Fiala):
    unloaded_radius: float,
    width: float,
    rolling_resistance: float,
    mu0: float,
    c_slip: float,
    c_alpha: float,
    u_min: float,
    u_max: float,
    normal_stiffness: float,
    normal_damping: float,
    has_vert_table: int,
    vert_xs: wp.array(dtype=float),
    vert_ys: wp.array(dtype=float),
    vert_n: int,
    vert_max_depth: float,
    vert_max_val: float,
    vert_slope: float,
    # Area-depth table (Chrono ConstructAreaDepthTable) for ENVELOPE:
    area_xs: wp.array(dtype=float),
    area_ys: wp.array(dtype=float),
    area_n: int,
    # Output:
    xfrc_applied_out: wp.array2d(dtype=wp.spatial_vector),
):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChForceElementTire.cpp::GetTireForce
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp::{Synchronize,Advance}
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollision*

    worldid, wheel_idx = wp.tid()
    bodyid = wheel_body_ids[wheel_idx]

    # Default: no tire wrench.
    xfrc_applied_out[worldid, bodyid] = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0))

    disc_center = xipos[worldid, bodyid]
    body_R = xmat[worldid, bodyid]
    disc_normal = wp.vec3(body_R[0, 1], body_R[1, 1], body_R[2, 1])  # wheel axis (Chrono: Y)
    disc_normal = _wp_normalize(disc_normal)

    # Terrain friction (Chrono clamps after DiscTerrainCollision).
    mu = geom_friction[worldid, terrain_geom_id][0]

    # ------------------------------------------------------------------
    # Disc-terrain collision (Chrono: DiscTerrainCollision*)
    # ------------------------------------------------------------------
    in_contact = int(0)
    depth = 0.0

    contact_pos = wp.vec3(0.0, 0.0, 0.0)
    contact_x = wp.vec3(1.0, 0.0, 0.0)
    contact_y = wp.vec3(0.0, 1.0, 0.0)
    contact_z = wp.vec3(0.0, 0.0, 1.0)

    world_vertical = wp.vec3(0.0, 0.0, 1.0)

    if collision_type == int(CollisionType.SINGLE_POINT):
        voffset = wp.vec3(0.0, 0.0, 2.0 * unloaded_radius)

        wheel_forward = _wp_cross(disc_normal, world_vertical)
        wheel_forward = _wp_normalize(wheel_forward)
        wheel_bottom = disc_center + unloaded_radius * _wp_cross(disc_normal, wheel_forward)

        hn = _wp_mujoco_terrain_height_normal(
            wheel_bottom + voffset,
            worldid,
            terrain_geom_id,
            geom_type,
            geom_dataid,
            geom_xpos,
            geom_xmat,
            hfield_size,
            hfield_nrow,
            hfield_ncol,
            hfield_adr,
            hfield_data,
        )
        hc = hn[0]
        normal = wp.vec3(hn[1], hn[2], hn[3])

        if _wp_height(disc_center) > hc:
            hc_height = _wp_height(wheel_bottom)
            depth = (hc - hc_height) * normal[2]

            wheel_forward_normal = _wp_cross(disc_normal, normal)
            sin_tilt2 = _wp_len2(wheel_forward_normal)
            if sin_tilt2 >= 1e-3:
                wheel_forward_normal = wheel_forward_normal * (1.0 / wp.sqrt(sin_tilt2))

                depth = unloaded_radius - ((unloaded_radius - depth) * _wp_dot(wheel_forward, wheel_forward_normal))
                if depth > 0.0:
                    wheel_bottom = disc_center + (unloaded_radius - depth) * _wp_cross(disc_normal, wheel_forward_normal)

                    longitudinal = _wp_cross(disc_normal, normal)
                    longitudinal = _wp_normalize(longitudinal)
                    lateral = _wp_cross(normal, longitudinal)

                    contact_pos = wheel_bottom
                    contact_x = longitudinal
                    contact_y = lateral
                    contact_z = normal
                    in_contact = int(1)

    elif collision_type == int(CollisionType.FOUR_POINTS):
        dx = 0.1 * unloaded_radius
        dy = 0.3 * width
        voffset = wp.vec3(0.0, 0.0, 2.0 * unloaded_radius)

        wheel_forward = _wp_cross(disc_normal, world_vertical)
        wheel_forward = _wp_normalize(wheel_forward)
        wheel_bottom = disc_center + unloaded_radius * _wp_cross(disc_normal, wheel_forward)

        hn = _wp_mujoco_terrain_height_normal(
            wheel_bottom + voffset,
            worldid,
            terrain_geom_id,
            geom_type,
            geom_dataid,
            geom_xpos,
            geom_xmat,
            hfield_size,
            hfield_nrow,
            hfield_ncol,
            hfield_adr,
            hfield_data,
        )
        hc = hn[0]
        normal = wp.vec3(hn[1], hn[2], hn[3])

        if _wp_height(disc_center) > hc:
            wheel_forward_normal = _wp_cross(disc_normal, normal)
            sin_tilt2 = _wp_len2(wheel_forward_normal)
            if sin_tilt2 >= 1e-3:
                wheel_forward_normal = wheel_forward_normal * (1.0 / wp.sqrt(sin_tilt2))
                wheel_bottom = disc_center + unloaded_radius * _wp_cross(disc_normal, wheel_forward_normal)

                longitudinal = _wp_cross(disc_normal, normal)
                longitudinal = _wp_normalize(longitudinal)
                lateral = _wp_cross(normal, longitudinal)

                # Project 4 points vertically onto the height field.
                ptQ1 = wheel_bottom + dx * longitudinal
                hn1 = _wp_mujoco_terrain_height_normal(
                    ptQ1 + voffset,
                    worldid,
                    terrain_geom_id,
                    geom_type,
                    geom_dataid,
                    geom_xpos,
                    geom_xmat,
                    hfield_size,
                    hfield_nrow,
                    hfield_ncol,
                    hfield_adr,
                    hfield_data,
                )
                ptQ1 = wp.vec3(ptQ1[0], ptQ1[1], ptQ1[2] - (_wp_height(ptQ1) - hn1[0]))

                ptQ2 = wheel_bottom - dx * longitudinal
                hn2 = _wp_mujoco_terrain_height_normal(
                    ptQ2 + voffset,
                    worldid,
                    terrain_geom_id,
                    geom_type,
                    geom_dataid,
                    geom_xpos,
                    geom_xmat,
                    hfield_size,
                    hfield_nrow,
                    hfield_ncol,
                    hfield_adr,
                    hfield_data,
                )
                ptQ2 = wp.vec3(ptQ2[0], ptQ2[1], ptQ2[2] - (_wp_height(ptQ2) - hn2[0]))

                ptQ3 = wheel_bottom + dy * lateral
                hn3 = _wp_mujoco_terrain_height_normal(
                    ptQ3 + voffset,
                    worldid,
                    terrain_geom_id,
                    geom_type,
                    geom_dataid,
                    geom_xpos,
                    geom_xmat,
                    hfield_size,
                    hfield_nrow,
                    hfield_ncol,
                    hfield_adr,
                    hfield_data,
                )
                ptQ3 = wp.vec3(ptQ3[0], ptQ3[1], ptQ3[2] - (_wp_height(ptQ3) - hn3[0]))

                ptQ4 = wheel_bottom - dy * lateral
                hn4 = _wp_mujoco_terrain_height_normal(
                    ptQ4 + voffset,
                    worldid,
                    terrain_geom_id,
                    geom_type,
                    geom_dataid,
                    geom_xpos,
                    geom_xmat,
                    hfield_size,
                    hfield_nrow,
                    hfield_ncol,
                    hfield_adr,
                    hfield_data,
                )
                ptQ4 = wp.vec3(ptQ4[0], ptQ4[1], ptQ4[2] - (_wp_height(ptQ4) - hn4[0]))

                rQ2Q1 = ptQ1 - ptQ2
                rQ4Q3 = ptQ3 - ptQ4

                terrain_normal = _wp_cross(rQ2Q1, rQ4Q3)
                terrain_normal = _wp_normalize(terrain_normal)

                wheel_bottom = 0.25 * (ptQ1 + ptQ2 + ptQ3 + ptQ4)
                d = wheel_bottom - disc_center
                da = wp.sqrt(_wp_len2(d))
                if da < unloaded_radius:
                    contact_pos = wheel_bottom
                    contact_x = longitudinal
                    contact_y = lateral
                    contact_z = terrain_normal
                    depth = unloaded_radius - da
                    in_contact = int(1)

    elif collision_type == int(CollisionType.ENVELOPE):
        voffset = wp.vec3(0.0, 0.0, unloaded_radius)

        hn = _wp_mujoco_terrain_height_normal(
            disc_center + voffset,
            worldid,
            terrain_geom_id,
            geom_type,
            geom_dataid,
            geom_xpos,
            geom_xmat,
            hfield_size,
            hfield_nrow,
            hfield_ncol,
            hfield_adr,
            hfield_data,
        )
        normal = wp.vec3(hn[1], hn[2], hn[3])
        longitudinal = _wp_cross(disc_normal, normal)
        longitudinal = _wp_normalize(longitudinal)

        x_step = 2.0 * unloaded_radius / 180.0
        A = float(0.0)
        i = float(1.0)
        while i < 180.0:
            x = -unloaded_radius + x_step * i
            p_test = disc_center + x * longitudinal
            hnq = _wp_mujoco_terrain_height_normal(
                p_test + voffset,
                worldid,
                terrain_geom_id,
                geom_type,
                geom_dataid,
                geom_xpos,
                geom_xmat,
                hfield_size,
                hfield_nrow,
                hfield_ncol,
                hfield_adr,
                hfield_data,
            )
            q = hnq[0]
            a = _wp_height(p_test) - wp.sqrt(unloaded_radius * unloaded_radius - x * x)
            if q > a:
                A += q - a
            i = i + 1.0
        A = A * x_step

        if A != 0.0:
            depth = _wp_interp_get_val(A, area_xs, area_ys, area_n)

            dir1 = _wp_cross(disc_normal, normal)
            sin_tilt2 = _wp_len2(dir1)
            if sin_tilt2 >= 1e-3:
                dir1_unit = dir1 * (1.0 / wp.sqrt(sin_tilt2))
                ptD = disc_center + (unloaded_radius - depth) * _wp_cross(disc_normal, dir1_unit)

                hn2 = _wp_mujoco_terrain_height_normal(
                    ptD + 2.0 * voffset,
                    worldid,
                    terrain_geom_id,
                    geom_type,
                    geom_dataid,
                    geom_xpos,
                    geom_xmat,
                    hfield_size,
                    hfield_nrow,
                    hfield_ncol,
                    hfield_adr,
                    hfield_data,
                )
                normal = wp.vec3(hn2[1], hn2[2], hn2[3])
                longitudinal = _wp_cross(disc_normal, normal)
                longitudinal = _wp_normalize(longitudinal)
                lateral = _wp_cross(normal, longitudinal)

                contact_pos = ptD
                contact_x = longitudinal
                contact_y = lateral
                contact_z = normal
                in_contact = int(1)

    # Apply Chrono's quaternion round-trip to match `ChCoordsys::rot` behavior.
    if in_contact != 0:
        R_rt = _wp_axes_via_chrono_quat_roundtrip(contact_x, contact_y, contact_z)
        contact_x = wp.vec3(R_rt[0, 0], R_rt[1, 0], R_rt[2, 0])
        contact_y = wp.vec3(R_rt[0, 1], R_rt[1, 1], R_rt[2, 1])
        contact_z = wp.vec3(R_rt[0, 2], R_rt[1, 2], R_rt[2, 2])

    # ------------------------------------------------------------------
    # Tire forces (Chrono: Synchronize + Advance)
    # ------------------------------------------------------------------
    if in_contact == 0:
        return

    mu = _wp_clamp(mu, 0.1, 1.0)

    w = cvel[worldid, bodyid]  # (rot:lin) in world
    omega_world = wp.vec3(w[0], w[1], w[2])
    vel_world = wp.vec3(w[3], w[4], w[5])

    # Wheel velocity in ISO-C contact frame.
    vx = _wp_dot(vel_world, contact_x)
    vy = _wp_dot(vel_world, contact_y)
    velz = _wp_dot(vel_world, contact_z)

    # Wheel spin about disc normal (Chrono wheel_state.omega).
    omega = _wp_dot(omega_world, disc_normal)

    f_stiff = _wp_normal_stiffness_force(
        depth,
        has_vert_table,
        normal_stiffness,
        vert_xs,
        vert_ys,
        vert_n,
        vert_max_depth,
        vert_max_val,
        vert_slope,
    )
    f_damp = normal_damping * (-velz)
    fz = f_stiff + f_damp
    if fz < 0.0:
        fz = 0.0

    abs_vx = wp.abs(vx)
    if abs_vx != 0.0:
        vsx = vx - omega * (unloaded_radius - depth)
        kappa = -vsx / abs_vx
        alpha = wp.atan2(vy, abs_vx)
    else:
        kappa = 0.0
        alpha = 0.0

    f = _wp_fiala_patch_forces(kappa, alpha, fz, mu, mu0, c_slip, c_alpha, u_min, u_max, width)
    fx = f[0]
    fy = f[1]
    mz = f[2]

    startup = _wp_sine_step_eval(abs_vx, 0.125, 0.0, 0.5, 1.0)
    my = -startup * rolling_resistance * fz * _wp_signum(omega)

    force_world = fx * contact_x + fy * contact_y + fz * contact_z
    moment_world = my * contact_y + mz * contact_z

    # Shift tire forces from the contact patch to the wheel center.
    r = (contact_pos + depth * contact_z) - disc_center
    moment_world = moment_world + _wp_cross(r, force_world)

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

    def apply(self, m, d) -> None:
        """Compute and write wheel `xfrc_applied` for the current MuJoCo-Warp state."""
        wp.init()
        device = d.xfrc_applied.device
        self._ensure_device_arrays(device)

        assert self._wheel_body_ids_wp is not None
        assert self._vert_xs is not None and self._vert_ys is not None
        assert self._area_xs is not None and self._area_ys is not None

        wp.launch(
            _mujoco_fiala_tire_forces_kernel,
            dim=(d.nworld, len(self.wheel_body_ids)),
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
                d.cvel,
                d.geom_xpos,
                d.geom_xmat,
                self._wheel_body_ids_wp,
                int(self.terrain_geom_id),
                int(self.collision_type),
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
                self._area_xs,
                self._area_ys,
                int(self._area_n),
            ],
            outputs=[
                d.xfrc_applied,
            ],
            device=device,
        )
