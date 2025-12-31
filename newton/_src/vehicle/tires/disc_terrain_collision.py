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
import math
from enum import IntEnum
from typing import Protocol, Sequence

import warp as wp

from .fiala_tire import ChFunctionInterp

Vec3 = tuple[float, float, float]

_WORLD_VERTICAL: Vec3 = (0.0, 0.0, 1.0)


def _v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _v_len2(a: Vec3) -> float:
    return _v_dot(a, a)


def _v_len(a: Vec3) -> float:
    return math.sqrt(_v_len2(a))


def _v_normalize(a: Vec3) -> Vec3:
    n = _v_len(a)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _height(p: Vec3) -> float:
    # Chrono reference: chrono/src/chrono_vehicle/ChWorldFrame.h::Height (default Z-up world).
    return _v_dot(p, _WORLD_VERTICAL)


def _axes_via_chrono_quat_roundtrip(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    # Chrono reference: chrono/src/chrono/core/ChMatrix33.h::{SetFromDirectionAxes,GetQuaternion,SetFromQuaternion}
    #
    # Chrono stores a quaternion in `ChCoordsys::rot`. Since `ChTire.cpp` creates a direction-axes matrix and converts
    # it to quaternion, the ground-truth outputs (which convert the quaternion back to a matrix) are the quaternion
    # round-trip axes, even if the original direction axes were not orthonormal (notably in the 4pt method).

    # Matrix elements from SetFromDirectionAxes(X,Y,Z): columns are the axes.
    m00, m10, m20 = x_axis
    m01, m11, m21 = y_axis
    m02, m12, m22 = z_axis

    # GetQuaternion()
    half = 0.5
    tr = m00 + m11 + m22
    if tr >= 0.0:
        s = math.sqrt(tr + 1.0)
        e0 = half * s
        s = half / s
        e1 = (m21 - m12) * s
        e2 = (m02 - m20) * s
        e3 = (m10 - m01) * s
    else:
        i = 0
        if m11 > m00:
            i = 1
            if m22 > m11:
                i = 2
        else:
            if m22 > m00:
                i = 2

        if i == 0:
            s = math.sqrt(m00 - m11 - m22 + 1.0)
            e1 = half * s
            s = half / s
            e2 = (m01 + m10) * s
            e3 = (m20 + m02) * s
            e0 = (m21 - m12) * s
        elif i == 1:
            s = math.sqrt(m11 - m22 - m00 + 1.0)
            e2 = half * s
            s = half / s
            e3 = (m12 + m21) * s
            e1 = (m01 + m10) * s
            e0 = (m02 - m20) * s
        else:
            s = math.sqrt(m22 - m00 - m11 + 1.0)
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

    rx = (rm00, rm10, rm20)
    ry = (rm01, rm11, rm21)
    rz = (rm02, rm12, rm22)
    return (rx, ry, rz)


class TerrainQuery(Protocol):
    def GetHeight(self, loc: Vec3) -> float: ...

    def GetNormal(self, loc: Vec3) -> Vec3: ...

    def GetCoefficientFriction(self, loc: Vec3) -> float: ...

    def GetProperties(self, loc: Vec3) -> tuple[float, Vec3, float]: ...


class CollisionType(IntEnum):
    SINGLE_POINT = 0
    FOUR_POINTS = 1
    ENVELOPE = 2


@dataclasses.dataclass(slots=True)
class ContactPatch:
    pos: Vec3 = (0.0, 0.0, 0.0)
    x_axis: Vec3 = (1.0, 0.0, 0.0)
    y_axis: Vec3 = (0.0, 1.0, 0.0)
    z_axis: Vec3 = (0.0, 0.0, 1.0)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "pos": [float(self.pos[0]), float(self.pos[1]), float(self.pos[2])],
            "x_axis": [float(self.x_axis[0]), float(self.x_axis[1]), float(self.x_axis[2])],
            "y_axis": [float(self.y_axis[0]), float(self.y_axis[1]), float(self.y_axis[2])],
            "z_axis": [float(self.z_axis[0]), float(self.z_axis[1]), float(self.z_axis[2])],
        }


def ConstructAreaDepthTable(disc_radius: float, areaDep: ChFunctionInterp) -> None:
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::ConstructAreaDepthTable
    n_lookup = 90
    dep_max = disc_radius  # should be high enough to avoid extrapolation
    dep_step = dep_max / float(n_lookup - 1)
    for i in range(n_lookup):
        dep = dep_step * float(i)
        alpha = 2.0 * math.acos(1.0 - dep / disc_radius)
        area = 0.5 * disc_radius * disc_radius * (alpha - math.sin(alpha))
        areaDep.AddPoint(area, dep)


def DiscTerrainCollision(
    method: CollisionType,
    terrain: TerrainQuery,
    disc_center: Vec3,
    disc_normal: Vec3,
    disc_radius: float,
    width: float,
    areaDep: ChFunctionInterp,
) -> dict[str, object]:
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollision
    if method == CollisionType.SINGLE_POINT:
        return DiscTerrainCollision1pt(terrain, disc_center, disc_normal, disc_radius)
    if method == CollisionType.FOUR_POINTS:
        return DiscTerrainCollision4pt(terrain, disc_center, disc_normal, disc_radius, width)
    if method == CollisionType.ENVELOPE:
        return DiscTerrainCollisionEnvelope(terrain, disc_center, disc_normal, disc_radius, width, areaDep)
    raise ValueError(f"Unsupported collision method: {method}")


def DiscTerrainCollision1pt(
    terrain: TerrainQuery,
    disc_center: Vec3,
    disc_normal: Vec3,
    disc_radius: float,
) -> dict[str, object]:
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollision1pt
    contact = ContactPatch()
    depth = 0.0

    disc_normal = _v_normalize(disc_normal)

    voffset = _v_mul(_WORLD_VERTICAL, 2.0 * disc_radius)

    wheel_forward = _v_cross(disc_normal, _WORLD_VERTICAL)
    wheel_forward = _v_normalize(wheel_forward)
    wheel_bottom_location = _v_add(disc_center, _v_mul(_v_cross(disc_normal, wheel_forward), disc_radius))

    hc, normal, mu = terrain.GetProperties(_v_add(wheel_bottom_location, voffset))

    disc_height = _height(disc_center)
    if disc_height <= hc:
        return {"in_contact": False, "contact": contact, "depth": depth, "mu": float(mu)}

    hc_height = _height(wheel_bottom_location)
    depth = (hc - hc_height) * _v_dot(_WORLD_VERTICAL, normal)

    wheel_forward_normal = _v_cross(disc_normal, normal)
    sin_tilt2 = _v_len2(wheel_forward_normal)
    if sin_tilt2 < 1e-3:
        return {"in_contact": False, "contact": contact, "depth": depth, "mu": float(mu)}
    wheel_forward_normal = _v_mul(wheel_forward_normal, 1.0 / math.sqrt(sin_tilt2))

    depth = disc_radius - ((disc_radius - depth) * _v_dot(wheel_forward, wheel_forward_normal))
    if depth <= 0.0:
        return {"in_contact": False, "contact": contact, "depth": depth, "mu": float(mu)}

    wheel_bottom_location = _v_add(disc_center, _v_mul(_v_cross(disc_normal, wheel_forward_normal), disc_radius - depth))

    longitudinal = _v_cross(disc_normal, normal)
    longitudinal = _v_normalize(longitudinal)
    lateral = _v_cross(normal, longitudinal)

    contact.pos = wheel_bottom_location
    contact.x_axis, contact.y_axis, contact.z_axis = _axes_via_chrono_quat_roundtrip(longitudinal, lateral, normal)

    return {"in_contact": True, "contact": contact, "depth": depth, "mu": float(mu)}


def DiscTerrainCollision4pt(
    terrain: TerrainQuery,
    disc_center: Vec3,
    disc_normal: Vec3,
    disc_radius: float,
    width: float,
) -> dict[str, object]:
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollision4pt
    contact = ContactPatch()
    depth = 0.0

    disc_normal = _v_normalize(disc_normal)

    dx = 0.1 * disc_radius
    dy = 0.3 * width

    voffset = _v_mul(_WORLD_VERTICAL, 2.0 * disc_radius)

    wheel_forward = _v_cross(disc_normal, _WORLD_VERTICAL)
    wheel_forward = _v_normalize(wheel_forward)
    wheel_bottom_location = _v_add(disc_center, _v_mul(_v_cross(disc_normal, wheel_forward), disc_radius))

    hc, normal, mu = terrain.GetProperties(_v_add(wheel_bottom_location, voffset))

    disc_height = _height(disc_center)
    if disc_height <= hc:
        return {"in_contact": False, "contact": contact, "depth": depth, "mu": float(mu)}

    wheel_forward_normal = _v_cross(disc_normal, normal)
    sin_tilt2 = _v_len2(wheel_forward_normal)
    if sin_tilt2 < 1e-3:
        return {"in_contact": False, "contact": contact, "depth": depth, "mu": float(mu)}
    wheel_forward_normal = _v_mul(wheel_forward_normal, 1.0 / math.sqrt(sin_tilt2))

    wheel_bottom_location = _v_add(disc_center, _v_mul(_v_cross(disc_normal, wheel_forward_normal), disc_radius))

    longitudinal = _v_cross(disc_normal, normal)
    longitudinal = _v_normalize(longitudinal)
    lateral = _v_cross(normal, longitudinal)

    # Four contact points in the contact patch.
    def _project_to_height(p: Vec3) -> Vec3:
        h = terrain.GetHeight(_v_add(p, voffset))
        return (p[0], p[1], p[2] - (_height(p) - h))

    ptQ1 = _project_to_height(_v_add(wheel_bottom_location, _v_mul(longitudinal, dx)))
    ptQ2 = _project_to_height(_v_add(wheel_bottom_location, _v_mul(longitudinal, -dx)))
    ptQ3 = _project_to_height(_v_add(wheel_bottom_location, _v_mul(lateral, dy)))
    ptQ4 = _project_to_height(_v_add(wheel_bottom_location, _v_mul(lateral, -dy)))

    rQ2Q1 = _v_sub(ptQ1, ptQ2)
    rQ4Q3 = _v_sub(ptQ3, ptQ4)

    terrain_normal = _v_cross(rQ2Q1, rQ4Q3)
    terrain_normal = _v_normalize(terrain_normal)

    wheel_bottom_location = _v_mul(_v_add(_v_add(ptQ1, ptQ2), _v_add(ptQ3, ptQ4)), 0.25)
    d = _v_sub(wheel_bottom_location, disc_center)
    da = _v_len(d)

    if da >= disc_radius:
        return {"in_contact": False, "contact": contact, "depth": depth, "mu": float(mu)}

    contact.pos = wheel_bottom_location
    contact.x_axis, contact.y_axis, contact.z_axis = _axes_via_chrono_quat_roundtrip(longitudinal, lateral, terrain_normal)

    depth = disc_radius - da
    return {"in_contact": True, "contact": contact, "depth": depth, "mu": float(mu)}


def DiscTerrainCollisionEnvelope(
    terrain: TerrainQuery,
    disc_center: Vec3,
    disc_normal: Vec3,
    disc_radius: float,
    width: float,
    areaDep: ChFunctionInterp,
) -> dict[str, object]:
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollisionEnvelope
    (void_width,) = (width,)  # Match Chrono signature; envelope uses `width` for API consistency.
    contact = ContactPatch()
    depth = 0.0
    mu = 0.0

    disc_normal = _v_normalize(disc_normal)

    voffset = _v_mul(_WORLD_VERTICAL, disc_radius)

    normal = terrain.GetNormal(_v_add(disc_center, voffset))
    longitudinal = _v_cross(disc_normal, normal)
    longitudinal = _v_normalize(longitudinal)

    n_div = 180
    x_step = 2.0 * disc_radius / float(n_div)
    A = 0.0
    for i in range(1, n_div):
        x = -disc_radius + x_step * float(i)
        p_test = _v_add(disc_center, _v_mul(longitudinal, x))
        q = terrain.GetHeight(_v_add(p_test, voffset))
        a = _height(p_test) - math.sqrt(disc_radius * disc_radius - x * x)
        if q > a:
            A += q - a
    A *= x_step

    if A == 0.0:
        return {"in_contact": False, "contact": contact, "depth": depth, "mu": float(mu)}

    depth = areaDep.GetVal(A)

    dir1 = _v_cross(disc_normal, normal)
    sin_tilt2 = _v_len2(dir1)
    if sin_tilt2 < 1e-3:
        return {"in_contact": False, "contact": contact, "depth": depth, "mu": float(mu)}

    dir1_unit = _v_mul(dir1, 1.0 / math.sqrt(sin_tilt2))
    ptD = _v_add(disc_center, _v_mul(_v_cross(disc_normal, dir1_unit), disc_radius - depth))

    normal = terrain.GetNormal(_v_add(ptD, _v_mul(voffset, 2.0)))
    longitudinal = _v_cross(disc_normal, normal)
    longitudinal = _v_normalize(longitudinal)
    lateral = _v_cross(normal, longitudinal)

    contact.pos = ptD
    contact.x_axis, contact.y_axis, contact.z_axis = _axes_via_chrono_quat_roundtrip(longitudinal, lateral, normal)

    mu = terrain.GetCoefficientFriction(ptD)
    return {"in_contact": True, "contact": contact, "depth": depth, "mu": float(mu)}


class TerrainType(IntEnum):
    PLANE = 0
    SINUSOID = 1


@dataclasses.dataclass(slots=True)
class AnalyticTerrain:
    """Engine-independent terrain for unit tests and as a reference terrain query implementation."""

    type: TerrainType
    mu: float = 0.8

    # Plane parameters:
    point: Vec3 = (0.0, 0.0, 0.0)
    normal: Vec3 = (0.0, 0.0, 1.0)

    # Sinusoid parameters:
    base: float = 0.0
    amp: float = 0.0
    freq: float = 1.0

    @staticmethod
    def Plane(height: float = 0.0, *, mu: float = 0.8) -> "AnalyticTerrain":
        return AnalyticTerrain(type=TerrainType.PLANE, mu=mu, point=(0.0, 0.0, float(height)), normal=(0.0, 0.0, 1.0))

    @staticmethod
    def PlaneFromPointNormal(point: Vec3, normal: Vec3, *, mu: float = 0.8) -> "AnalyticTerrain":
        n = _v_normalize(normal)
        return AnalyticTerrain(type=TerrainType.PLANE, mu=mu, point=tuple(map(float, point)), normal=n)

    @staticmethod
    def Sinusoid(base: float, amp: float, freq: float, *, mu: float = 0.8) -> "AnalyticTerrain":
        return AnalyticTerrain(type=TerrainType.SINUSOID, mu=mu, base=float(base), amp=float(amp), freq=float(freq))

    def GetHeight(self, loc: Vec3) -> float:
        # Chrono reference: newton/newton/tests/tires/chrono_gt/newton_chrono_gt.cpp::AnalyticTerrain::GetHeight
        if self.type == TerrainType.PLANE:
            nz = self.normal[2]
            if abs(nz) < 1e-12:
                return float(self.point[2])
            dx = loc[0] - self.point[0]
            dy = loc[1] - self.point[1]
            dz = -(self.normal[0] * dx + self.normal[1] * dy) / nz
            return float(self.point[2] + dz)

        return float(self.base + self.amp * math.sin(self.freq * loc[0]) * math.sin(self.freq * loc[1]))

    def GetNormal(self, loc: Vec3) -> Vec3:
        # As in newton/newton/tests/tires/chrono_gt/newton_chrono_gt.cpp::AnalyticTerrain::GetNormal
        # NOTE (Lukas): F(x,y,z) = f(x,y) - z = 0 defines the surface. ∇F is a normal to the surface.
        # consider any curve r  on surface F(r(t), then clearly d​F/dt(r(t))=∇F(r(t))⋅r′(t) = 0, r′(t) is any tangent vector.
        # In simple terms: Surface is defined as a level curve. Gradient is always orthogonal to level curves. 
        if self.type == TerrainType.PLANE:
            return self.normal

        sx = math.sin(self.freq * loc[0])
        sy = math.sin(self.freq * loc[1])
        cx = math.cos(self.freq * loc[0])
        cy = math.cos(self.freq * loc[1])
        dfdx = self.amp * self.freq * cx * sy
        dfdy = self.amp * self.freq * sx * cy
        return _v_normalize((-dfdx, -dfdy, 1.0))

    def GetCoefficientFriction(self, loc: Vec3) -> float:
        (void_loc,) = (loc,)
        return float(self.mu)

    def GetProperties(self, loc: Vec3) -> tuple[float, Vec3, float]:
        return (self.GetHeight(loc), self.GetNormal(loc), float(self.mu))


@dataclasses.dataclass(slots=True)
class HFieldTerrain:
    """Simple axis-aligned heightfield terrain using MuJoCo's grid layout and triangulation.

    This matches the ground-truth implementation in:
      - newton/newton/tests/tires/chrono_gt/newton_chrono_gt.cpp::HFieldTerrain

    Notes:
      - The heightfield is axis-aligned (no rotation); only a translation `pos` is supported.
      - Heights are computed from `data * size[2]`, consistent with the ground-truth CLI.
    """

    size: tuple[float, float, float, float]  # (size_x, size_y, size_z_top, size_z_bottom)
    nrow: int
    ncol: int
    data: Sequence[float]
    pos: Vec3 = (0.0, 0.0, 0.0)
    mu: float = 0.8

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.nrow <= 0 or self.ncol <= 0:
            raise ValueError("HFieldTerrain: nrow/ncol must be > 0")
        if len(self.data) != self.nrow * self.ncol:
            raise ValueError("HFieldTerrain: data length must match nrow*ncol")

    def GetHeight(self, loc: Vec3) -> float:
        # Chrono reference: newton/newton/tests/tires/chrono_gt/newton_chrono_gt.cpp::HFieldTerrain::GetHeight
        if self.nrow < 2 or self.ncol < 2 or len(self.data) == 0:
            return float(self.pos[2])

        size_x, size_y, size_z, _size_z_bottom = self.size
        dx = (2.0 * size_x) / float(self.ncol - 1)
        dy = (2.0 * size_y) / float(self.nrow - 1)
        if dx == 0.0 or dy == 0.0:
            return float(self.pos[2])

        x = loc[0] - self.pos[0]
        y = loc[1] - self.pos[1]

        u = (x + size_x) / dx
        v = (y + size_y) / dy
        u = max(0.0, min(u, float(self.ncol - 1)))
        v = max(0.0, min(v, float(self.nrow - 1)))

        c = int(math.floor(u))
        r = int(math.floor(v))
        if c > self.ncol - 2:
            c = self.ncol - 2
        if r > self.nrow - 2:
            r = self.nrow - 2

        tx = u - float(c)
        ty = v - float(r)

        def h(rr: int, cc: int) -> float:
            return float(self.data[rr * self.ncol + cc]) * float(size_z)

        h00 = h(r, c)
        h10 = h(r, c + 1)
        h01 = h(r + 1, c)
        h11 = h(r + 1, c + 1)

        if tx >= ty:
            z = (1.0 - tx) * h00 + (tx - ty) * h10 + ty * h11
        else:
            z = (1.0 - ty) * h00 + (ty - tx) * h01 + tx * h11

        return float(self.pos[2] + z)

    def GetNormal(self, loc: Vec3) -> Vec3:
        # Chrono reference: newton/newton/tests/tires/chrono_gt/newton_chrono_gt.cpp::HFieldTerrain::GetNormal
        if self.nrow < 2 or self.ncol < 2 or len(self.data) == 0:
            return (0.0, 0.0, 1.0)

        size_x, size_y, size_z, _size_z_bottom = self.size
        dx = (2.0 * size_x) / float(self.ncol - 1)
        dy = (2.0 * size_y) / float(self.nrow - 1)
        if dx == 0.0 or dy == 0.0:
            return (0.0, 0.0, 1.0)

        x = loc[0] - self.pos[0]
        y = loc[1] - self.pos[1]

        u = (x + size_x) / dx
        v = (y + size_y) / dy
        u = max(0.0, min(u, float(self.ncol - 1)))
        v = max(0.0, min(v, float(self.nrow - 1)))

        c = int(math.floor(u))
        r = int(math.floor(v))
        if c > self.ncol - 2:
            c = self.ncol - 2
        if r > self.nrow - 2:
            r = self.nrow - 2

        tx = u - float(c)
        ty = v - float(r)

        def h(rr: int, cc: int) -> float:
            return float(self.data[rr * self.ncol + cc]) * float(size_z)

        h00 = h(r, c)
        h10 = h(r, c + 1)
        h01 = h(r + 1, c)
        h11 = h(r + 1, c + 1)

        if tx >= ty:
            dz10 = h10 - h00
            dz11 = h11 - h00
            n = (-dz10 * dy, dx * (dz10 - dz11), dx * dy)
        else:
            dz01 = h01 - h00
            dz11 = h11 - h00
            n = (dy * (dz01 - dz11), -dx * dz01, dx * dy)

        return _v_normalize(n)

    def GetCoefficientFriction(self, loc: Vec3) -> float:
        (void_loc,) = (loc,)
        return float(self.mu)

    def GetProperties(self, loc: Vec3) -> tuple[float, Vec3, float]:
        return (self.GetHeight(loc), self.GetNormal(loc), float(self.mu))


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

# NOTE (Lukas): Currently the terrain height and normals are obtained from these functions, which are part of the warp kernel. 
# Thus the height and normal function is hardcoded.
# At a later point, I need to find a way to cleanly integrate it in a general way which Mujoco/ Newton's terrain representation. 
@wp.func
def _wp_terrain_height(
    loc: wp.vec3,
    terrain_type: int,
    plane_px: float,
    plane_py: float,
    plane_pz: float,
    plane_nx: float,
    plane_ny: float,
    plane_nz: float,
    base: float,
    amp: float,
    freq: float,
) -> float:
    if terrain_type == 0:
        nz = plane_nz
        if wp.abs(nz) < 1e-12:
            return plane_pz
        dx = loc[0] - plane_px
        dy = loc[1] - plane_py
        dz = -(plane_nx * dx + plane_ny * dy) / nz
        return plane_pz + dz

    return base + amp * wp.sin(freq * loc[0]) * wp.sin(freq * loc[1])


@wp.func
def _wp_terrain_normal(
    loc: wp.vec3,
    terrain_type: int,
    plane_nx: float,
    plane_ny: float,
    plane_nz: float,
    amp: float,
    freq: float,
) -> wp.vec3:
    if terrain_type == 0:
        return wp.vec3(plane_nx, plane_ny, plane_nz)

    sx = wp.sin(freq * loc[0])
    sy = wp.sin(freq * loc[1])
    cx = wp.cos(freq * loc[0])
    cy = wp.cos(freq * loc[1])
    dfdx = amp * freq * cx * sy
    dfdy = amp * freq * sx * cy
    return _wp_normalize(wp.vec3(-dfdx, -dfdy, 1.0))


# --------------------------------------------------------------------------------------
# MuJoCo terrain query helpers (plane + heightfield)
# --------------------------------------------------------------------------------------
#
# These functions implement a minimal Chrono-style terrain query interface (height/normal)
# for MuJoCo terrain geoms. They are used in the MuJoCo-Warp tire integration (substep 4).
#
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

    # Default flat result (also used for degenerate grids).
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

    # Clamp to the grid domain.
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
        # tri1 (v00, v11, v10): weights w00=1-tx, w10=tx-ty, w11=ty
        # NOTE (Lukas): v00 = (r, c, h00), v10 = (r, c+1, h10), v11 = (r+1, c+1, h11)
        # v_query = v00 + tx * (v10 - v00) + ty * (v11 - v10)
        height = (1.0 - tx) * h00 + (tx - ty) * h10 + ty * h11

        dz10 = h10 - h00
        dz11 = h11 - h00
        # normal ~ cross(v10-v00, v11-v00)
        # e1 = v10 - v00 = (dx, 0, h10 - h00) = (dx, 0, dz10)
        # e2 = v11 - v00 = (dx, dy, h11 - h00) = (dx, dy, dz11)
        normal = _wp_normalize(wp.vec3(-dz10 * dy, dx * (dz10 - dz11), dx * dy))
    else:
        # tri2 (v00, v01, v11): weights w00=1-ty, w01=ty-tx, w11=tx
        height = (1.0 - ty) * h00 + (ty - tx) * h01 + tx * h11

        dz01 = h01 - h00
        dz11 = h11 - h00
        # normal ~ cross(v11-v00, v01-v00)
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
        if wp.abs(nz) < 1e-12: # vertical plane
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

        # Convert query location to the heightfield local frame (full 3D delta; z-component matters for rotated hfields).
        delta = loc - pos
        x_local = wp.dot(delta, ax)
        y_local = wp.dot(delta, ay)

        hn = _wp_hfield_height_normal_local(x_local, y_local, size[0], size[1], size[2], nrow, ncol, adr, hfield_data)
        h_local = hn[0]
        n_local = wp.vec3(hn[1], hn[2], hn[3])

        # World-space height is the z-coordinate of the surface point with local coords (x_local, y_local, h_local).
        h_world = pos[2] + ax[2] * x_local + ay[2] * y_local + az[2] * h_local

        # Rotate normal into world frame.
        n_world = _wp_normalize(ax * n_local[0] + ay * n_local[1] + az * n_local[2])
        return wp.vec4(h_world, n_world[0], n_world[1], n_world[2])

    # Unsupported terrain type: return NaNs (caller should validate supported types on the host).
    return wp.vec4(wp.nan, wp.nan, wp.nan, wp.nan)


@wp.func
def _wp_axes_via_chrono_quat_roundtrip(x_axis: wp.vec3, y_axis: wp.vec3, z_axis: wp.vec3) -> wp.mat33:
    # Chrono reference: chrono/src/chrono/core/ChMatrix33.h::{SetFromDirectionAxes,GetQuaternion,SetFromQuaternion}
    #
    # Chrono stores the contact frame rotation as a quaternion (ChCoordsys::rot). When Chrono constructs a direction
    # axes matrix and converts it to a quaternion, the downstream frame (converted back to a matrix) corresponds to
    # the quaternion round-trip of those axes (important for `DiscTerrainCollision4pt`, where X/Y are not guaranteed
    # to be orthogonal to the computed terrain normal).

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
def _mujoco_disc_terrain_collision_kernel(
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
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    # Tire config:
    wheel_body_ids: wp.array(dtype=int),
    nwheels: int,
    terrain_geom_id: int,
    collision_type: int,
    disc_radius: float,
    width: float,
    # Area-depth table (Chrono ConstructAreaDepthTable) for ENVELOPE:
    area_xs: wp.array(dtype=float),
    area_ys: wp.array(dtype=float),
    area_n: int,
    # Outputs (flat, length = nworld*nwheels):
    in_contact_out: wp.array(dtype=int),
    depth_out: wp.array(dtype=float),
    mu_out: wp.array(dtype=float),
    pos_out: wp.array(dtype=wp.vec3),
    x_axis_out: wp.array(dtype=wp.vec3),
    y_axis_out: wp.array(dtype=wp.vec3),
    z_axis_out: wp.array(dtype=wp.vec3),
):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollision*
    tid = wp.tid()

    worldid = tid // nwheels
    wheel_idx = tid - worldid * nwheels
    bodyid = wheel_body_ids[wheel_idx]

    contact_pos = wp.vec3(0.0, 0.0, 0.0)
    contact_x = wp.vec3(1.0, 0.0, 0.0)
    contact_y = wp.vec3(0.0, 1.0, 0.0)
    contact_z = wp.vec3(0.0, 0.0, 1.0)

    in_contact = int(0)
    depth = 0.0

    disc_center = xipos[worldid, bodyid]
    body_R = xmat[worldid, bodyid]
    disc_normal = wp.vec3(body_R[0, 1], body_R[1, 1], body_R[2, 1])  # wheel axis (Chrono: Y)
    disc_normal = _wp_normalize(disc_normal)

    # Terrain friction (Chrono clamps after DiscTerrainCollision).
    # Note: `DiscTerrainCollisionEnvelope` only sets friction if contact is found.
    mu_terrain = geom_friction[worldid, terrain_geom_id][0]
    mu = 0.0

    world_vertical = wp.vec3(0.0, 0.0, 1.0)

    if collision_type == 0:
        # SINGLE_POINT
        mu = mu_terrain
        voffset = wp.vec3(0.0, 0.0, 2.0 * disc_radius)

        wheel_forward = _wp_cross(disc_normal, world_vertical)
        wheel_forward = _wp_normalize(wheel_forward)
        wheel_bottom = disc_center + disc_radius * _wp_cross(disc_normal, wheel_forward)

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

                depth = disc_radius - ((disc_radius - depth) * _wp_dot(wheel_forward, wheel_forward_normal))
                if depth > 0.0:
                    wheel_bottom = disc_center + (disc_radius - depth) * _wp_cross(disc_normal, wheel_forward_normal)

                    longitudinal = _wp_cross(disc_normal, normal)
                    longitudinal = _wp_normalize(longitudinal)
                    lateral = _wp_cross(normal, longitudinal)

                    contact_pos = wheel_bottom
                    contact_x = longitudinal
                    contact_y = lateral
                    contact_z = normal
                    in_contact = int(1)

    elif collision_type == 1:
        # FOUR_POINTS
        mu = mu_terrain
        dx = 0.1 * disc_radius
        dy = 0.3 * width
        voffset = wp.vec3(0.0, 0.0, 2.0 * disc_radius)

        wheel_forward = _wp_cross(disc_normal, world_vertical)
        wheel_forward = _wp_normalize(wheel_forward)
        wheel_bottom = disc_center + disc_radius * _wp_cross(disc_normal, wheel_forward)

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
                wheel_bottom = disc_center + disc_radius * _wp_cross(disc_normal, wheel_forward_normal)

                longitudinal = _wp_cross(disc_normal, normal)
                longitudinal = _wp_normalize(longitudinal)
                lateral = _wp_cross(normal, longitudinal)

                # Project 4 points vertically onto the terrain.
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
                if da < disc_radius:
                    contact_pos = wheel_bottom
                    contact_x = longitudinal
                    contact_y = lateral
                    contact_z = terrain_normal
                    depth = disc_radius - da
                    in_contact = int(1)

    else:
        # ENVELOPE
        voffset = wp.vec3(0.0, 0.0, disc_radius)

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

        x_step = 2.0 * disc_radius / 180.0
        A = float(0.0)
        i = float(1.0)
        while i < 180.0:
            x = -disc_radius + x_step * i
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
            a = _wp_height(p_test) - wp.sqrt(disc_radius * disc_radius - x * x)
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
                ptD = disc_center + (disc_radius - depth) * _wp_cross(disc_normal, dir1_unit)

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
                mu = mu_terrain

    # Apply Chrono's quaternion round-trip to match `ChCoordsys::rot` behavior.
    if in_contact != 0:
        R_rt = _wp_axes_via_chrono_quat_roundtrip(contact_x, contact_y, contact_z)
        contact_x = wp.vec3(R_rt[0, 0], R_rt[1, 0], R_rt[2, 0])
        contact_y = wp.vec3(R_rt[0, 1], R_rt[1, 1], R_rt[2, 1])
        contact_z = wp.vec3(R_rt[0, 2], R_rt[1, 2], R_rt[2, 2])

    in_contact_out[tid] = in_contact
    depth_out[tid] = depth
    mu_out[tid] = mu
    pos_out[tid] = contact_pos
    x_axis_out[tid] = contact_x
    y_axis_out[tid] = contact_y
    z_axis_out[tid] = contact_z


@wp.func
def _wp_hfield_height_normal_world_aligned(
    loc: wp.vec3,
    pos_x: float,
    pos_y: float,
    pos_z: float,
    size_x: float,
    size_y: float,
    size_z_top: float,
    nrow: int,
    ncol: int,
    hfield_data: wp.array(dtype=float),
) -> wp.vec4:
    hn = _wp_hfield_height_normal_local(
        loc[0] - pos_x,
        loc[1] - pos_y,
        size_x,
        size_y,
        size_z_top,
        nrow,
        ncol,
        int(0),
        hfield_data,
    )
    return wp.vec4(pos_z + hn[0], hn[1], hn[2], hn[3])


@wp.kernel
def _disc_collision_1pt_hfield_kernel(
    disc_center_in: wp.array(dtype=wp.vec3),
    disc_normal_in: wp.array(dtype=wp.vec3),
    disc_radius: float,
    terrain_mu: float,
    hfield_pos_x: float,
    hfield_pos_y: float,
    hfield_pos_z: float,
    hfield_size_x: float,
    hfield_size_y: float,
    hfield_size_z_top: float,
    hfield_nrow: int,
    hfield_ncol: int,
    hfield_data: wp.array(dtype=float),
    in_contact_out: wp.array(dtype=int),
    depth_out: wp.array(dtype=float),
    mu_out: wp.array(dtype=float),
    pos_out: wp.array(dtype=wp.vec3),
    x_axis_out: wp.array(dtype=wp.vec3),
    y_axis_out: wp.array(dtype=wp.vec3),
    z_axis_out: wp.array(dtype=wp.vec3),
):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollision1pt
    tid = wp.tid()

    contact_pos = wp.vec3(0.0, 0.0, 0.0)
    contact_x = wp.vec3(1.0, 0.0, 0.0)
    contact_y = wp.vec3(0.0, 1.0, 0.0)
    contact_z = wp.vec3(0.0, 0.0, 1.0)

    in_contact = 0
    depth = 0.0
    mu = terrain_mu

    disc_center = disc_center_in[tid]
    disc_normal = _wp_normalize(disc_normal_in[tid])

    voffset = wp.vec3(0.0, 0.0, 2.0 * disc_radius)

    wheel_forward = _wp_cross(disc_normal, wp.vec3(0.0, 0.0, 1.0))
    wheel_forward = _wp_normalize(wheel_forward)
    wheel_bottom = disc_center + disc_radius * _wp_cross(disc_normal, wheel_forward)

    hn = _wp_hfield_height_normal_world_aligned(
        wheel_bottom + voffset,
        hfield_pos_x,
        hfield_pos_y,
        hfield_pos_z,
        hfield_size_x,
        hfield_size_y,
        hfield_size_z_top,
        hfield_nrow,
        hfield_ncol,
        hfield_data,
    )
    hc = hn[0]
    normal = wp.vec3(hn[1], hn[2], hn[3])

    disc_height = _wp_height(disc_center)
    if disc_height > hc:
        hc_height = _wp_height(wheel_bottom)
        depth = (hc - hc_height) * normal[2]

        wheel_forward_normal = _wp_cross(disc_normal, normal)
        sin_tilt2 = _wp_len2(wheel_forward_normal)
        if sin_tilt2 >= 1e-3:
            wheel_forward_normal = wheel_forward_normal * (1.0 / wp.sqrt(sin_tilt2))

            depth = disc_radius - ((disc_radius - depth) * _wp_dot(wheel_forward, wheel_forward_normal))
            if depth > 0.0:
                wheel_bottom = disc_center + (disc_radius - depth) * _wp_cross(disc_normal, wheel_forward_normal)

                longitudinal = _wp_cross(disc_normal, normal)
                longitudinal = _wp_normalize(longitudinal)
                lateral = _wp_cross(normal, longitudinal)

                contact_pos = wheel_bottom
                contact_x = longitudinal
                contact_y = lateral
                contact_z = normal
                in_contact = 1

    if in_contact != 0:
        R_rt = _wp_axes_via_chrono_quat_roundtrip(contact_x, contact_y, contact_z)
        contact_x = wp.vec3(R_rt[0, 0], R_rt[1, 0], R_rt[2, 0])
        contact_y = wp.vec3(R_rt[0, 1], R_rt[1, 1], R_rt[2, 1])
        contact_z = wp.vec3(R_rt[0, 2], R_rt[1, 2], R_rt[2, 2])

    in_contact_out[tid] = in_contact
    depth_out[tid] = depth
    mu_out[tid] = mu
    pos_out[tid] = contact_pos
    x_axis_out[tid] = contact_x
    y_axis_out[tid] = contact_y
    z_axis_out[tid] = contact_z


@wp.kernel
def _disc_collision_4pt_hfield_kernel(
    disc_center_in: wp.array(dtype=wp.vec3),
    disc_normal_in: wp.array(dtype=wp.vec3),
    disc_radius: float,
    width: float,
    terrain_mu: float,
    hfield_pos_x: float,
    hfield_pos_y: float,
    hfield_pos_z: float,
    hfield_size_x: float,
    hfield_size_y: float,
    hfield_size_z_top: float,
    hfield_nrow: int,
    hfield_ncol: int,
    hfield_data: wp.array(dtype=float),
    in_contact_out: wp.array(dtype=int),
    depth_out: wp.array(dtype=float),
    mu_out: wp.array(dtype=float),
    pos_out: wp.array(dtype=wp.vec3),
    x_axis_out: wp.array(dtype=wp.vec3),
    y_axis_out: wp.array(dtype=wp.vec3),
    z_axis_out: wp.array(dtype=wp.vec3),
):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollision4pt
    tid = wp.tid()

    contact_pos = wp.vec3(0.0, 0.0, 0.0)
    contact_x = wp.vec3(1.0, 0.0, 0.0)
    contact_y = wp.vec3(0.0, 1.0, 0.0)
    contact_z = wp.vec3(0.0, 0.0, 1.0)

    in_contact = 0
    depth = 0.0
    mu = terrain_mu

    disc_center = disc_center_in[tid]
    disc_normal = _wp_normalize(disc_normal_in[tid])

    dx = 0.1 * disc_radius
    dy = 0.3 * width

    voffset = wp.vec3(0.0, 0.0, 2.0 * disc_radius)

    wheel_forward = _wp_cross(disc_normal, wp.vec3(0.0, 0.0, 1.0))
    wheel_forward = _wp_normalize(wheel_forward)
    wheel_bottom = disc_center + disc_radius * _wp_cross(disc_normal, wheel_forward)

    hn = _wp_hfield_height_normal_world_aligned(
        wheel_bottom + voffset,
        hfield_pos_x,
        hfield_pos_y,
        hfield_pos_z,
        hfield_size_x,
        hfield_size_y,
        hfield_size_z_top,
        hfield_nrow,
        hfield_ncol,
        hfield_data,
    )
    hc = hn[0]
    normal = wp.vec3(hn[1], hn[2], hn[3])

    disc_height = _wp_height(disc_center)
    if disc_height > hc:
        wheel_forward_normal = _wp_cross(disc_normal, normal)
        sin_tilt2 = _wp_len2(wheel_forward_normal)
        if sin_tilt2 >= 1e-3:
            wheel_forward_normal = wheel_forward_normal * (1.0 / wp.sqrt(sin_tilt2))
            wheel_bottom = disc_center + disc_radius * _wp_cross(disc_normal, wheel_forward_normal)

            longitudinal = _wp_cross(disc_normal, normal)
            longitudinal = _wp_normalize(longitudinal)
            lateral = _wp_cross(normal, longitudinal)

            ptQ1 = wheel_bottom + dx * longitudinal
            hQ1 = _wp_hfield_height_normal_world_aligned(
                ptQ1 + voffset,
                hfield_pos_x,
                hfield_pos_y,
                hfield_pos_z,
                hfield_size_x,
                hfield_size_y,
                hfield_size_z_top,
                hfield_nrow,
                hfield_ncol,
                hfield_data,
            )[0]
            ptQ1 = wp.vec3(ptQ1[0], ptQ1[1], ptQ1[2] - (_wp_height(ptQ1) - hQ1))

            ptQ2 = wheel_bottom - dx * longitudinal
            hQ2 = _wp_hfield_height_normal_world_aligned(
                ptQ2 + voffset,
                hfield_pos_x,
                hfield_pos_y,
                hfield_pos_z,
                hfield_size_x,
                hfield_size_y,
                hfield_size_z_top,
                hfield_nrow,
                hfield_ncol,
                hfield_data,
            )[0]
            ptQ2 = wp.vec3(ptQ2[0], ptQ2[1], ptQ2[2] - (_wp_height(ptQ2) - hQ2))

            ptQ3 = wheel_bottom + dy * lateral
            hQ3 = _wp_hfield_height_normal_world_aligned(
                ptQ3 + voffset,
                hfield_pos_x,
                hfield_pos_y,
                hfield_pos_z,
                hfield_size_x,
                hfield_size_y,
                hfield_size_z_top,
                hfield_nrow,
                hfield_ncol,
                hfield_data,
            )[0]
            ptQ3 = wp.vec3(ptQ3[0], ptQ3[1], ptQ3[2] - (_wp_height(ptQ3) - hQ3))

            ptQ4 = wheel_bottom - dy * lateral
            hQ4 = _wp_hfield_height_normal_world_aligned(
                ptQ4 + voffset,
                hfield_pos_x,
                hfield_pos_y,
                hfield_pos_z,
                hfield_size_x,
                hfield_size_y,
                hfield_size_z_top,
                hfield_nrow,
                hfield_ncol,
                hfield_data,
            )[0]
            ptQ4 = wp.vec3(ptQ4[0], ptQ4[1], ptQ4[2] - (_wp_height(ptQ4) - hQ4))

            rQ2Q1 = ptQ1 - ptQ2
            rQ4Q3 = ptQ3 - ptQ4

            terrain_normal = _wp_cross(rQ2Q1, rQ4Q3)
            terrain_normal = _wp_normalize(terrain_normal)

            wheel_bottom = 0.25 * (ptQ1 + ptQ2 + ptQ3 + ptQ4)
            d = wheel_bottom - disc_center
            da = wp.sqrt(_wp_len2(d))
            if da < disc_radius:
                contact_pos = wheel_bottom
                contact_x = longitudinal
                contact_y = lateral
                contact_z = terrain_normal
                depth = disc_radius - da
                in_contact = 1

    if in_contact != 0:
        R_rt = _wp_axes_via_chrono_quat_roundtrip(contact_x, contact_y, contact_z)
        contact_x = wp.vec3(R_rt[0, 0], R_rt[1, 0], R_rt[2, 0])
        contact_y = wp.vec3(R_rt[0, 1], R_rt[1, 1], R_rt[2, 1])
        contact_z = wp.vec3(R_rt[0, 2], R_rt[1, 2], R_rt[2, 2])

    in_contact_out[tid] = in_contact
    depth_out[tid] = depth
    mu_out[tid] = mu
    pos_out[tid] = contact_pos
    x_axis_out[tid] = contact_x
    y_axis_out[tid] = contact_y
    z_axis_out[tid] = contact_z


@wp.kernel
def _disc_collision_envelope_hfield_kernel(
    disc_center_in: wp.array(dtype=wp.vec3),
    disc_normal_in: wp.array(dtype=wp.vec3),
    disc_radius: float,
    width: float,
    area_xs: wp.array(dtype=float),
    area_ys: wp.array(dtype=float),
    area_n: int,
    terrain_mu: float,
    hfield_pos_x: float,
    hfield_pos_y: float,
    hfield_pos_z: float,
    hfield_size_x: float,
    hfield_size_y: float,
    hfield_size_z_top: float,
    hfield_nrow: int,
    hfield_ncol: int,
    hfield_data: wp.array(dtype=float),
    in_contact_out: wp.array(dtype=int),
    depth_out: wp.array(dtype=float),
    mu_out: wp.array(dtype=float),
    pos_out: wp.array(dtype=wp.vec3),
    x_axis_out: wp.array(dtype=wp.vec3),
    y_axis_out: wp.array(dtype=wp.vec3),
    z_axis_out: wp.array(dtype=wp.vec3),
):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollisionEnvelope
    (void_width,) = (width,)
    tid = wp.tid()

    contact_pos = wp.vec3(0.0, 0.0, 0.0)
    contact_x = wp.vec3(1.0, 0.0, 0.0)
    contact_y = wp.vec3(0.0, 1.0, 0.0)
    contact_z = wp.vec3(0.0, 0.0, 1.0)

    in_contact = 0
    depth = 0.0
    mu = 0.0

    disc_center = disc_center_in[tid]
    disc_normal = _wp_normalize(disc_normal_in[tid])

    voffset = wp.vec3(0.0, 0.0, disc_radius)

    hn0 = _wp_hfield_height_normal_world_aligned(
        disc_center + voffset,
        hfield_pos_x,
        hfield_pos_y,
        hfield_pos_z,
        hfield_size_x,
        hfield_size_y,
        hfield_size_z_top,
        hfield_nrow,
        hfield_ncol,
        hfield_data,
    )
    normal = wp.vec3(hn0[1], hn0[2], hn0[3])
    longitudinal = _wp_cross(disc_normal, normal)
    longitudinal = _wp_normalize(longitudinal)

    x_step = 2.0 * disc_radius / 180.0
    A = float(0.0)
    i = float(1.0)
    while i < 180.0:
        x = -disc_radius + x_step * i
        p_test = disc_center + x * longitudinal
        q = _wp_hfield_height_normal_world_aligned(
            p_test + voffset,
            hfield_pos_x,
            hfield_pos_y,
            hfield_pos_z,
            hfield_size_x,
            hfield_size_y,
            hfield_size_z_top,
            hfield_nrow,
            hfield_ncol,
            hfield_data,
        )[0]
        a = _wp_height(p_test) - wp.sqrt(disc_radius * disc_radius - x * x)
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
            ptD = disc_center + (disc_radius - depth) * _wp_cross(disc_normal, dir1_unit)

            hn2 = _wp_hfield_height_normal_world_aligned(
                ptD + 2.0 * voffset,
                hfield_pos_x,
                hfield_pos_y,
                hfield_pos_z,
                hfield_size_x,
                hfield_size_y,
                hfield_size_z_top,
                hfield_nrow,
                hfield_ncol,
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

            mu = terrain_mu
            in_contact = 1

    if in_contact != 0:
        R_rt = _wp_axes_via_chrono_quat_roundtrip(contact_x, contact_y, contact_z)
        contact_x = wp.vec3(R_rt[0, 0], R_rt[1, 0], R_rt[2, 0])
        contact_y = wp.vec3(R_rt[0, 1], R_rt[1, 1], R_rt[2, 1])
        contact_z = wp.vec3(R_rt[0, 2], R_rt[1, 2], R_rt[2, 2])

    in_contact_out[tid] = in_contact
    depth_out[tid] = depth
    mu_out[tid] = mu
    pos_out[tid] = contact_pos
    x_axis_out[tid] = contact_x
    y_axis_out[tid] = contact_y
    z_axis_out[tid] = contact_z


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


@wp.kernel
def _disc_collision_1pt_kernel(
    disc_center_in: wp.array(dtype=wp.vec3),
    disc_normal_in: wp.array(dtype=wp.vec3),
    disc_radius: float,
    terrain_type: int,
    terrain_mu: float,
    plane_px: float,
    plane_py: float,
    plane_pz: float,
    plane_nx: float,
    plane_ny: float,
    plane_nz: float,
    base: float,
    amp: float,
    freq: float,
    in_contact_out: wp.array(dtype=int),
    depth_out: wp.array(dtype=float),
    mu_out: wp.array(dtype=float),
    pos_out: wp.array(dtype=wp.vec3),
    x_axis_out: wp.array(dtype=wp.vec3),
    y_axis_out: wp.array(dtype=wp.vec3),
    z_axis_out: wp.array(dtype=wp.vec3),
):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollision1pt
    tid = wp.tid()

    contact_pos = wp.vec3(0.0, 0.0, 0.0)
    contact_x = wp.vec3(1.0, 0.0, 0.0)
    contact_y = wp.vec3(0.0, 1.0, 0.0)
    contact_z = wp.vec3(0.0, 0.0, 1.0)

    in_contact = 0
    depth = 0.0
    mu = 0.0

    disc_center = disc_center_in[tid]
    disc_normal = _wp_normalize(disc_normal_in[tid])

    voffset = wp.vec3(0.0, 0.0, 2.0 * disc_radius)

    wheel_forward = _wp_cross(disc_normal, wp.vec3(0.0, 0.0, 1.0))
    wheel_forward = _wp_normalize(wheel_forward)
    wheel_bottom = disc_center + disc_radius * _wp_cross(disc_normal, wheel_forward)

    hc = _wp_terrain_height(wheel_bottom + voffset, terrain_type, plane_px, plane_py, plane_pz, plane_nx, plane_ny, plane_nz, base, amp, freq)
    normal = _wp_terrain_normal(wheel_bottom + voffset, terrain_type, plane_nx, plane_ny, plane_nz, amp, freq)
    mu = terrain_mu

    disc_height = _wp_height(disc_center)
    if disc_height > hc:
        hc_height = _wp_height(wheel_bottom)
        depth = (hc - hc_height) * normal[2]

        wheel_forward_normal = _wp_cross(disc_normal, normal)
        sin_tilt2 = _wp_len2(wheel_forward_normal)
        if sin_tilt2 >= 1e-3:
            wheel_forward_normal = wheel_forward_normal * (1.0 / wp.sqrt(sin_tilt2))

            depth = disc_radius - ((disc_radius - depth) * _wp_dot(wheel_forward, wheel_forward_normal))
            if depth > 0.0:
                wheel_bottom = disc_center + (disc_radius - depth) * _wp_cross(disc_normal, wheel_forward_normal)

                longitudinal = _wp_cross(disc_normal, normal)
                longitudinal = _wp_normalize(longitudinal)
                lateral = _wp_cross(normal, longitudinal)

                contact_pos = wheel_bottom
                contact_x = longitudinal
                contact_y = lateral
                contact_z = normal
                in_contact = 1

    in_contact_out[tid] = in_contact
    depth_out[tid] = depth
    mu_out[tid] = mu
    pos_out[tid] = contact_pos
    x_axis_out[tid] = contact_x
    y_axis_out[tid] = contact_y
    z_axis_out[tid] = contact_z


@wp.kernel
def _disc_collision_4pt_kernel(
    disc_center_in: wp.array(dtype=wp.vec3),
    disc_normal_in: wp.array(dtype=wp.vec3),
    disc_radius: float,
    width: float,
    terrain_type: int,
    terrain_mu: float,
    plane_px: float,
    plane_py: float,
    plane_pz: float,
    plane_nx: float,
    plane_ny: float,
    plane_nz: float,
    base: float,
    amp: float,
    freq: float,
    in_contact_out: wp.array(dtype=int),
    depth_out: wp.array(dtype=float),
    mu_out: wp.array(dtype=float),
    pos_out: wp.array(dtype=wp.vec3),
    x_axis_out: wp.array(dtype=wp.vec3),
    y_axis_out: wp.array(dtype=wp.vec3),
    z_axis_out: wp.array(dtype=wp.vec3),
):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollision4pt
    tid = wp.tid()

    contact_pos = wp.vec3(0.0, 0.0, 0.0)
    contact_x = wp.vec3(1.0, 0.0, 0.0)
    contact_y = wp.vec3(0.0, 1.0, 0.0)
    contact_z = wp.vec3(0.0, 0.0, 1.0)

    in_contact = 0
    depth = 0.0
    mu = 0.0

    disc_center = disc_center_in[tid]
    disc_normal = _wp_normalize(disc_normal_in[tid])

    dx = 0.1 * disc_radius
    dy = 0.3 * width

    voffset = wp.vec3(0.0, 0.0, 2.0 * disc_radius)

    wheel_forward = _wp_cross(disc_normal, wp.vec3(0.0, 0.0, 1.0))
    wheel_forward = _wp_normalize(wheel_forward)
    wheel_bottom = disc_center + disc_radius * _wp_cross(disc_normal, wheel_forward)

    hc = _wp_terrain_height(wheel_bottom + voffset, terrain_type, plane_px, plane_py, plane_pz, plane_nx, plane_ny, plane_nz, base, amp, freq)
    normal = _wp_terrain_normal(wheel_bottom + voffset, terrain_type, plane_nx, plane_ny, plane_nz, amp, freq)
    mu = terrain_mu

    disc_height = _wp_height(disc_center)
    if disc_height > hc:
        wheel_forward_normal = _wp_cross(disc_normal, normal)
        sin_tilt2 = _wp_len2(wheel_forward_normal)
        if sin_tilt2 >= 1e-3:
            wheel_forward_normal = wheel_forward_normal * (1.0 / wp.sqrt(sin_tilt2))

            wheel_bottom = disc_center + disc_radius * _wp_cross(disc_normal, wheel_forward_normal)

            longitudinal = _wp_cross(disc_normal, normal)
            longitudinal = _wp_normalize(longitudinal)
            lateral = _wp_cross(normal, longitudinal)

            ptQ1 = wheel_bottom + dx * longitudinal
            hQ1 = _wp_terrain_height(
                ptQ1 + voffset,
                terrain_type,
                plane_px,
                plane_py,
                plane_pz,
                plane_nx,
                plane_ny,
                plane_nz,
                base,
                amp,
                freq,
            )
            ptQ1 = wp.vec3(ptQ1[0], ptQ1[1], ptQ1[2] - (_wp_height(ptQ1) - hQ1))

            ptQ2 = wheel_bottom - dx * longitudinal
            hQ2 = _wp_terrain_height(
                ptQ2 + voffset,
                terrain_type,
                plane_px,
                plane_py,
                plane_pz,
                plane_nx,
                plane_ny,
                plane_nz,
                base,
                amp,
                freq,
            )
            ptQ2 = wp.vec3(ptQ2[0], ptQ2[1], ptQ2[2] - (_wp_height(ptQ2) - hQ2))

            ptQ3 = wheel_bottom + dy * lateral
            hQ3 = _wp_terrain_height(
                ptQ3 + voffset,
                terrain_type,
                plane_px,
                plane_py,
                plane_pz,
                plane_nx,
                plane_ny,
                plane_nz,
                base,
                amp,
                freq,
            )
            ptQ3 = wp.vec3(ptQ3[0], ptQ3[1], ptQ3[2] - (_wp_height(ptQ3) - hQ3))

            ptQ4 = wheel_bottom - dy * lateral
            hQ4 = _wp_terrain_height(
                ptQ4 + voffset,
                terrain_type,
                plane_px,
                plane_py,
                plane_pz,
                plane_nx,
                plane_ny,
                plane_nz,
                base,
                amp,
                freq,
            )
            ptQ4 = wp.vec3(ptQ4[0], ptQ4[1], ptQ4[2] - (_wp_height(ptQ4) - hQ4))

            rQ2Q1 = ptQ1 - ptQ2
            rQ4Q3 = ptQ3 - ptQ4

            terrain_normal = _wp_cross(rQ2Q1, rQ4Q3)
            terrain_normal = _wp_normalize(terrain_normal)

            wheel_bottom = 0.25 * (ptQ1 + ptQ2 + ptQ3 + ptQ4)
            d = wheel_bottom - disc_center
            da = wp.sqrt(_wp_len2(d))
            if da < disc_radius:
                contact_pos = wheel_bottom
                contact_x = longitudinal
                contact_y = lateral
                contact_z = terrain_normal
                depth = disc_radius - da
                in_contact = 1

    in_contact_out[tid] = in_contact
    depth_out[tid] = depth
    mu_out[tid] = mu
    pos_out[tid] = contact_pos
    x_axis_out[tid] = contact_x
    y_axis_out[tid] = contact_y
    z_axis_out[tid] = contact_z


@wp.kernel
def _disc_collision_envelope_kernel(
    disc_center_in: wp.array(dtype=wp.vec3),
    disc_normal_in: wp.array(dtype=wp.vec3),
    disc_radius: float,
    width: float,
    area_xs: wp.array(dtype=float),
    area_ys: wp.array(dtype=float),
    area_n: int,
    terrain_type: int,
    terrain_mu: float,
    plane_px: float,
    plane_py: float,
    plane_pz: float,
    plane_nx: float,
    plane_ny: float,
    plane_nz: float,
    base: float,
    amp: float,
    freq: float,
    in_contact_out: wp.array(dtype=int),
    depth_out: wp.array(dtype=float),
    mu_out: wp.array(dtype=float),
    pos_out: wp.array(dtype=wp.vec3),
    x_axis_out: wp.array(dtype=wp.vec3),
    y_axis_out: wp.array(dtype=wp.vec3),
    z_axis_out: wp.array(dtype=wp.vec3),
):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/ChTire.cpp::DiscTerrainCollisionEnvelope
    (void_width,) = (width,)
    tid = wp.tid()

    contact_pos = wp.vec3(0.0, 0.0, 0.0)
    contact_x = wp.vec3(1.0, 0.0, 0.0)
    contact_y = wp.vec3(0.0, 1.0, 0.0)
    contact_z = wp.vec3(0.0, 0.0, 1.0)

    in_contact = 0
    depth = 0.0
    mu = 0.0

    disc_center = disc_center_in[tid]
    disc_normal = _wp_normalize(disc_normal_in[tid])

    voffset = wp.vec3(0.0, 0.0, disc_radius)

    normal = _wp_terrain_normal(disc_center + voffset, terrain_type, plane_nx, plane_ny, plane_nz, amp, freq)
    longitudinal = _wp_cross(disc_normal, normal)
    longitudinal = _wp_normalize(longitudinal)

    x_step = 2.0 * disc_radius / 180.0
    A = float(0.0)
    i = float(1.0)
    while i < 180.0:
        x = -disc_radius + x_step * i
        p_test = disc_center + x * longitudinal
        q = _wp_terrain_height(p_test + voffset, terrain_type, plane_px, plane_py, plane_pz, plane_nx, plane_ny, plane_nz, base, amp, freq)
        a = _wp_height(p_test) - wp.sqrt(disc_radius * disc_radius - x * x)
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
            ptD = disc_center + (disc_radius - depth) * _wp_cross(disc_normal, dir1_unit)

            normal = _wp_terrain_normal(ptD + 2.0 * voffset, terrain_type, plane_nx, plane_ny, plane_nz, amp, freq)
            longitudinal = _wp_cross(disc_normal, normal)
            longitudinal = _wp_normalize(longitudinal)
            lateral = _wp_cross(normal, longitudinal)

            contact_pos = ptD
            contact_x = longitudinal
            contact_y = lateral
            contact_z = normal

            mu = terrain_mu
            in_contact = 1

    in_contact_out[tid] = in_contact
    depth_out[tid] = depth
    mu_out[tid] = mu
    pos_out[tid] = contact_pos
    x_axis_out[tid] = contact_x
    y_axis_out[tid] = contact_y
    z_axis_out[tid] = contact_z


def run_disc_terrain_collision_batched(
    method: CollisionType,
    terrain: AnalyticTerrain,
    disc_center: Sequence[Vec3],
    disc_normal: Sequence[Vec3],
    disc_radius: float,
    width: float,
    *,
    device: str | wp.context.Device = "cpu",
) -> dict[str, list[object]]:
    if len(disc_center) != len(disc_normal):
        raise ValueError("disc_center and disc_normal must have the same length.")

    wp.init()

    n = len(disc_center)
    disc_center_wp = wp.array(disc_center, dtype=wp.vec3, device=device)
    disc_normal_wp = wp.array(disc_normal, dtype=wp.vec3, device=device)

    in_contact_out = wp.empty(n, dtype=int, device=device)
    depth_out = wp.empty(n, dtype=float, device=device)
    mu_out = wp.empty(n, dtype=float, device=device)
    pos_out = wp.empty(n, dtype=wp.vec3, device=device)
    x_axis_out = wp.empty(n, dtype=wp.vec3, device=device)
    y_axis_out = wp.empty(n, dtype=wp.vec3, device=device)
    z_axis_out = wp.empty(n, dtype=wp.vec3, device=device)

    plane_px, plane_py, plane_pz = (float(terrain.point[0]), float(terrain.point[1]), float(terrain.point[2]))
    plane_nx, plane_ny, plane_nz = (float(terrain.normal[0]), float(terrain.normal[1]), float(terrain.normal[2]))

    if method == CollisionType.SINGLE_POINT:
        wp.launch(
            _disc_collision_1pt_kernel,
            dim=n,
            inputs=[
                disc_center_wp,
                disc_normal_wp,
                float(disc_radius),
                int(terrain.type),
                float(terrain.mu),
                plane_px,
                plane_py,
                plane_pz,
                plane_nx,
                plane_ny,
                plane_nz,
                float(terrain.base),
                float(terrain.amp),
                float(terrain.freq),
            ],
            outputs=[in_contact_out, depth_out, mu_out, pos_out, x_axis_out, y_axis_out, z_axis_out],
            device=device,
        )
    elif method == CollisionType.FOUR_POINTS:
        wp.launch(
            _disc_collision_4pt_kernel,
            dim=n,
            inputs=[
                disc_center_wp,
                disc_normal_wp,
                float(disc_radius),
                float(width),
                int(terrain.type),
                float(terrain.mu),
                plane_px,
                plane_py,
                plane_pz,
                plane_nx,
                plane_ny,
                plane_nz,
                float(terrain.base),
                float(terrain.amp),
                float(terrain.freq),
            ],
            outputs=[in_contact_out, depth_out, mu_out, pos_out, x_axis_out, y_axis_out, z_axis_out],
            device=device,
        )
    elif method == CollisionType.ENVELOPE:
        area_dep = ChFunctionInterp()
        ConstructAreaDepthTable(float(disc_radius), area_dep)
        pairs = area_dep.table
        area_xs = wp.array([p[0] for p in pairs], dtype=float, device=device)
        area_ys = wp.array([p[1] for p in pairs], dtype=float, device=device)

        wp.launch(
            _disc_collision_envelope_kernel,
            dim=n,
            inputs=[
                disc_center_wp,
                disc_normal_wp,
                float(disc_radius),
                float(width),
                area_xs,
                area_ys,
                int(len(pairs)),
                int(terrain.type),
                float(terrain.mu),
                plane_px,
                plane_py,
                plane_pz,
                plane_nx,
                plane_ny,
                plane_nz,
                float(terrain.base),
                float(terrain.amp),
                float(terrain.freq),
            ],
            outputs=[in_contact_out, depth_out, mu_out, pos_out, x_axis_out, y_axis_out, z_axis_out],
            device=device,
        )
    else:
        raise ValueError(f"Unsupported collision method: {method}")

    return {
        "in_contact": [bool(x) for x in in_contact_out.numpy().tolist()],
        "depth": depth_out.numpy().tolist(),
        "mu": mu_out.numpy().tolist(),
        "pos": pos_out.numpy().tolist(),
        "x_axis": x_axis_out.numpy().tolist(),
        "y_axis": y_axis_out.numpy().tolist(),
        "z_axis": z_axis_out.numpy().tolist(),
    }


def run_disc_terrain_collision_hfield_batched(
    method: CollisionType,
    terrain: HFieldTerrain,
    disc_center: Sequence[Vec3],
    disc_normal: Sequence[Vec3],
    disc_radius: float,
    width: float,
    *,
    device: str | wp.context.Device = "cpu",
) -> dict[str, list[object]]:
    """Warp-batched disc-terrain collision for axis-aligned heightfields."""
    if len(disc_center) != len(disc_normal):
        raise ValueError("disc_center and disc_normal must have the same length.")

    wp.init()

    n = len(disc_center)
    disc_center_wp = wp.array(disc_center, dtype=wp.vec3, device=device)
    disc_normal_wp = wp.array(disc_normal, dtype=wp.vec3, device=device)

    in_contact_out = wp.empty(n, dtype=int, device=device)
    depth_out = wp.empty(n, dtype=float, device=device)
    mu_out = wp.empty(n, dtype=float, device=device)
    pos_out = wp.empty(n, dtype=wp.vec3, device=device)
    x_axis_out = wp.empty(n, dtype=wp.vec3, device=device)
    y_axis_out = wp.empty(n, dtype=wp.vec3, device=device)
    z_axis_out = wp.empty(n, dtype=wp.vec3, device=device)

    size_x, size_y, size_z_top, _size_z_bottom = terrain.size
    pos_x, pos_y, pos_z = (float(terrain.pos[0]), float(terrain.pos[1]), float(terrain.pos[2]))
    hdata = wp.array([float(x) for x in terrain.data], dtype=float, device=device)

    if method == CollisionType.SINGLE_POINT:
        wp.launch(
            _disc_collision_1pt_hfield_kernel,
            dim=n,
            inputs=[
                disc_center_wp,
                disc_normal_wp,
                float(disc_radius),
                float(terrain.mu),
                pos_x,
                pos_y,
                pos_z,
                float(size_x),
                float(size_y),
                float(size_z_top),
                int(terrain.nrow),
                int(terrain.ncol),
                hdata,
            ],
            outputs=[in_contact_out, depth_out, mu_out, pos_out, x_axis_out, y_axis_out, z_axis_out],
            device=device,
        )
    elif method == CollisionType.FOUR_POINTS:
        wp.launch(
            _disc_collision_4pt_hfield_kernel,
            dim=n,
            inputs=[
                disc_center_wp,
                disc_normal_wp,
                float(disc_radius),
                float(width),
                float(terrain.mu),
                pos_x,
                pos_y,
                pos_z,
                float(size_x),
                float(size_y),
                float(size_z_top),
                int(terrain.nrow),
                int(terrain.ncol),
                hdata,
            ],
            outputs=[in_contact_out, depth_out, mu_out, pos_out, x_axis_out, y_axis_out, z_axis_out],
            device=device,
        )
    elif method == CollisionType.ENVELOPE:
        area_dep = ChFunctionInterp()
        ConstructAreaDepthTable(float(disc_radius), area_dep)
        pairs = area_dep.table
        area_xs = wp.array([p[0] for p in pairs], dtype=float, device=device)
        area_ys = wp.array([p[1] for p in pairs], dtype=float, device=device)

        wp.launch(
            _disc_collision_envelope_hfield_kernel,
            dim=n,
            inputs=[
                disc_center_wp,
                disc_normal_wp,
                float(disc_radius),
                float(width),
                area_xs,
                area_ys,
                int(len(pairs)),
                float(terrain.mu),
                pos_x,
                pos_y,
                pos_z,
                float(size_x),
                float(size_y),
                float(size_z_top),
                int(terrain.nrow),
                int(terrain.ncol),
                hdata,
            ],
            outputs=[in_contact_out, depth_out, mu_out, pos_out, x_axis_out, y_axis_out, z_axis_out],
            device=device,
        )
    else:
        raise ValueError(f"Unsupported collision method: {method}")

    return {
        "in_contact": [bool(x) for x in in_contact_out.numpy().tolist()],
        "depth": depth_out.numpy().tolist(),
        "mu": mu_out.numpy().tolist(),
        "pos": pos_out.numpy().tolist(),
        "x_axis": x_axis_out.numpy().tolist(),
        "y_axis": y_axis_out.numpy().tolist(),
        "z_axis": z_axis_out.numpy().tolist(),
    }
