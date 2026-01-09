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
from typing import Sequence

import warp as wp


@dataclasses.dataclass(frozen=True, slots=True)
class SpringDamper:
    """Specification for a single spring-damper connecting two bodies.

    The attachment points `p0_local` and `p1_local` are expressed in each body's local frame with the origin located at
    the body's center of mass (COM). The applied forces are mapped to a world-frame wrench at the COM and accumulated
    into MuJoCo-Warp `d.xfrc_applied`.
    """

    body0: int
    body1: int
    p0_local: tuple[float, float, float]
    p1_local: tuple[float, float, float]
    rest_length: float
    stiffness: float
    damping: float


@wp.kernel
def _mujoco_spring_damper_kernel(
    xmat: wp.array2d(dtype=wp.mat33),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    body_rootid: wp.array(dtype=int),
    subtree_com: wp.array2d(dtype=wp.vec3),
    xipos: wp.array2d(dtype=wp.vec3),
    body0_ids: wp.array(dtype=int),
    body1_ids: wp.array(dtype=int),
    p0_local: wp.array(dtype=wp.vec3),
    p1_local: wp.array(dtype=wp.vec3),
    rest_length: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    damping: wp.array(dtype=float),
    nsprings: int,
    xfrc_applied_io: wp.array2d(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    worldid = tid // nsprings
    spring_idx = tid - worldid * nsprings

    b0 = body0_ids[spring_idx]
    b1 = body1_ids[spring_idx]
    if b0 < 0 or b1 < 0 or b0 == b1:
        return

    com0 = xipos[worldid, b0]
    com1 = xipos[worldid, b1]

    R0 = xmat[worldid, b0]
    R1 = xmat[worldid, b1]

    r0 = R0 @ p0_local[spring_idx]
    r1 = R1 @ p1_local[spring_idx]

    p0_world = com0 + r0
    p1_world = com1 + r1

    dpos = p1_world - p0_world
    L = wp.length(dpos)
    if L < 1.0e-8:
        return

    e = dpos / L
    x = L - rest_length[spring_idx]

    v0 = wp.vec3(0.0, 0.0, 0.0)
    v1 = wp.vec3(0.0, 0.0, 0.0)

    if b0 != 0:
        w0 = cvel[worldid, b0]
        omega0 = wp.spatial_top(w0)
        vel_root0 = wp.spatial_bottom(w0) # velocity w.r.t subtree COM
        root0 = body_rootid[b0]
        subtree0 = subtree_com[worldid, root0]
        v0 = vel_root0 + wp.cross(omega0, p0_world - subtree0)

    if b1 != 0:
        w1 = cvel[worldid, b1]
        omega1 = wp.spatial_top(w1)
        vel_root1 = wp.spatial_bottom(w1)
        root1 = body_rootid[b1]
        subtree1 = subtree_com[worldid, root1]
        v1 = vel_root1 + wp.cross(omega1, p1_world - subtree1)

    xdot = wp.dot(v1 - v0, e)
    f_mag = -stiffness[spring_idx] * x - damping[spring_idx] * xdot

    f1 = f_mag * e
    f0 = -f1

    t0 = wp.cross(r0, f0)
    t1 = wp.cross(r1, f1)

    if b0 != 0:
        wp.atomic_add(xfrc_applied_io, worldid, b0, wp.spatial_vector(f0, t0))
    if b1 != 0:
        wp.atomic_add(xfrc_applied_io, worldid, b1, wp.spatial_vector(f1, t1))


@dataclasses.dataclass(slots=True)
class MujocoSpringDamperModule:
    """Spring-damper force elements applied via MuJoCo-Warp `mjcb_control`.
    Note that in Mujoco, typically tendons can be used to directly get spring-damper forces between different body sites.
    However, as tendons are not really supported in the newton modelBuilder yet, I decided to add this module. 
    Also, we anyways might want to improve the spring-damper forces with non-linear spring laws later, 
    so adding the forces via a custom model allows us to easily improve the simulation by replacing the spring law. 
    Also, Mujoco tendons damping are not properly considered during implicit integration anyways (only joint damping is considered), so it probably does not make a numerical difference
    to using mujoco's build in tendons (Reference: https://mujoco.readthedocs.io/en/stable/computation/index.html)

    The module accumulates applied wrenches into `d.xfrc_applied` (world-frame force/torque), so it can be chained with
    other force modules (e.g., tires) as long as the callback clears the applied-force buffers each invocation.
    """

    springs: Sequence[SpringDamper]

    _device_key: str | None = None
    _body0_ids_wp: wp.array | None = None
    _body1_ids_wp: wp.array | None = None
    _p0_local_wp: wp.array | None = None
    _p1_local_wp: wp.array | None = None
    _rest_wp: wp.array | None = None
    _k_wp: wp.array | None = None
    _c_wp: wp.array | None = None

    def _ensure_device_arrays(self, device: wp.context.Device) -> None:
        device_key = str(device)
        if self._device_key == device_key and self._body0_ids_wp is not None:
            return
        self._device_key = device_key

        springs = list(self.springs)
        self._body0_ids_wp = wp.array([int(s.body0) for s in springs], dtype=wp.int32, device=device)
        self._body1_ids_wp = wp.array([int(s.body1) for s in springs], dtype=wp.int32, device=device)
        self._p0_local_wp = wp.array([wp.vec3(*s.p0_local) for s in springs], dtype=wp.vec3, device=device)
        self._p1_local_wp = wp.array([wp.vec3(*s.p1_local) for s in springs], dtype=wp.vec3, device=device)
        self._rest_wp = wp.array([float(s.rest_length) for s in springs], dtype=float, device=device)
        self._k_wp = wp.array([float(s.stiffness) for s in springs], dtype=float, device=device)
        self._c_wp = wp.array([float(s.damping) for s in springs], dtype=float, device=device)

    def apply(self, m, d) -> None:
        wp.init()
        if not self.springs:
            return

        device = d.xfrc_applied.device
        self._ensure_device_arrays(device)

        nsprings = len(self.springs)
        n = int(d.nworld) * int(nsprings)
        if n == 0:
            return

        assert self._body0_ids_wp is not None
        assert self._body1_ids_wp is not None
        assert self._p0_local_wp is not None
        assert self._p1_local_wp is not None
        assert self._rest_wp is not None
        assert self._k_wp is not None
        assert self._c_wp is not None

        wp.launch(
            _mujoco_spring_damper_kernel,
            dim=n,
            inputs=[
                d.xmat,
                d.cvel,
                m.body_rootid,
                d.subtree_com,
                d.xipos,
                self._body0_ids_wp,
                self._body1_ids_wp,
                self._p0_local_wp,
                self._p1_local_wp,
                self._rest_wp,
                self._k_wp,
                self._c_wp,
                int(nsprings),
            ],
            outputs=[d.xfrc_applied],
            device=device,
        )
