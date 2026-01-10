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
class AntiRollBar:
    """Specification for a single anti-roll bar acting on two 1-DOF suspension coordinates.

    The anti-roll bar operates on the difference

        l(q) = q_left - q_right

    and applies equal/opposite generalized forces on the corresponding DOFs:

        p = -k (l - l0) - b l_dot
        tau_left  += p
        tau_right -= p

    Indices are given in MuJoCo coordinate space:
    - `qpos_left` / `qpos_right` index into `qpos`
    - `dof_left` / `dof_right` index into `qvel` and `qfrc_applied`
    """

    qpos_left: int
    dof_left: int
    qpos_right: int
    dof_right: int
    rest_diff: float
    stiffness: float
    damping: float


@wp.kernel
def _mujoco_anti_roll_bar_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    qpos_left: wp.array(dtype=int),
    dof_left: wp.array(dtype=int),
    qpos_right: wp.array(dtype=int),
    dof_right: wp.array(dtype=int),
    rest_diff: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    damping: wp.array(dtype=float),
    nbars: int,
    qfrc_applied_io: wp.array2d(dtype=float),
):
    tid = wp.tid()

    worldid = tid // nbars
    bar_idx = tid - worldid * nbars

    qli = qpos_left[bar_idx]
    qri = qpos_right[bar_idx]
    dli = dof_left[bar_idx]
    dri = dof_right[bar_idx]

    ql = qpos[worldid, qli]
    qr = qpos[worldid, qri]
    l = ql - qr

    qdl = qvel[worldid, dli]
    qdr = qvel[worldid, dri]
    ldot = qdl - qdr

    p = -stiffness[bar_idx] * (l - rest_diff[bar_idx]) - damping[bar_idx] * ldot

    wp.atomic_add(qfrc_applied_io, worldid, dli, p)
    wp.atomic_add(qfrc_applied_io, worldid, dri, -p)


def _resolve_1dof_joint_qpos_dof(mj_model, joint_id: int, *, joint_name: str) -> tuple[int, int]:
    import mujoco  # noqa: PLC0415

    if joint_id < 0 or joint_id >= int(mj_model.njnt):
        raise ValueError(f"Invalid joint id {joint_id} for {joint_name!r}")

    jtype = int(mj_model.jnt_type[joint_id])
    if jtype not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError(
            f"Anti-roll bar requires 1-DOF hinge/slide joints; got joint type {jtype} for {joint_name!r}."
        )

    qpos_idx = int(mj_model.jnt_qposadr[joint_id])
    dof_idx = int(mj_model.jnt_dofadr[joint_id])

    next_qpos = int(mj_model.nq)
    next_dof = int(mj_model.nv)
    if joint_id + 1 < int(mj_model.njnt):
        next_qpos = int(mj_model.jnt_qposadr[joint_id + 1])
        next_dof = int(mj_model.jnt_dofadr[joint_id + 1])

    if next_qpos - qpos_idx != 1 or next_dof - dof_idx != 1:
        raise ValueError(
            f"Anti-roll bar requires scalar joints; got qpos_dim={next_qpos - qpos_idx}, dof_dim={next_dof - dof_idx} "
            f"for {joint_name!r}."
        )

    return qpos_idx, dof_idx


@dataclasses.dataclass(slots=True)
class MujocoAntiRollBarModule:
    """Anti-roll bar force elements applied via MuJoCo-Warp `mjcb_control`.

    The module accumulates generalized forces into `d.qfrc_applied` (MuJoCo(-Warp) generalized forces) so it can be
    chained with other force modules as long as the callback restores the baseline applied-force buffers each invocation.
    """

    bars: Sequence[AntiRollBar]

    _device_key: str | None = None
    _qpos_left_wp: wp.array | None = None
    _dof_left_wp: wp.array | None = None
    _qpos_right_wp: wp.array | None = None
    _dof_right_wp: wp.array | None = None
    _rest_wp: wp.array | None = None
    _k_wp: wp.array | None = None
    _b_wp: wp.array | None = None

    @classmethod
    def from_mujoco_names( 
        # NOTE: from_mujoco_names currently only supports creating modules with a single ARB. 
        # The kernel can handle multiple ARB's in the same module. Hence from_mujoco_names could be generalized to multi ARB support.
        cls,
        mj_model,
        *,
        joint_left_name: str,
        joint_right_name: str,
        rest_diff: float = 0.0,
        stiffness: float,
        damping: float,
    ) -> "MujocoAntiRollBarModule":
        import mujoco  # noqa: PLC0415

        jl = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_left_name)))
        if jl < 0:
            raise ValueError(f"Unknown MuJoCo joint name: {joint_left_name!r}")
        jr = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_right_name)))
        if jr < 0:
            raise ValueError(f"Unknown MuJoCo joint name: {joint_right_name!r}")

        qpos_l, dof_l = _resolve_1dof_joint_qpos_dof(mj_model, jl, joint_name=str(joint_left_name))
        qpos_r, dof_r = _resolve_1dof_joint_qpos_dof(mj_model, jr, joint_name=str(joint_right_name))

        return cls( 
            bars=(
                AntiRollBar(
                    qpos_left=qpos_l,
                    dof_left=dof_l,
                    qpos_right=qpos_r,
                    dof_right=dof_r,
                    rest_diff=float(rest_diff),
                    stiffness=float(stiffness),
                    damping=float(damping),
                ),
            )
        )

    def _ensure_device_arrays(self, device: wp.context.Device) -> None:
        device_key = str(device)
        if self._device_key == device_key and self._qpos_left_wp is not None:
            return
        self._device_key = device_key

        bars = list(self.bars)
        self._qpos_left_wp = wp.array([int(b.qpos_left) for b in bars], dtype=wp.int32, device=device)
        self._dof_left_wp = wp.array([int(b.dof_left) for b in bars], dtype=wp.int32, device=device)
        self._qpos_right_wp = wp.array([int(b.qpos_right) for b in bars], dtype=wp.int32, device=device)
        self._dof_right_wp = wp.array([int(b.dof_right) for b in bars], dtype=wp.int32, device=device)
        self._rest_wp = wp.array([float(b.rest_diff) for b in bars], dtype=float, device=device)
        self._k_wp = wp.array([float(b.stiffness) for b in bars], dtype=float, device=device)
        self._b_wp = wp.array([float(b.damping) for b in bars], dtype=float, device=device)

    def apply(self, m, d) -> None:  # noqa: ARG002
        wp.init()
        if not self.bars:
            return

        device = d.qfrc_applied.device
        self._ensure_device_arrays(device)

        nbars = len(self.bars)
        n = int(d.nworld) * int(nbars)
        if n == 0:
            return

        assert self._qpos_left_wp is not None
        assert self._dof_left_wp is not None
        assert self._qpos_right_wp is not None
        assert self._dof_right_wp is not None
        assert self._rest_wp is not None
        assert self._k_wp is not None
        assert self._b_wp is not None

        wp.launch(
            _mujoco_anti_roll_bar_kernel,
            dim=n,
            inputs=[
                d.qpos,
                d.qvel,
                self._qpos_left_wp,
                self._dof_left_wp,
                self._qpos_right_wp,
                self._dof_right_wp,
                self._rest_wp,
                self._k_wp,
                self._b_wp,
                int(nbars),
            ],
            outputs=[d.qfrc_applied],
            device=device,
        )

