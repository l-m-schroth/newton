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

import os
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Keep Warp cache inside the repo by default (helps when running in sandboxed environments).
os.environ.setdefault("WARP_CACHE_PATH", str(_REPO_ROOT / ".warp_cache"))

import warp as wp

# Import mujoco_warp as a package (not the workspace namespace dir).
sys.path.insert(0, str(_REPO_ROOT / "mujoco_warp"))
import mujoco_warp  # noqa: E402

import newton  # noqa: E402
from newton._src.vehicle.mujoco_anti_roll_bar_module import MujocoAntiRollBarModule  # noqa: E402
from newton.solvers import SolverMuJoCo  # noqa: E402

if wp.config.kernel_cache_dir != os.environ["WARP_CACHE_PATH"]:
    Path(os.environ["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]


def _build_suspension_world() -> newton.ModelBuilder:
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

    I = wp.mat33(np.eye(3))
    m = 1.0

    link_fl = b.add_link(mass=m, com=wp.vec3(0.0, 0.0, 0.0), I_m=I)
    j_fl = b.add_joint_prismatic(-1, link_fl, axis=wp.vec3(0.0, 0.0, 1.0), key="q_FL")
    b.add_articulation([j_fl])
    b.add_shape_sphere(body=link_fl, radius=0.01, as_site=True, key="site_FL")

    link_fr = b.add_link(mass=m, com=wp.vec3(0.0, 0.0, 0.0), I_m=I)
    j_fr = b.add_joint_prismatic(-1, link_fr, axis=wp.vec3(0.0, 0.0, 1.0), key="q_FR")
    b.add_articulation([j_fr])
    b.add_shape_sphere(body=link_fr, radius=0.01, as_site=True, key="site_FR")

    return b


def _save_dof_trajectory_plot(path: Path, t: np.ndarray, q_fl: np.ndarray, q_fr: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t, q_fl[:, 0], label="world0 q_FL")
    ax.plot(t, q_fr[:, 0], label="world0 q_FR")
    ax.plot(t, q_fl[:, 1], label="world1 q_FL")
    ax.plot(t, q_fr[:, 1], label="world1 q_FR")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Suspension travel q (m)")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class TestMujocoAntiRollBarModule(unittest.TestCase):
    def tearDown(self) -> None:
        mujoco_warp.mj_resetCallbacks()

    def test_static_anti_roll_equilibrium_nworld_2(self):
        # Two independent 1-DOF suspension coordinates (no gravity, no contacts). The anti-roll bar applies equal/opposite
        # generalized forces, so the sum coordinate (q_FL + q_FR) should remain constant over time.
        a = 0.2
        k = 200.0
        b = 20.0

        main = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        w0 = _build_suspension_world()
        w1 = _build_suspension_world()
        main.add_world(w0)
        main.add_world(w1)
        model = main.finalize()

        nworld = int(model.num_worlds)
        self.assertEqual(nworld, 2)

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        contacts = model.collide(state_0)

        joint_world = model.joint_world.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        def _joint_idx(world: int, key: str) -> int:
            for ji in np.where(joint_world == world)[0]:
                if model.joint_key[int(ji)] == key:
                    return int(ji)
            raise AssertionError(f"joint {key!r} not found in world {world}")

        j_fl_w0 = _joint_idx(0, "q_FL")
        j_fr_w0 = _joint_idx(0, "q_FR")
        j_fl_w1 = _joint_idx(1, "q_FL")
        j_fr_w1 = _joint_idx(1, "q_FR")

        q = state_0.joint_q.numpy()
        qd = state_0.joint_qd.numpy()

        q[int(joint_q_start[j_fl_w0])] = +a
        q[int(joint_q_start[j_fr_w0])] = 0.0
        q[int(joint_q_start[j_fl_w1])] = -a
        q[int(joint_q_start[j_fr_w1])] = 0.0

        qd[int(joint_qd_start[j_fl_w0])] = 0.0
        qd[int(joint_qd_start[j_fr_w0])] = 0.0
        qd[int(joint_qd_start[j_fl_w1])] = 0.0
        qd[int(joint_qd_start[j_fr_w1])] = 0.0

        state_0.joint_q = wp.array(q, dtype=float, device=state_0.joint_q.device)
        state_0.joint_qd = wp.array(qd, dtype=float, device=state_0.joint_qd.device)

        solver = SolverMuJoCo(model, use_mujoco_contacts=True, integrator="euler", update_data_interval=0)
        module = MujocoAntiRollBarModule.from_mujoco_names(
            solver.mj_model,
            joint_left_name="q_FL",
            joint_right_name="q_FR",
            rest_diff=0.0,
            stiffness=k,
            damping=b,
        )
        solver.add_anti_roll_bar_modules([module])

        dt = 0.002
        steps = int(4.0 / dt)
        decimate = 5

        times: list[float] = []
        q_fl_hist: list[np.ndarray] = []
        q_fr_hist: list[np.ndarray] = []

        for step in range(steps):
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

            if step % decimate == 0:
                times.append(step * dt)
                qn = state_0.joint_q.numpy()
                q_fl_hist.append(
                    np.array(
                        [
                            float(qn[int(joint_q_start[j_fl_w0])]),
                            float(qn[int(joint_q_start[j_fl_w1])]),
                        ],
                        dtype=np.float64,
                    )
                )
                q_fr_hist.append(
                    np.array(
                        [
                            float(qn[int(joint_q_start[j_fr_w0])]),
                            float(qn[int(joint_q_start[j_fr_w1])]),
                        ],
                        dtype=np.float64,
                    )
                )

        t = np.asarray(times)
        q_fl = np.asarray(q_fl_hist)  # [T, nworld]
        q_fr = np.asarray(q_fr_hist)

        plot_dir = Path(__file__).resolve().parent / "anti_roll_bar_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        _save_dof_trajectory_plot(plot_dir / "anti_roll_bar_equilibrium_nworld2.png", t, q_fl, q_fr)

        l_final = q_fl[-1, :] - q_fr[-1, :]
        self.assertLess(abs(float(l_final[0])), 2.0e-3)
        self.assertLess(abs(float(l_final[1])), 2.0e-3)

        qsum = q_fl + q_fr
        qsum0 = qsum[0, :]
        max_dev = np.max(np.abs(qsum - qsum0[None, :]), axis=0)
        self.assertLess(float(max_dev[0]), 2.0e-3)
        self.assertLess(float(max_dev[1]), 2.0e-3)
