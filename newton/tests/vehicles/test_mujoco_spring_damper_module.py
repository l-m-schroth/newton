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

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Keep Warp cache inside the repo by default (helps when running in sandboxed environments).
os.environ.setdefault("WARP_CACHE_PATH", str(_REPO_ROOT / ".warp_cache"))

import warp as wp

# Import mujoco_warp as a package (not the workspace namespace dir).
sys.path.insert(0, str(_REPO_ROOT / "mujoco_warp"))
import mujoco_warp  # noqa: E402

import newton  # noqa: E402
from newton._src.vehicle.mujoco_spring_damper_module import MujocoSpringDamperModule, SpringDamper  # noqa: E402
from newton._src.vehicle.tires import MujocoFialaTireModule  # noqa: E402
from newton._src.vehicle.tires.disc_terrain_collision import CollisionType  # noqa: E402
from newton.solvers import SolverMuJoCo  # noqa: E402


def _chrono_tire_path(rel: str) -> str:
    return str(_REPO_ROOT / "chrono" / "build" / "data" / rel)


def _make_dummy_newton_model() -> newton.Model:
    builder = newton.ModelBuilder()
    link = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), I_m=wp.mat33(np.eye(3)))
    joint = builder.add_joint_revolute(-1, link)
    builder.add_articulation([joint])
    return builder.finalize()


def _save_mass_trajectory_plot(path: Path, t: np.ndarray, xyz: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.plot(t, xyz[:, 0, i], label=f"world0 {labels[i]}")
        ax.plot(t, xyz[:, 1, i], label=f"world1 {labels[i]}")
        ax.set_ylabel(f"{labels[i]} (m)")
        ax.grid(True)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class TestMujocoSpringDamperModule(unittest.TestCase):
    def tearDown(self) -> None:
        mujoco_warp.mj_resetCallbacks()

    def test_mass_on_vertical_spring_converges_to_static_equilibrium_nworld_2(self):
        xml = r"""
<mujoco model="mass_on_spring">
  <option timestep="0.001" integrator="Euler" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="mass" pos="0 0 0.8">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""
        mjm = mujoco.MjModel.from_xml_string(xml)
        mjd = mujoco.MjData(mjm)

        m = mujoco_warp.put_model(mjm)
        d = mujoco_warp.put_data(mjm, mjd, nworld=2)

        body_mass = float(mjm.body_mass[1])
        g = abs(float(mjm.opt.gravity[2]))
        k = 1000.0
        c = 20.0
        z_anchor = 1.0
        L0 = 0.4

        # Equilibrium: k*(L-L0) = m*g, L = z_anchor - z_eq.
        z_eq = z_anchor - (L0 + body_mass * g / k)

        qpos = d.qpos.numpy()
        qpos[0, 2] = z_eq + 0.2
        qpos[1, 2] = z_eq - 0.1
        d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

        body_id = int(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "mass"))

        module = MujocoSpringDamperModule(
            springs=[
                SpringDamper(
                    body0=0,
                    body1=body_id,
                    p0_local=(0.0, 0.0, z_anchor),
                    p1_local=(0.0, 0.0, 0.0),
                    rest_length=L0,
                    stiffness=k,
                    damping=c,
                )
            ]
        )

        solver = SolverMuJoCo(_make_dummy_newton_model(), mjw_model=m, mjw_data=d, update_data_interval=0)
        solver.add_spring_damper_modules([module])

        dt = float(mjm.opt.timestep)
        steps = 3000
        decimate = 10

        times: list[float] = []
        xyz_hist: list[np.ndarray] = []

        for step in range(steps):
            mujoco_warp.step(m, d)
            if step % decimate == 0:
                times.append(step * dt)
                xyz_hist.append(d.xipos.numpy()[:, body_id].copy())

        t = np.asarray(times)
        xyz = np.asarray(xyz_hist)

        plot_dir = Path(__file__).resolve().parent / "spring_damper_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        _save_mass_trajectory_plot(plot_dir / "mass_on_vertical_spring_nworld2.png", t, xyz)

        final_xyz = d.xipos.numpy()[:, body_id]
        for world in range(2):
            self.assertAlmostEqual(float(final_xyz[world, 0]), 0.0, places=3)
            self.assertAlmostEqual(float(final_xyz[world, 1]), 0.0, places=3)
            self.assertAlmostEqual(float(final_xyz[world, 2]), z_eq, delta=2.0e-3)

    def test_spring_damper_chains_with_tire_module_and_no_contact_forces(self):
        tire_json = _chrono_tire_path("vehicle/generic/tire/FialaTire.json")

        xml = r"""
<mujoco model="wheel_on_spring_no_contact">
  <option timestep="0.001" integrator="Euler" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1" friction="0.8 0 0"/>
    <body name="wheel" pos="0 0 0.55">
      <freejoint/>
      <inertial pos="0 0 0" mass="10" diaginertia="1.0 1.0 1.0"/>
    </body>
  </worldbody>
</mujoco>
"""
        mjm = mujoco.MjModel.from_xml_string(xml)
        mjd = mujoco.MjData(mjm)

        m = mujoco_warp.put_model(mjm)
        d = mujoco_warp.put_data(mjm, mjd, nworld=2)

        wheel_body_id = int(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "wheel"))

        body_mass = float(mjm.body_mass[wheel_body_id])
        g = abs(float(mjm.opt.gravity[2]))
        k = 10000.0
        c = 50.0
        z_anchor = 1.0
        z_eq = 0.55
        L0 = (z_anchor - z_eq) - body_mass * g / k

        qpos = d.qpos.numpy()
        qpos[:, 2] = z_eq
        d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

        spring = MujocoSpringDamperModule(
            springs=[
                SpringDamper(
                    body0=0,
                    body1=wheel_body_id,
                    p0_local=(0.0, 0.0, z_anchor),
                    p1_local=(0.0, 0.0, 0.0),
                    rest_length=L0,
                    stiffness=k,
                    damping=c,
                )
            ]
        )

        tire = MujocoFialaTireModule.from_mujoco_names(
            mjm,
            tire_json,
            wheel_body_names=["wheel"],
            terrain_geom_name="ground",
            collision_type=CollisionType.SINGLE_POINT,
        )

        solver = SolverMuJoCo(_make_dummy_newton_model(), mjw_model=m, mjw_data=d, update_data_interval=0)
        solver.add_tire_modules([tire])
        solver.add_spring_damper_modules([spring])

        any_contact = False
        for _ in range(1000):
            mujoco_warp.step(m, d)
            in_contact = tire._in_contact  # noqa: SLF001
            self.assertIsNotNone(in_contact)
            if int(in_contact.numpy().max()) != 0:
                any_contact = True
                break

        wheel_z = d.xipos.numpy()[:, wheel_body_id, 2]
        self.assertAlmostEqual(float(wheel_z[0]), z_eq, delta=2.0e-3)
        self.assertAlmostEqual(float(wheel_z[1]), z_eq, delta=2.0e-3)
        self.assertFalse(any_contact, "wheel should remain above the ground (no disc-terrain contact)")
