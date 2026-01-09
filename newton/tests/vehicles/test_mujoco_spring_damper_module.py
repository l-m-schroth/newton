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


def _build_world_from_mjcf(xml: str, *, gravity: float) -> newton.ModelBuilder:
    """Parse a minimal MJCF into a Newton world builder.

    We set `ignore_inertial_definitions=False` because these MJCF snippets intentionally omit collision geometry and
    rely on explicit `<inertial ...>` tags to define mass/inertia.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-float(gravity))
    b.add_mjcf(
        xml,
        ignore_inertial_definitions=False,
        ensure_nonstatic_links=False,
        parse_sites=True,
        parse_visuals=False,
        parse_meshes=False,
    )
    return b


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
        gravity = 9.81
        body_mass = 1.0
        g = gravity
        k = 1000.0
        c = 20.0
        z_anchor = 1.0
        L0 = 0.4

        # Equilibrium: k*(L-L0) = m*g, L = z_anchor - z_eq.
        z_eq = z_anchor - (L0 + body_mass * g / k)

        xml0 = f"""
<mujoco model="mass_on_spring">
  <option timestep="0.001" integrator="Euler" gravity="0 0 -{gravity}"/>
  <worldbody>
    <body name="mass" pos="0 0 {z_eq + 0.2}">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""
        xml1 = f"""
<mujoco model="mass_on_spring">
  <option timestep="0.001" integrator="Euler" gravity="0 0 -{gravity}"/>
  <worldbody>
    <body name="mass" pos="0 0 {z_eq - 0.1}">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

        main = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-float(gravity))
        w0 = _build_world_from_mjcf(xml0, gravity=gravity)
        w1 = _build_world_from_mjcf(xml1, gravity=gravity)
        # Add a world-local site to ensure `separate_worlds=True` has a per-world shape for selecting a template world.
        for w in (w0, w1):
            if "mass" in w.body_key:
                mass_idx = w.body_key.index("mass")
                w.add_shape_sphere(body=mass_idx, radius=0.01, as_site=True, key="mass_site")
        main.add_world(w0)
        main.add_world(w1)
        model = main.finalize()
        nworld = int(model.num_worlds)
        body_world = model.body_world.numpy()
        body_indices = [int(np.nonzero(body_world == w)[0][0]) for w in range(nworld)]
        self.assertEqual(len(body_indices), 2)

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        contacts = model.collide(state_0)

        solver = SolverMuJoCo(model, use_mujoco_contacts=True, integrator="euler", update_data_interval=0)
        body_id = int(mujoco.mj_name2id(solver.mj_model, mujoco.mjtObj.mjOBJ_BODY, "mass"))

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

        solver.add_spring_damper_modules([module])

        dt = 0.001
        steps = 3000
        decimate = 10

        times: list[float] = []
        xyz_hist: list[np.ndarray] = []

        for step in range(steps):
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
            if step % decimate == 0:
                times.append(step * dt)
                body_q = state_0.body_q.numpy()
                xyz_world = np.zeros((nworld, 3), dtype=np.float64)
                for w in range(nworld):
                    bi = body_indices[w]
                    xyz_world[w, :] = body_q[bi, 0:3]
                xyz_hist.append(xyz_world)

        t = np.asarray(times)
        xyz = np.asarray(xyz_hist)

        plot_dir = Path(__file__).resolve().parent / "spring_damper_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        _save_mass_trajectory_plot(plot_dir / "mass_on_vertical_spring_nworld2.png", t, xyz)

        body_q = state_0.body_q.numpy()
        final_xyz = np.zeros((nworld, 3), dtype=np.float64)
        for w in range(nworld):
            bi = body_indices[w]
            final_xyz[w, :] = body_q[bi, 0:3]
        for world in range(2):
            self.assertAlmostEqual(float(final_xyz[world, 0]), 0.0, places=3)
            self.assertAlmostEqual(float(final_xyz[world, 1]), 0.0, places=3)
            self.assertAlmostEqual(float(final_xyz[world, 2]), z_eq, delta=2.0e-3)

    def test_spring_damper_chains_with_tire_module_and_no_contact_forces(self):
        tire_json = _chrono_tire_path("vehicle/generic/tire/FialaTire.json")
        gravity = 9.81
        g = gravity
        k = 10000.0
        c = 50.0
        z_anchor = 1.0
        z_eq = 0.55
        body_mass = 10.0
        L0 = (z_anchor - z_eq) - body_mass * g / k

        xml = f"""
<mujoco model="wheel_on_spring_no_contact">
  <option timestep="0.001" integrator="Euler" gravity="0 0 -{gravity}"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1" friction="0.8 0 0"/>
    <body name="wheel" pos="0 0 {z_eq}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{body_mass}" diaginertia="1.0 1.0 1.0"/>
    </body>
  </worldbody>
</mujoco>
"""
        main = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-float(gravity))
        main.add_world(_build_world_from_mjcf(xml, gravity=gravity))
        main.add_world(_build_world_from_mjcf(xml, gravity=gravity))
        model = main.finalize()
        nworld = int(model.num_worlds)
        body_world = model.body_world.numpy()
        body_indices = [int(np.nonzero(body_world == w)[0][0]) for w in range(nworld)]
        self.assertEqual(len(body_indices), 2)

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        contacts = model.collide(state_0)
        solver = SolverMuJoCo(model, use_mujoco_contacts=True, integrator="euler", update_data_interval=0)

        wheel_body_id = int(mujoco.mj_name2id(solver.mj_model, mujoco.mjtObj.mjOBJ_BODY, "wheel"))
        terrain_geom_name = "ground"
        for gid in range(int(solver.mj_model.ngeom)):
            if int(solver.mj_model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                name = mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if name is not None:
                    terrain_geom_name = str(name)
                break

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
            solver.mj_model,
            tire_json,
            wheel_body_names=["wheel"],
            terrain_geom_name=terrain_geom_name,
            collision_type=CollisionType.SINGLE_POINT,
        )

        solver.add_tire_modules([tire])
        solver.add_spring_damper_modules([spring])

        any_contact = False
        for _ in range(1000):
            solver.step(state_0, state_1, control, contacts, 0.001)
            state_0, state_1 = state_1, state_0
            in_contact = tire._in_contact  # noqa: SLF001
            self.assertIsNotNone(in_contact)  # NOTE (Lukas): Make sure that the tire module actually allocates in_contact
            if int(in_contact.numpy().max()) != 0:
                any_contact = True
                break

        body_q = state_0.body_q.numpy()
        wheel_z = np.zeros((nworld,), dtype=np.float64)
        for w in range(nworld):
            bi = body_indices[w]
            wheel_z[w] = float(body_q[bi, 2])
        self.assertAlmostEqual(float(wheel_z[0]), z_eq, delta=2.0e-3)
        self.assertAlmostEqual(float(wheel_z[1]), z_eq, delta=2.0e-3)
        self.assertFalse(any_contact, "wheel should remain above the ground (no disc-terrain contact)")
