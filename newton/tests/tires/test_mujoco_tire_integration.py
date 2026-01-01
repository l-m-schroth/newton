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

import math
import os
import unittest
from pathlib import Path

import sys

import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Keep Warp cache inside the repo by default (helps when running in sandboxed environments).
os.environ.setdefault("WARP_CACHE_PATH", str(_REPO_ROOT / ".warp_cache"))

import warp as wp

# Import mujoco_warp as a package (not the workspace namespace dir).
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mujoco_warp"))
import mujoco_warp  # noqa: E402

from newton._src.vehicle.tires import MujocoFialaTireModule
from newton._src.vehicle.tires.disc_terrain_collision import CollisionType, HFieldTerrain
from newton.tests.tires.chrono_gt import run_chrono_gt


def _repo_root() -> Path:
    return _REPO_ROOT


if wp.config.kernel_cache_dir != os.environ["WARP_CACHE_PATH"]:
    Path(os.environ["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]


def _chrono_tire_path(rel: str) -> str:
    return str(_repo_root() / "chrono" / "build" / "data" / rel)


def _assert_close(testcase: unittest.TestCase, actual: float, expected: float, *, rtol: float, atol: float) -> None:
    err = abs(actual - expected)
    tol = atol + rtol * abs(expected)
    testcase.assertLessEqual(err, tol, msg=f"actual={actual} expected={expected} err={err} tol={tol}")


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_mul(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _chronos_force_wrench_world(
    *,
    tire_json: str,
    collision_type: CollisionType,
    disc_center: tuple[float, float, float],
    disc_normal: tuple[float, float, float],
    vel_world: tuple[float, float, float],
    omega_world: tuple[float, float, float],
    wheel_center: tuple[float, float, float],
    terrain: dict[str, object],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute Chrono tire wrench at wheel center using the GT CLI subfunctions."""
    params = run_chrono_gt({"cmd": "get_params", "tire_json": tire_json})
    if not params["ok"]:
        raise RuntimeError(params.get("error", "get_params failed"))

    disc_radius = float(params["unloaded_radius"])
    width = float(params["width"])

    gt = run_chrono_gt(
        {
            "cmd": "disc_terrain_collision",
            "collision_type": collision_type.name.lower(),
            "disc_center": list(disc_center),
            "disc_normal": list(disc_normal),
            "disc_radius": disc_radius,
            "width": width,
            "terrain": terrain,
        }
    )
    if not gt["ok"]:
        raise RuntimeError(gt.get("error", "disc_terrain_collision failed"))

    if not gt["in_contact"]:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    depth = float(gt["depth"])
    mu = float(gt["mu"])
    contact = gt["contact"]
    x_axis = tuple(float(x) for x in contact["x_axis"])
    y_axis = tuple(float(x) for x in contact["y_axis"])
    z_axis = tuple(float(x) for x in contact["z_axis"])
    contact_pos = tuple(float(x) for x in contact["pos"])

    vx = _dot(vel_world, x_axis)
    vy = _dot(vel_world, y_axis)
    vel_z = _dot(vel_world, z_axis)
    omega = _dot(omega_world, disc_normal)

    normal = run_chrono_gt({"cmd": "normal_load", "tire_json": tire_json, "depth": depth, "vel_z": vel_z})
    if not normal["ok"]:
        raise RuntimeError(normal.get("error", "normal_load failed"))
    fz = float(normal["fz"])

    slip = run_chrono_gt({"cmd": "fiala_slip", "v_x": vx, "v_y": vy, "omega": omega, "r_eff": disc_radius - depth})
    if not slip["ok"]:
        raise RuntimeError(slip.get("error", "fiala_slip failed"))
    kappa = float(slip["kappa"])
    alpha = float(slip["alpha"])

    patch = run_chrono_gt(
        {
            "cmd": "fiala_patch_forces",
            "tire_json": tire_json,
            "kappa": kappa,
            "alpha": alpha,
            "fz": fz,
            "mu": mu,
        }
    )
    if not patch["ok"]:
        raise RuntimeError(patch.get("error", "fiala_patch_forces failed"))
    fx = float(patch["fx"])
    fy = float(patch["fy"])
    mz = float(patch["mz"])

    rr = run_chrono_gt({"cmd": "rolling_resistance_moment", "tire_json": tire_json, "abs_vx": abs(vx), "fz": fz, "omega": omega})
    if not rr["ok"]:
        raise RuntimeError(rr.get("error", "rolling_resistance_moment failed"))
    my = float(rr["My"])

    force_world = _v_add(_v_add(_v_mul(x_axis, fx), _v_mul(y_axis, fy)), _v_mul(z_axis, fz))
    moment_world = _v_add(_v_mul(y_axis, my), _v_mul(z_axis, mz))

    r = _v_sub(_v_add(contact_pos, _v_mul(z_axis, depth)), wheel_center)
    moment_world = _v_add(moment_world, _cross(r, force_world))

    return force_world, moment_world


class TestMujocoWarpFialaTireIntegration(unittest.TestCase):
    def tearDown(self) -> None:
        mujoco_warp.mj_resetCallbacks()

    def test_rk4_calls_mjcb_control_4_times_per_step(self):
        xml = r"""
<mujoco model="rk4_cb_count">
  <option timestep="0.01" integrator="RK4" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1" friction="0.8 0 0"/>
    <body name="wheel" pos="0 0 0.45">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

        mjm = mujoco.MjModel.from_xml_string(xml)
        mjd = mujoco.MjData(mjm)
        m = mujoco_warp.put_model(mjm)
        d = mujoco_warp.put_data(mjm, mjd, nworld=1)

        calls = {"n": 0}

        def cb(m_cb, d_cb):  # noqa: ANN001
            calls["n"] += 1

        mujoco_warp.mjcb_control = cb
        mujoco_warp.step(m, d)
        self.assertEqual(calls["n"], 4)

    def test_plane_matches_chrono_for_nworld_2_all_collision_modes(self):
        tire_json = _chrono_tire_path("vehicle/generic/tire/FialaTire.json")

        xml = r"""
<mujoco model="plane_tire">
  <option timestep="0.01" integrator="Euler" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1" friction="0.8 0 0"/>
    <body name="wheel" pos="0 0 0.45">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

        mjm = mujoco.MjModel.from_xml_string(xml)
        mjd = mujoco.MjData(mjm)

        # Roll forward at ~1 m/s and ~rolling speed.
        mjd.qvel[0:6] = [1.0, 0.1, 0.0, 0.0, 2.0, 0.0]

        m = mujoco_warp.put_model(mjm)
        d = mujoco_warp.put_data(mjm, mjd, nworld=2)

        params = run_chrono_gt({"cmd": "get_params", "tire_json": tire_json})
        self.assertTrue(params["ok"])
        disc_radius = float(params["unloaded_radius"])

        qpos = d.qpos.numpy()
        # world 0: in contact (depth ~ 0.05)
        qpos[0, 2] = disc_radius - 0.05
        # world 1: no contact
        qpos[1, 2] = disc_radius + 0.05
        d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

        terrain = {"type": "plane", "height": 0.0, "mu": 0.8}
        wheel_id = int(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "wheel"))

        for method in [CollisionType.SINGLE_POINT, CollisionType.FOUR_POINTS, CollisionType.ENVELOPE]:
            with self.subTest(method=method):
                # Sentinel to ensure the callback overwrites (MuJoCo does not clear xfrc_applied between RK4 stages).
                d.xfrc_applied.fill_(wp.spatial_vector(123.0, -456.0, 789.0, 1.0, 2.0, 3.0))
                module = MujocoFialaTireModule.from_mujoco_names(
                    mjm,
                    tire_json,
                    wheel_body_names=["wheel"],
                    terrain_geom_name="ground",
                    collision_type=method,
                )
                mujoco_warp.mjcb_control = module.apply
                mujoco_warp.forward(m, d)

                xipos = d.xipos.numpy()
                xmat = d.xmat.numpy()
                cvel = d.cvel.numpy()
                xfrc = d.xfrc_applied.numpy()

                for worldid in range(2):
                    disc_center = tuple(float(x) for x in xipos[worldid, wheel_id])
                    disc_normal = (
                        float(xmat[worldid, wheel_id, 0, 1]),
                        float(xmat[worldid, wheel_id, 1, 1]),
                        float(xmat[worldid, wheel_id, 2, 1]),
                    )
                    omega_world = tuple(float(x) for x in cvel[worldid, wheel_id, 0:3])
                    vel_world = tuple(float(x) for x in cvel[worldid, wheel_id, 3:6])

                    exp_f, exp_m = _chronos_force_wrench_world(
                        tire_json=tire_json,
                        collision_type=method,
                        disc_center=disc_center,
                        disc_normal=disc_normal,
                        vel_world=vel_world,
                        omega_world=omega_world,
                        wheel_center=disc_center,
                        terrain=terrain,
                    )

                    act_f = tuple(float(x) for x in xfrc[worldid, wheel_id, 0:3])
                    act_m = tuple(float(x) for x in xfrc[worldid, wheel_id, 3:6])

                    if worldid == 0: # Make sure the contact scenario actually leads to contacts
                        self.assertGreater(exp_f[2], 0.0)
                        self.assertGreater(act_f[2], 0.0)
                    else: # no contact scenario should lead to zero contact forces
                        self.assertEqual(exp_f, (0.0, 0.0, 0.0))
                        self.assertEqual(exp_m, (0.0, 0.0, 0.0))
                        self.assertEqual(act_f, (0.0, 0.0, 0.0))
                        self.assertEqual(act_m, (0.0, 0.0, 0.0))

                    for i in range(3):
                        _assert_close(self, act_f[i], exp_f[i], rtol=2e-4, atol=5e-4)
                        _assert_close(self, act_m[i], exp_m[i], rtol=2e-4, atol=5e-4)

    def test_hfield_matches_chrono_for_single_point_nworld_2(self):
        tire_json = _chrono_tire_path("vehicle/generic/tire/FialaTire.json")

        # Small sinusoidal heightfield, encoded via `elevation` and normalized by MuJoCo to [0, 1].
        nrow, ncol = 17, 17
        elev_lines = []
        for r in range(nrow):
            y = -1.0 + 2.0 * r / float(nrow - 1)
            row = []
            for c in range(ncol):
                x = -1.0 + 2.0 * c / float(ncol - 1)
                row.append(f"{math.sin(2.0 * x) * math.sin(2.0 * y):.8f}")
            elev_lines.append(" ".join(row))
        elevation = "\n".join(elev_lines)

        xml = f"""
<mujoco model="hfield_tire">
  <option timestep="0.01" integrator="Euler" gravity="0 0 -9.81"/>
  <asset>
    <hfield name="hf" nrow="{nrow}" ncol="{ncol}" size="1 1 0.2 0.1" elevation="{elevation}"/>
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" hfield="hf" friction="0.7 0 0"/>
    <body name="wheel" pos="0 0 0.45">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

        mjm = mujoco.MjModel.from_xml_string(xml)
        mjd = mujoco.MjData(mjm)
        mjd.qvel[0:6] = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0] 

        m = mujoco_warp.put_model(mjm)
        d = mujoco_warp.put_data(mjm, mjd, nworld=2)

        params = run_chrono_gt({"cmd": "get_params", "tire_json": tire_json})
        self.assertTrue(params["ok"])
        disc_radius = float(params["unloaded_radius"])

        wheel_id = int(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "wheel"))

        # Build Chrono terrain request matching MuJoCo's internal normalized hfield representation.
        ground_geom_id = int(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ground"))
        hid = int(mjm.geom_dataid[ground_geom_id])
        size_arr = mjm.hfield_size
        if getattr(size_arr, "ndim", 1) == 2:
            size = size_arr[hid].tolist()
        else:
            size = size_arr[4 * hid : 4 * hid + 4].tolist()
        nrow_m = int(mjm.hfield_nrow[hid])
        ncol_m = int(mjm.hfield_ncol[hid])
        adr = int(mjm.hfield_adr[hid])
        data = mjm.hfield_data[adr : adr + nrow_m * ncol_m].tolist()

        terrain = {"type": "hfield", "mu": 0.7, "size": size, "nrow": nrow_m, "ncol": ncol_m, "data": data}

        # Ensure one world has contact and the other does not (heightfield surface is generally not at z=0).
        hf = HFieldTerrain(
            size=tuple(float(x) for x in size),
            nrow=nrow_m,
            ncol=ncol_m,
            data=data,
            pos=(0.0, 0.0, 0.0),
            mu=0.7,
        )
        h0 = float(hf.GetHeight((0.0, 0.0, 0.0)))

        qpos = d.qpos.numpy()
        qpos[0, 2] = disc_radius + h0 - 0.05
        qpos[1, 2] = disc_radius + h0 + 0.05
        d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

        for method in [CollisionType.SINGLE_POINT, CollisionType.FOUR_POINTS, CollisionType.ENVELOPE]:
            with self.subTest(method=method):
                # Sentinel to ensure the callback overwrites.
                d.xfrc_applied.fill_(wp.spatial_vector(123.0, -456.0, 789.0, 1.0, 2.0, 3.0))

                module = MujocoFialaTireModule.from_mujoco_names(
                    mjm,
                    tire_json,
                    wheel_body_names=["wheel"],
                    terrain_geom_name="ground",
                    collision_type=method,
                )
                mujoco_warp.mjcb_control = module.apply
                mujoco_warp.forward(m, d)

                xipos = d.xipos.numpy()
                xmat = d.xmat.numpy()
                cvel = d.cvel.numpy()
                xfrc = d.xfrc_applied.numpy()

                for worldid in range(2):
                    disc_center = tuple(float(x) for x in xipos[worldid, wheel_id])
                    disc_normal = (
                        float(xmat[worldid, wheel_id, 0, 1]),
                        float(xmat[worldid, wheel_id, 1, 1]),
                        float(xmat[worldid, wheel_id, 2, 1]),
                    )
                    omega_world = tuple(float(x) for x in cvel[worldid, wheel_id, 0:3])
                    vel_world = tuple(float(x) for x in cvel[worldid, wheel_id, 3:6])

                    exp_f, exp_m = _chronos_force_wrench_world(
                        tire_json=tire_json,
                        collision_type=method,
                        disc_center=disc_center,
                        disc_normal=disc_normal,
                        vel_world=vel_world,
                        omega_world=omega_world,
                        wheel_center=disc_center,
                        terrain=terrain,
                    )

                    act_f = tuple(float(x) for x in xfrc[worldid, wheel_id, 0:3])
                    act_m = tuple(float(x) for x in xfrc[worldid, wheel_id, 3:6])

                    if worldid == 0:
                        self.assertGreater(exp_f[2], 0.0)
                        self.assertGreater(act_f[2], 0.0)
                    else:
                        self.assertEqual(exp_f, (0.0, 0.0, 0.0))
                        self.assertEqual(exp_m, (0.0, 0.0, 0.0))
                        self.assertEqual(act_f, (0.0, 0.0, 0.0))
                        self.assertEqual(act_m, (0.0, 0.0, 0.0))

                    for i in range(3):
                        _assert_close(self, act_f[i], exp_f[i], rtol=3e-4, atol=1e-2)
                        _assert_close(self, act_m[i], exp_m[i], rtol=3e-4, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
