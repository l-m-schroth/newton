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

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Keep Warp cache inside the repo by default (helps when running in sandboxed environments).
if not os.environ.get("WARP_CACHE_PATH"):
    os.environ["WARP_CACHE_PATH"] = str(_REPO_ROOT / ".warp_cache")

import warp as wp  # noqa: E402

from newton._src.vehicle.tires.disc_terrain_collision import (
    ChFunctionInterp,
    CollisionType,
    ConstructAreaDepthTable,
    HFieldTerrain,
    _tests_disc_terrain_collision_kernel,
    _v_normalize,
)
from newton.tests.tires.chrono_gt import run_chrono_gt


def _repo_root() -> Path:
    return _REPO_ROOT


if wp.config.kernel_cache_dir != os.environ["WARP_CACHE_PATH"]:
    Path(os.environ["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]


def _assert_close(testcase: unittest.TestCase, actual: float, expected: float, *, rtol: float, atol: float) -> None:
    err = abs(actual - expected)
    tol = atol + rtol * abs(expected)
    testcase.assertLessEqual(err, tol, msg=f"actual={actual} expected={expected} err={err} tol={tol}")


def _assert_vec3_close(
    testcase: unittest.TestCase, actual: tuple[float, float, float], expected: list[float], *, rtol: float, atol: float
) -> None:
    for i in range(3):
        _assert_close(testcase, float(actual[i]), float(expected[i]), rtol=rtol, atol=atol)

def _method_str(method: CollisionType) -> str:
    if method == CollisionType.SINGLE_POINT:
        return "single_point"
    if method == CollisionType.FOUR_POINTS:
        return "four_points"
    if method == CollisionType.ENVELOPE:
        return "envelope"
    raise ValueError(f"Unsupported collision method: {method}")

def _run_warp_mujoco_disc_terrain_collision(
    *,
    method: CollisionType,
    disc_center: list[tuple[float, float, float]],
    disc_normal: list[tuple[float, float, float]],
    disc_radius: float,
    width: float,
    terrain: dict[str, object],
    device: str | wp.context.Device = "cpu",
) -> dict[str, list[object]]:
    wp.init()

    nworld = len(disc_center)
    if nworld == 0 or len(disc_normal) != nworld:
        raise ValueError("disc_center and disc_normal must have the same non-zero length.")

    terrain_geom_id = 0
    ngeom = 1

    # Build area-depth table (used by ENVELOPE, but passed for all modes for simplicity).
    area_dep = ChFunctionInterp()
    ConstructAreaDepthTable(float(disc_radius), area_dep)
    pairs = area_dep.table
    area_xs = wp.array(np.array([p[0] for p in pairs], dtype=np.float32), dtype=float, device=device)
    area_ys = wp.array(np.array([p[1] for p in pairs], dtype=np.float32), dtype=float, device=device)
    area_n = int(len(pairs))

    # Common MuJoCo arrays (shape: (nworld, ngeom, ...)).
    geom_friction_np = np.zeros((nworld, ngeom, 3), dtype=np.float32)
    geom_friction_np[:, :, 0] = float(terrain.get("mu", 0.8))

    geom_xpos_np = np.zeros((nworld, ngeom, 3), dtype=np.float32)
    geom_xmat_np = np.broadcast_to(np.eye(3, dtype=np.float32), (nworld, ngeom, 3, 3)).copy()

    if terrain["type"] == "plane":
        geom_type_np = np.array([0], dtype=np.int32)  # mjGEOM_PLANE
        geom_dataid_np = np.array([0], dtype=np.int32)
        geom_xpos_np[:, terrain_geom_id, 2] = float(terrain.get("height", 0.0))

        # Dummy hfield arrays (unused for planes).
        hfield_size_np = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        hfield_nrow_np = np.array([0], dtype=np.int32)
        hfield_ncol_np = np.array([0], dtype=np.int32)
        hfield_adr_np = np.array([0], dtype=np.int32)
        hfield_data_np = np.array([0.0], dtype=np.float32)

    elif terrain["type"] == "hfield":
        geom_type_np = np.array([1], dtype=np.int32)  # mjGEOM_HFIELD
        geom_dataid_np = np.array([0], dtype=np.int32)  # single hfield

        size = tuple(float(x) for x in terrain["size"])
        nrow = int(terrain["nrow"])
        ncol = int(terrain["ncol"])
        data = np.array([float(x) for x in terrain["data"]], dtype=np.float32)
        if data.size != nrow * ncol:
            raise ValueError("hfield data length must match nrow*ncol")

        hfield_size_np = np.array([list(size)], dtype=np.float32)
        hfield_nrow_np = np.array([nrow], dtype=np.int32)
        hfield_ncol_np = np.array([ncol], dtype=np.int32)
        hfield_adr_np = np.array([0], dtype=np.int32)
        hfield_data_np = data

        if "pos" in terrain:
            pos = tuple(float(x) for x in terrain["pos"])
            geom_xpos_np[:, terrain_geom_id, :] = np.array(pos, dtype=np.float32)

    else:
        raise ValueError(f"Unsupported terrain type: {terrain['type']!r}")

    geom_type = wp.array(geom_type_np, dtype=wp.int32, device=device)
    geom_dataid = wp.array(geom_dataid_np, dtype=wp.int32, device=device)
    geom_friction = wp.array(geom_friction_np, dtype=wp.vec3, device=device)
    geom_xpos = wp.array(geom_xpos_np, dtype=wp.vec3, device=device)
    geom_xmat = wp.array(geom_xmat_np, dtype=wp.mat33, device=device)

    hfield_size = wp.array(hfield_size_np, dtype=wp.vec4, device=device)
    hfield_nrow = wp.array(hfield_nrow_np, dtype=wp.int32, device=device)
    hfield_ncol = wp.array(hfield_ncol_np, dtype=wp.int32, device=device)
    hfield_adr = wp.array(hfield_adr_np, dtype=wp.int32, device=device)
    hfield_data = wp.array(hfield_data_np, dtype=float, device=device)

    disc_center_wp = wp.array(np.array(disc_center, dtype=np.float32), dtype=wp.vec3, device=device)
    disc_normal_wp = wp.array(np.array(disc_normal, dtype=np.float32), dtype=wp.vec3, device=device)

    in_contact_out = wp.empty(nworld, dtype=wp.int32, device=device)
    depth_out = wp.empty(nworld, dtype=float, device=device)
    mu_out = wp.empty(nworld, dtype=float, device=device)
    pos_out = wp.empty(nworld, dtype=wp.vec3, device=device)
    x_axis_out = wp.empty(nworld, dtype=wp.vec3, device=device)
    y_axis_out = wp.empty(nworld, dtype=wp.vec3, device=device)
    z_axis_out = wp.empty(nworld, dtype=wp.vec3, device=device)

    wp.launch(
        _tests_disc_terrain_collision_kernel,
        dim=nworld,
        inputs=[
            geom_type,
            geom_dataid,
            geom_friction,
            hfield_size,
            hfield_nrow,
            hfield_ncol,
            hfield_adr,
            hfield_data,
            geom_xpos,
            geom_xmat,
            disc_center_wp,
            disc_normal_wp,
            int(terrain_geom_id),
            int(method),
            float(disc_radius),
            float(width),
            area_xs,
            area_ys,
            int(area_n),
        ],
        outputs=[
            in_contact_out,
            depth_out,
            mu_out,
            pos_out,
            x_axis_out,
            y_axis_out,
            z_axis_out,
        ],
        device=device,
    )

    return {
        "in_contact": [bool(x) for x in in_contact_out.numpy().tolist()],
        "depth": depth_out.numpy().tolist(),
        "mu": mu_out.numpy().tolist(),
        "pos": pos_out.numpy().tolist(),
        "x_axis": x_axis_out.numpy().tolist(),
        "y_axis": y_axis_out.numpy().tolist(),
        "z_axis": z_axis_out.numpy().tolist(),
    }


class TestDiscTerrainCollisionAgainstChrono(unittest.TestCase):
    def test_mujoco_plane_matches_chrono_for_nworld_2_all_collision_modes(self):
        disc_radius = 0.5
        width = 0.2
        mu = 0.8
        terrain = {"type": "plane", "height": 0.0, "mu": mu}

        disc_centers = [(0.0, 0.0, disc_radius - 0.05), (0.0, 0.0, disc_radius + 0.05)]
        disc_normals = [
            _v_normalize((0.0, 1.0, 0.0)),
            _v_normalize((0.0, 1.0, 0.1)),
            _v_normalize((0.0, 1.0, -0.1)),
        ]

        for disc_normal in disc_normals:
            for method in [CollisionType.SINGLE_POINT, CollisionType.FOUR_POINTS, CollisionType.ENVELOPE]:
                with self.subTest(method=method, disc_normal=disc_normal):
                    batched = _run_warp_mujoco_disc_terrain_collision(
                        method=method,
                        disc_center=list(disc_centers),
                        disc_normal=[disc_normal, disc_normal],
                        disc_radius=disc_radius,
                        width=width,
                        terrain=terrain,
                        device="cpu",
                    )

                    for i in range(2):
                        gt = run_chrono_gt(
                            {
                                "cmd": "disc_terrain_collision",
                                "collision_type": _method_str(method),
                                "disc_center": list(disc_centers[i]),
                                "disc_normal": list(disc_normal),
                                "disc_radius": disc_radius,
                                "width": width,
                                "terrain": terrain,
                            }
                        )
                        self.assertTrue(gt["ok"])

                        self.assertEqual(batched["in_contact"][i], gt["in_contact"])
                        _assert_close(self, float(batched["depth"][i]), float(gt["depth"]), rtol=1e-6, atol=1e-6)
                        _assert_close(self, float(batched["mu"][i]), float(gt["mu"]), rtol=1e-6, atol=1e-6)

                        if i == 0: # Make sure we are in contact for the scenario where we actually want contacts
                            self.assertTrue(gt["in_contact"])
                            self.assertTrue(batched["in_contact"][i])
                        else: # No contact in contact scenario
                            self.assertFalse(gt["in_contact"])
                            self.assertFalse(batched["in_contact"][i])

                        for k in ["pos", "x_axis", "y_axis", "z_axis"]:
                            for j in range(3):
                                _assert_close(
                                    self,
                                    float(batched[k][i][j]),
                                    float(gt["contact"][k][j]),
                                    rtol=1e-6,
                                    atol=1e-6,
                                )

    def test_mujoco_hfield_matches_chrono_for_nworld_2_all_collision_modes(self):
        nrow, ncol = 17, 17
        size_x, size_y, size_z_top, size_z_bottom = (1.0, 1.0, 0.2, 0.0)
        mu = 0.7

        data: list[float] = []
        for r in range(nrow):
            y = -size_y + 2.0 * size_y * r / float(nrow - 1)
            for c in range(ncol):
                x = -size_x + 2.0 * size_x * c / float(ncol - 1)
                raw = math.sin(2.0 * x) * math.sin(2.0 * y)  # [-1, 1]
                data.append(0.5 * (raw + 1.0))  # normalize to [0, 1]

        terrain = {"type": "hfield", "mu": mu, "size": [size_x, size_y, size_z_top, size_z_bottom], "nrow": nrow, "ncol": ncol, "data": data}

        disc_radius = 0.5
        width = 0.2

        hf = HFieldTerrain(size=(size_x, size_y, size_z_top, size_z_bottom), nrow=nrow, ncol=ncol, data=data, pos=(0.0, 0.0, 0.0), mu=mu)
        h0 = float(hf.GetHeight((0.0, 0.0, 0.0)))

        disc_centers = [(0.0, 0.0, disc_radius + h0 - 0.05), (0.0, 0.0, disc_radius + h0 + 0.05)]
        disc_normals = [tuple(_v_normalize((0.0, 1.0, 0.1))), tuple(_v_normalize((0.0, 1.0, -0.1)))]

        for method in [CollisionType.SINGLE_POINT, CollisionType.FOUR_POINTS, CollisionType.ENVELOPE]:
            with self.subTest(method=method):
                batched = _run_warp_mujoco_disc_terrain_collision(
                    method=method,
                    disc_center=list(disc_centers),
                    disc_normal=disc_normals,
                    disc_radius=disc_radius,
                    width=width,
                    terrain=terrain,
                    device="cpu",
                )

                for i in range(2):
                    gt = run_chrono_gt(
                        {
                            "cmd": "disc_terrain_collision",
                            "collision_type": _method_str(method),
                            "disc_center": list(disc_centers[i]),
                            "disc_normal": list(disc_normals[i]),
                            "disc_radius": disc_radius,
                            "width": width,
                            "terrain": terrain,
                        }
                    )
                    self.assertTrue(gt["ok"])

                    self.assertEqual(batched["in_contact"][i], gt["in_contact"])
                    _assert_close(self, float(batched["depth"][i]), float(gt["depth"]), rtol=1e-6, atol=1e-6)
                    _assert_close(self, float(batched["mu"][i]), float(gt["mu"]), rtol=1e-6, atol=1e-6)

                    if i == 0:
                        self.assertTrue(gt["in_contact"])
                        self.assertTrue(batched["in_contact"][i])
                    else:
                        self.assertFalse(gt["in_contact"])
                        self.assertFalse(batched["in_contact"][i])

                    for k in ["pos", "x_axis", "y_axis", "z_axis"]:
                        for j in range(3):
                            _assert_close(
                                self,
                                float(batched[k][i][j]),
                                float(gt["contact"][k][j]),
                                rtol=1e-6,
                                atol=1e-6,
                            )
if __name__ == "__main__":
    unittest.main()
