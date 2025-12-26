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

import unittest

from newton._src.vehicle.tires.disc_terrain_collision import (
    AnalyticTerrain,
    ChFunctionInterp,
    CollisionType,
    ConstructAreaDepthTable,
    DiscTerrainCollision,
    TerrainType,
    _v_normalize,
    run_disc_terrain_collision_batched,
)
from newton.tests.tires.chrono_gt import run_chrono_gt


def _assert_close(testcase: unittest.TestCase, actual: float, expected: float, *, rtol: float, atol: float) -> None:
    err = abs(actual - expected)
    tol = atol + rtol * abs(expected)
    testcase.assertLessEqual(err, tol, msg=f"actual={actual} expected={expected} err={err} tol={tol}")


def _assert_vec3_close(
    testcase: unittest.TestCase, actual: tuple[float, float, float], expected: list[float], *, rtol: float, atol: float
) -> None:
    for i in range(3):
        _assert_close(testcase, float(actual[i]), float(expected[i]), rtol=rtol, atol=atol)


def _terrain_request(t: AnalyticTerrain) -> dict[str, object]:
    if t.type == TerrainType.PLANE:
        if t.point == (0.0, 0.0, t.point[2]) and t.normal == (0.0, 0.0, 1.0):
            return {"type": "plane", "height": float(t.point[2]), "mu": float(t.mu)}
        return {"type": "plane", "point": list(t.point), "normal": list(t.normal), "mu": float(t.mu)}
    if t.type == TerrainType.SINUSOID:
        return {"type": "sinusoid", "base": float(t.base), "amp": float(t.amp), "freq": float(t.freq), "mu": float(t.mu)}
    raise ValueError(f"Unsupported terrain type: {t.type}")


def _method_str(method: CollisionType) -> str:
    if method == CollisionType.SINGLE_POINT:
        return "single_point"
    if method == CollisionType.FOUR_POINTS:
        return "four_points"
    if method == CollisionType.ENVELOPE:
        return "envelope"
    raise ValueError(f"Unsupported collision method: {method}")


class TestDiscTerrainCollisionAgainstChrono(unittest.TestCase):
    def test_plane_flat_contact_and_no_contact_all_modes(self):
        terrain = AnalyticTerrain.Plane(height=0.0, mu=0.8)
        disc_radius = 0.5
        width = 0.2
        disc_normal = _v_normalize((0.0, 1.0, 0.0))

        area_dep = ChFunctionInterp()
        ConstructAreaDepthTable(disc_radius, area_dep)

        cases = [
            ("contact", (0.0, 0.0, disc_radius - 0.05)),
            ("no_contact", (0.0, 0.0, disc_radius + 0.05)),
        ]

        for method in [CollisionType.SINGLE_POINT, CollisionType.FOUR_POINTS, CollisionType.ENVELOPE]:
            for name, disc_center in cases:
                with self.subTest(method=method, case=name):
                    gt = run_chrono_gt(
                        {
                            "cmd": "disc_terrain_collision",
                            "collision_type": _method_str(method),
                            "disc_center": list(disc_center),
                            "disc_normal": list(disc_normal),
                            "disc_radius": disc_radius,
                            "width": width,
                            "terrain": _terrain_request(terrain),
                        }
                    )
                    self.assertTrue(gt["ok"])

                    res = DiscTerrainCollision(method, terrain, disc_center, disc_normal, disc_radius, width, area_dep)

                    self.assertEqual(res["in_contact"], gt["in_contact"])
                    _assert_close(self, float(res["depth"]), float(gt["depth"]), rtol=1e-12, atol=1e-10)
                    _assert_close(self, float(res["mu"]), float(gt["mu"]), rtol=0.0, atol=1e-6)

                    contact = res["contact"]
                    _assert_vec3_close(self, contact.pos, gt["contact"]["pos"], rtol=1e-12, atol=1e-10)
                    _assert_vec3_close(self, contact.x_axis, gt["contact"]["x_axis"], rtol=1e-12, atol=1e-10)
                    _assert_vec3_close(self, contact.y_axis, gt["contact"]["y_axis"], rtol=1e-12, atol=1e-10)
                    _assert_vec3_close(self, contact.z_axis, gt["contact"]["z_axis"], rtol=1e-12, atol=1e-10)

    def test_sinusoid_contact_all_modes(self):
        terrain = AnalyticTerrain.Sinusoid(base=0.0, amp=0.1, freq=2.0, mu=0.7)
        disc_radius = 0.5
        width = 0.2
        disc_center = (0.3, -0.2, 0.4)
        disc_normal = _v_normalize((0.05, 1.0, 0.1))

        area_dep = ChFunctionInterp()
        ConstructAreaDepthTable(disc_radius, area_dep)

        for method in [CollisionType.SINGLE_POINT, CollisionType.FOUR_POINTS, CollisionType.ENVELOPE]:
            with self.subTest(method=method):
                gt = run_chrono_gt(
                    {
                        "cmd": "disc_terrain_collision",
                        "collision_type": _method_str(method),
                        "disc_center": list(disc_center),
                        "disc_normal": list(disc_normal),
                        "disc_radius": disc_radius,
                        "width": width,
                        "terrain": _terrain_request(terrain),
                    }
                )
                self.assertTrue(gt["ok"])

                res = DiscTerrainCollision(method, terrain, disc_center, disc_normal, disc_radius, width, area_dep)

                self.assertEqual(res["in_contact"], gt["in_contact"])
                _assert_close(self, float(res["depth"]), float(gt["depth"]), rtol=1e-12, atol=1e-10)
                _assert_close(self, float(res["mu"]), float(gt["mu"]), rtol=0.0, atol=1e-6)

                contact = res["contact"]
                _assert_vec3_close(self, contact.pos, gt["contact"]["pos"], rtol=1e-12, atol=1e-10)
                _assert_vec3_close(self, contact.x_axis, gt["contact"]["x_axis"], rtol=1e-12, atol=1e-10)
                _assert_vec3_close(self, contact.y_axis, gt["contact"]["y_axis"], rtol=1e-12, atol=1e-10)
                _assert_vec3_close(self, contact.z_axis, gt["contact"]["z_axis"], rtol=1e-12, atol=1e-10)

    def test_warp_batched_plane_matches_chrono_for_nworld_2(self):
        terrain = AnalyticTerrain.Plane(height=0.0, mu=0.8)
        disc_radius = 0.5
        width = 0.2

        disc_center = [(0.0, 0.0, disc_radius - 0.05), (0.0, 0.0, disc_radius + 0.05)]
        disc_normal = [tuple(_v_normalize((0.0, 1.0, 0.0))) for _ in range(2)]

        for method in [CollisionType.SINGLE_POINT, CollisionType.FOUR_POINTS, CollisionType.ENVELOPE]:
            with self.subTest(method=method):
                batched = run_disc_terrain_collision_batched(
                    method,
                    terrain,
                    disc_center,
                    disc_normal,
                    disc_radius,
                    width,
                    device="cpu",
                )

                for i in range(2):
                    gt = run_chrono_gt(
                        {
                            "cmd": "disc_terrain_collision",
                            "collision_type": _method_str(method),
                            "disc_center": list(disc_center[i]),
                            "disc_normal": list(disc_normal[i]),
                            "disc_radius": disc_radius,
                            "width": width,
                            "terrain": _terrain_request(terrain),
                        }
                    )
                    self.assertTrue(gt["ok"])

                    self.assertEqual(batched["in_contact"][i], gt["in_contact"])
                    _assert_close(self, float(batched["depth"][i]), float(gt["depth"]), rtol=1e-5, atol=1e-5)
                    _assert_close(self, float(batched["mu"][i]), float(gt["mu"]), rtol=1e-5, atol=1e-5)
                    for k in ["pos", "x_axis", "y_axis", "z_axis"]:
                        for j in range(3):
                            _assert_close(
                                self,
                                float(batched[k][i][j]),
                                float(gt["contact"][k][j]),
                                rtol=1e-5,
                                atol=1e-5,
                            )


if __name__ == "__main__":
    unittest.main()
