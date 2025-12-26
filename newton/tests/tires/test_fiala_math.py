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
import unittest
from pathlib import Path

import warp as wp

from newton._src.vehicle.tires import FialaTire
from newton._src.vehicle.tires.fiala_tire import run_fiala_tire_advance_batched
from newton.tests.tires.chrono_gt import run_chrono_gt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _chrono_tire_path(rel: str) -> str:
    return str(_repo_root() / "chrono" / "build" / "data" / rel)


def _assert_close(testcase: unittest.TestCase, actual: float, expected: float, *, rtol: float, atol: float) -> None:
    err = abs(actual - expected)
    tol = atol + rtol * abs(expected)
    testcase.assertLessEqual(err, tol, msg=f"actual={actual} expected={expected} err={err} tol={tol}")


class TestFialaMathAgainstChrono(unittest.TestCase):
    def test_normal_stiffness_force_linear(self):
        tire_json = _chrono_tire_path("vehicle/generic/tire/FialaTire.json")
        tire = FialaTire.from_json(tire_json)

        for depth in [0.0, 1e-4, 0.01, 0.05]:
            with self.subTest(depth=depth):
                gt = run_chrono_gt({"cmd": "normal_stiffness_force", "tire_json": tire_json, "depth": depth})
                self.assertTrue(gt["ok"])
                _assert_close(self, tire.GetNormalStiffnessForce(depth), gt["fz_stiff"], rtol=1e-12, atol=1e-12)

    def test_normal_stiffness_force_vertical_curve_table(self):
        tire_json = _chrono_tire_path("vehicle/hmmwv/tire/HMMWV_FialaTire.json")
        tire = FialaTire.from_json(tire_json)
        self.assertTrue(tire.m_has_vert_table)

        depths = [-0.01, 0.0, 0.0123, 0.08, 0.09]
        for depth in depths:
            with self.subTest(depth=depth):
                gt = run_chrono_gt({"cmd": "normal_stiffness_force", "tire_json": tire_json, "depth": depth})
                self.assertTrue(gt["ok"])
                _assert_close(self, tire.GetNormalStiffnessForce(depth), gt["fz_stiff"], rtol=1e-12, atol=1e-10)

    def test_normal_damping_force(self):
        for rel in ["vehicle/generic/tire/FialaTire.json", "vehicle/hmmwv/tire/HMMWV_FialaTire.json"]:
            tire_json = _chrono_tire_path(rel)
            tire = FialaTire.from_json(tire_json)

            for velocity in [-10.0, -0.3, 0.0, 0.2, 4.0]:
                with self.subTest(tire=rel, velocity=velocity):
                    gt = run_chrono_gt(
                        {"cmd": "normal_damping_force", "tire_json": tire_json, "depth": 0.02, "velocity": velocity}
                    )
                    self.assertTrue(gt["ok"])
                    _assert_close(self, tire.GetNormalDampingForce(0.02, velocity), gt["fz_damp"], rtol=1e-12, atol=1e-12)

    def test_normal_load_clamp(self):
        tire_json = _chrono_tire_path("vehicle/generic/tire/FialaTire.json")
        tire = FialaTire.from_json(tire_json)

        cases = [
            (0.02, 0.0),
            (0.02, 0.5),
            (0.02, -0.5),
            (0.001, 10.0),  # should clamp to 0
        ]
        for depth, vel_z in cases:
            with self.subTest(depth=depth, vel_z=vel_z):
                gt = run_chrono_gt({"cmd": "normal_load", "tire_json": tire_json, "depth": depth, "vel_z": vel_z})
                self.assertTrue(gt["ok"])
                _assert_close(self, tire.GetNormalLoad(depth, vel_z), gt["fz"], rtol=1e-12, atol=1e-10)

    def test_fiala_patch_forces_and_mu_clamp(self):
        for rel in ["vehicle/generic/tire/FialaTire.json", "vehicle/hmmwv/tire/HMMWV_FialaTire.json"]:
            tire_json = _chrono_tire_path(rel)
            tire = FialaTire.from_json(tire_json)

            cases = [
                # (kappa, alpha, fz, mu)
                (0.05, 0.02, 3000.0, 0.8),
                (0.5, 0.02, 3000.0, 0.8),
                (0.05, 0.2, 3000.0, 0.8),
                (0.05, 0.02, 3000.0, 0.05),  # clamp to 0.1
                (0.05, 0.02, 3000.0, 1.5),  # clamp to 1.0
            ]

            for kappa, alpha, fz, mu in cases:
                with self.subTest(tire=rel, kappa=kappa, alpha=alpha, mu=mu):
                    gt = run_chrono_gt(
                        {
                            "cmd": "fiala_patch_forces",
                            "tire_json": tire_json,
                            "kappa": kappa,
                            "alpha": alpha,
                            "fz": fz,
                            "mu": mu,
                        }
                    )
                    self.assertTrue(gt["ok"])

                    tire.SetMu(mu, clamp_mu=True)
                    fx, fy, mz = tire.FialaPatchForces(kappa, alpha, fz)
                    _assert_close(self, fx, gt["fx"], rtol=1e-12, atol=1e-10)
                    _assert_close(self, fy, gt["fy"], rtol=1e-12, atol=1e-10)
                    _assert_close(self, mz, gt["mz"], rtol=1e-12, atol=1e-10)

    def test_rolling_resistance_moment(self):
        for rel in ["vehicle/generic/tire/FialaTire.json", "vehicle/hmmwv/tire/HMMWV_FialaTire.json"]:
            tire_json = _chrono_tire_path(rel)
            tire = FialaTire.from_json(tire_json)

            cases = [
                # abs_vx, omega, fz
                (0.0, 10.0, 4000.0),
                (0.1, 10.0, 4000.0),
                (0.2, 10.0, 4000.0),
                (1.0, 10.0, 4000.0),
                (1.0, -10.0, 4000.0),
                (1.0, 0.0, 4000.0),
            ]

            for abs_vx, omega, fz in cases:
                with self.subTest(tire=rel, abs_vx=abs_vx, omega=omega):
                    gt = run_chrono_gt(
                        {
                            "cmd": "rolling_resistance_moment",
                            "tire_json": tire_json,
                            "abs_vx": abs_vx,
                            "omega": omega,
                            "fz": fz,
                        }
                    )
                    self.assertTrue(gt["ok"])
                    my = tire.RollingResistanceMoment(abs_vx, omega, fz)
                    _assert_close(self, my, gt["My"], rtol=1e-12, atol=1e-10)

    def test_batched_advance_matches_chrono_for_nworld_2(self):
        wp.init()

        for rel in ["vehicle/generic/tire/FialaTire.json", "vehicle/hmmwv/tire/HMMWV_FialaTire.json"]:
            tire_json = _chrono_tire_path(rel)
            tire = FialaTire.from_json(tire_json)

            vx = [0.05, 3.0]
            vy = [0.01, -0.4]
            vel_z = [0.0, 0.2]
            omega = [10.0, -30.0]
            depth = [0.02, 0.04]
            mu = [0.8, 0.05]  # second clamps to 0.1

            gt_all = []
            for i in range(2):
                fz_gt = run_chrono_gt(
                    {"cmd": "normal_load", "tire_json": tire_json, "depth": depth[i], "vel_z": vel_z[i]}
                )["fz"]

                r_eff = tire.m_unloaded_radius - depth[i]
                slip = run_chrono_gt({"cmd": "fiala_slip", "v_x": vx[i], "v_y": vy[i], "omega": omega[i], "r_eff": r_eff})

                patch = run_chrono_gt(
                    {
                        "cmd": "fiala_patch_forces",
                        "tire_json": tire_json,
                        "kappa": slip["kappa"],
                        "alpha": slip["alpha"],
                        "fz": fz_gt,
                        "mu": mu[i],
                    }
                )

                rr = run_chrono_gt(
                    {
                        "cmd": "rolling_resistance_moment",
                        "tire_json": tire_json,
                        "abs_vx": abs(vx[i]),
                        "omega": omega[i],
                        "fz": fz_gt,
                    }
                )

                gt_all.append(
                    {
                        "kappa": slip["kappa"],
                        "alpha": slip["alpha"],
                        "fx": patch["fx"],
                        "fy": patch["fy"],
                        "fz": fz_gt,
                        "mz": patch["mz"],
                        "my": rr["My"],
                    }
                )

            py0 = tire.AdvancePure(vx[0], vy[0], vel_z[0], omega[0], depth[0], mu[0])
            py1 = tire.AdvancePure(vx[1], vy[1], vel_z[1], omega[1], depth[1], mu[1])

            for k in ["kappa", "alpha", "fx", "fy", "fz", "mz", "my"]:
                with self.subTest(tire=rel, impl="python", key=k, idx=0):
                    _assert_close(self, py0[k], gt_all[0][k], rtol=1e-12, atol=1e-10)
                with self.subTest(tire=rel, impl="python", key=k, idx=1):
                    _assert_close(self, py1[k], gt_all[1][k], rtol=1e-12, atol=1e-10)

            batched = run_fiala_tire_advance_batched(tire, vx, vy, vel_z, omega, depth, mu, device="cpu")
            for i in range(2):
                for k in ["kappa", "alpha", "fx", "fy", "fz", "mz", "my"]:
                    with self.subTest(tire=rel, impl="warp", key=k, idx=i):
                        _assert_close(self, batched[k][i], gt_all[i][k], rtol=1e-5, atol=1e-5)

    def test_batched_advance_matches_python_for_additional_cases_nworld_2(self):
        wp.init()

        for rel in ["vehicle/generic/tire/FialaTire.json", "vehicle/hmmwv/tire/HMMWV_FialaTire.json"]:
            tire_json = _chrono_tire_path(rel)
            tire = FialaTire.from_json(tire_json)

            cases: list[dict[str, list[float]]] = [
                # Explicit low-speed branch: slip and rolling resistance should be zeroed when vx == 0.
                {
                    "vx": [0.0, 0.0],
                    "vy": [0.5, -0.2],
                    "vel_z": [0.0, 0.0],
                    "omega": [10.0, -10.0],
                    "depth": [0.02, 0.02],
                    "mu": [0.8, 0.8],
                },
                # Rolling resistance sine-step mid-range, and sign handling for omega.
                {
                    "vx": [0.2, 0.3],
                    "vy": [0.05, -0.15],
                    "vel_z": [0.0, 0.1],
                    "omega": [5.0, -5.0],
                    "depth": [0.02, 0.04],
                    "mu": [0.8, 0.8],
                },
                # Mu clamping in the warp path (warp always clamps).
                {
                    "vx": [2.0, 2.0],
                    "vy": [0.1, -0.3],
                    "vel_z": [0.0, 0.1],
                    "omega": [5.0, -5.0],
                    "depth": [0.02, 0.02],
                    "mu": [-1.0, 10.0],
                },
                # No-contact style case (negative penetration): fz should clamp to 0, and forces should remain finite.
                {
                    "vx": [1.0, -1.0],
                    "vy": [0.2, -0.2],
                    "vel_z": [0.0, 0.0],
                    "omega": [0.0, 0.0],
                    "depth": [-0.02, -0.01],
                    "mu": [0.8, 0.8],
                },
            ]

            if tire.m_has_vert_table:
                cases.append(
                    # Extrapolation beyond the last vertical curve table entry.
                    {
                        "vx": [1.0, 1.0],
                        "vy": [0.1, -0.1],
                        "vel_z": [0.0, 0.0],
                        "omega": [10.0, -10.0],
                        "depth": [tire.m_max_depth + 0.01, tire.m_max_depth + 0.02],
                        "mu": [0.8, 0.8],
                    }
                )

            for case_idx, c in enumerate(cases):
                with self.subTest(tire=rel, case_idx=case_idx):
                    py0 = tire.AdvancePure(c["vx"][0], c["vy"][0], c["vel_z"][0], c["omega"][0], c["depth"][0], c["mu"][0])
                    py1 = tire.AdvancePure(c["vx"][1], c["vy"][1], c["vel_z"][1], c["omega"][1], c["depth"][1], c["mu"][1])

                    batched = run_fiala_tire_advance_batched(
                        tire,
                        c["vx"],
                        c["vy"],
                        c["vel_z"],
                        c["omega"],
                        c["depth"],
                        c["mu"],
                        device="cpu",
                    )
                    for i, py in enumerate([py0, py1]):
                        for k in ["kappa", "alpha", "fx", "fy", "fz", "mz", "my"]:
                            with self.subTest(tire=rel, case_idx=case_idx, impl="warp_vs_python", key=k, idx=i):
                                _assert_close(self, batched[k][i], py[k], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
