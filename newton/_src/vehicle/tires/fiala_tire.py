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
import json
import math
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Sequence

import warp as wp


def _strip_json_comments(text: str) -> str:
    in_string = False
    escape = False
    out: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]

        if in_string:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue

        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue

        if c == "/" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < len(text) and text[i] not in "\n\r":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue

        out.append(c)
        i += 1

    return "".join(out)


def load_chrono_vehicle_json(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(_strip_json_comments(text))


def ch_clamp(value: float, limit_min: float, limit_max: float) -> float:
    # Chrono reference: chrono/src/chrono/utils/ChUtils.h::ChClamp
    if value < limit_min:
        return limit_min
    if value > limit_max:
        return limit_max
    return value


def ch_signum(x: float) -> int:
    # Chrono reference: chrono/src/chrono/utils/ChUtils.h::ChSignum
    return int(x > 0.0) - int(x < 0.0)


class ChFunctionInterp:
    # Chrono reference: chrono/src/chrono/functions/ChFunctionInterp.cpp::GetVal

    def __init__(self) -> None:
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._extrapolate: bool = False

    def SetExtrapolate(self, extrapolate: bool) -> None:
        self._extrapolate = bool(extrapolate)

    def AddPoint(self, x: float, y: float, overwrite_if_existing: bool = False) -> None:
        i = bisect_left(self._xs, x)
        if i < len(self._xs) and self._xs[i] == x:
            if overwrite_if_existing:
                self._ys[i] = y
                return
            raise ValueError("Point already exists and overwrite flag was not set.")
        self._xs.insert(i, x)
        self._ys.insert(i, y)

    def GetVal(self, x: float) -> float:
        if not self._xs:
            return 0.0

        if x <= self._xs[0]:
            return self._ys[0] - self.GetDer(x) * (self._xs[0] - x)

        if x >= self._xs[-1]:
            return self._ys[-1] + self.GetDer(x) * (x - self._xs[-1])

        i = bisect_right(self._xs, x)  # first element strictly greater than x
        x_prev = self._xs[i - 1]
        y_prev = self._ys[i - 1]
        x_next = self._xs[i]
        y_next = self._ys[i]
        return y_prev + (y_next - y_prev) * (x - x_prev) / (x_next - x_prev)

    def GetDer(self, x: float) -> float:
        if not self._xs:
            return 0.0

        if x <= self._xs[0]:
            if self._extrapolate and len(self._xs) > 1:
                return (self._ys[1] - self._ys[0]) / (self._xs[1] - self._xs[0])
            return 0.0

        if x >= self._xs[-1]:
            if self._extrapolate and len(self._xs) > 1:
                return (self._ys[-1] - self._ys[-2]) / (self._xs[-1] - self._xs[-2])
            return 0.0

        i = bisect_right(self._xs, x)  # first element strictly greater than x
        return (self._ys[i] - self._ys[i - 1]) / (self._xs[i] - self._xs[i - 1])

    @property
    def table(self) -> list[tuple[float, float]]:
        return list(zip(self._xs, self._ys, strict=True))


class ChFunctionSineStep:
    # Chrono reference: chrono/src/chrono/functions/ChFunctionSineStep.cpp::Eval

    @staticmethod
    def Eval(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
        if x <= x1:
            return y1
        if x >= x2:
            return y2
        xx = (x - x1) / (x2 - x1)
        return y1 + (y2 - y1) * (xx - math.sin(2.0 * math.pi * xx) / (2.0 * math.pi))


@dataclasses.dataclass(slots=True)
class ChFialaTire:
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp
    m_mu: float = 0.8
    m_mu_0: float = 0.8
    m_unloaded_radius: float = 0.0
    m_width: float = 0.0
    m_rolling_resistance: float = 0.0
    m_c_slip: float = 0.0
    m_c_alpha: float = 0.0
    m_u_min: float = 0.0
    m_u_max: float = 0.0

    def SetMu(self, mu: float, clamp_mu: bool = True) -> None:
        # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp::Synchronize
        if clamp_mu:
            mu = ch_clamp(mu, 0.1, 1.0)
        self.m_mu = float(mu)

    def ComputeSlip(self, vx: float, vy: float, omega: float, depth: float) -> tuple[float, float]:
        # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp::Advance
        abs_vx = abs(vx)
        if abs_vx != 0.0:
            vsx = vx - omega * (self.m_unloaded_radius - depth)
            kappa = -vsx / abs_vx
            alpha = math.atan2(vy, abs_vx)
        else:
            kappa = 0.0
            alpha = 0.0
        return kappa, alpha

    def RollingResistanceMoment(self, abs_vx: float, omega: float, normal_force: float) -> float:
        # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp::Advance
        vx_min = 0.125
        vx_max = 0.5
        my_startup = ChFunctionSineStep.Eval(abs_vx, vx_min, 0.0, vx_max, 1.0)
        return -my_startup * self.m_rolling_resistance * normal_force * ch_signum(omega)

    def FialaPatchForces(self, kappa: float, alpha: float, fz: float) -> tuple[float, float, float]:
        # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp::FialaPatchForces
        tan_alpha = math.tan(alpha)
        SsA = min(1.0, math.sqrt(kappa * kappa + tan_alpha * tan_alpha))
        U = self.m_u_max - (self.m_u_max - self.m_u_min) * SsA
        S_critical = abs(U * fz / (2.0 * self.m_c_slip))
        Alpha_critical = math.atan(3.0 * U * fz / self.m_c_alpha)

        U *= self.m_mu / self.m_mu_0

        if abs(kappa) < S_critical:
            fx = self.m_c_slip * kappa
        else:
            Fx1 = U * fz
            Fx2 = abs((U * fz) * (U * fz) / (4.0 * kappa * self.m_c_slip))
            fx = ch_signum(kappa) * (Fx1 - Fx2)

        if abs(alpha) <= Alpha_critical:
            H = 1.0 - self.m_c_alpha * abs(tan_alpha) / (3.0 * U * fz)
            fy = -U * fz * (1.0 - H**3) * ch_signum(alpha)
            mz = U * fz * self.m_width * (1.0 - H) * (H**3) * ch_signum(alpha)
        else:
            fy = -U * fz * ch_signum(alpha)
            mz = 0.0

        return fx, fy, mz


@dataclasses.dataclass(slots=True)
class FialaTire(ChFialaTire):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/FialaTire.cpp
    m_normalStiffness: float = 0.0
    m_normalDamping: float = 0.0

    m_has_vert_table: bool = False
    m_vert_map: ChFunctionInterp = dataclasses.field(default_factory=ChFunctionInterp)
    m_max_depth: float = 0.0
    m_max_val: float = 0.0
    m_slope: float = 0.0

    @classmethod
    def from_json(cls, filename: str | Path) -> "FialaTire":
        data = load_chrono_vehicle_json(filename)
        obj = cls()
        obj.Create(data)
        return obj

    def Create(self, d: dict[str, Any]) -> None:
        # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/FialaTire.cpp::Create
        if "Coefficient of Friction" in d:
            self.m_mu_0 = float(d["Coefficient of Friction"])

        p = d["Fiala Parameters"]

        self.m_unloaded_radius = float(p["Unloaded Radius"])
        self.m_width = float(p["Width"])
        self.m_normalStiffness = float(p["Vertical Stiffness"])

        if "Vertical Curve Data" in p:
            data = p["Vertical Curve Data"]
            self.m_vert_map = ChFunctionInterp()
            for entry in data:
                self.m_vert_map.AddPoint(float(entry[0]), float(entry[1]))
            self.m_max_depth = float(data[-1][0])
            self.m_max_val = float(data[-1][1])
            self.m_slope = (float(data[-1][1]) - float(data[-2][1])) / (float(data[-1][0]) - float(data[-2][0]))
            self.m_has_vert_table = True
        else:
            self.m_has_vert_table = False

        self.m_normalDamping = float(p["Vertical Damping"])
        self.m_rolling_resistance = float(p["Rolling Resistance"])
        self.m_c_slip = float(p["CSLIP"])
        self.m_c_alpha = float(p["CALPHA"])
        self.m_u_min = float(p["UMIN"])
        self.m_u_max = float(p["UMAX"])

    def GetNormalStiffnessForce(self, depth: float) -> float:
        # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/FialaTire.cpp::GetNormalStiffnessForce
        if self.m_has_vert_table:
            if depth > self.m_max_depth:
                return self.m_max_val + self.m_slope * (depth - self.m_max_depth)
            return self.m_vert_map.GetVal(depth)
        return self.m_normalStiffness * depth

    def GetNormalDampingForce(self, depth: float, velocity: float) -> float:
        # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/FialaTire.cpp::GetNormalDampingForce
        return self.m_normalDamping * velocity

    def GetNormalLoad(self, depth: float, vel_z: float) -> float:
        # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp::Synchronize
        fn = self.GetNormalStiffnessForce(depth) + self.GetNormalDampingForce(depth, -vel_z)
        return max(0.0, fn)

    def AdvancePure(
        self,
        vx: float,
        vy: float,
        vel_z: float,
        omega: float,
        depth: float,
        mu: float,
        clamp_mu: bool = True,
    ) -> dict[str, float]:
        # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp::{Synchronize,Advance}
        self.SetMu(mu, clamp_mu=clamp_mu)

        normal_force = self.GetNormalLoad(depth, vel_z)
        abs_vx = abs(vx)
        kappa, alpha = self.ComputeSlip(vx, vy, omega, depth)
        fx, fy, mz = self.FialaPatchForces(kappa, alpha, normal_force)
        my = self.RollingResistanceMoment(abs_vx, omega, normal_force)

        return {
            "fx": fx,
            "fy": fy,
            "fz": normal_force,
            "my": my,
            "mz": mz,
            "kappa": kappa,
            "alpha": alpha,
        }


@wp.func
def _wp_signum(x: float) -> float:
    return wp.where(x > 0.0, 1.0, wp.where(x < 0.0, -1.0, 0.0))


@wp.func
def _wp_clamp(x: float, a: float, b: float) -> float:
    return wp.min(wp.max(x, a), b)


@wp.func
def _wp_sine_step_eval(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    # Chrono reference: chrono/src/chrono/functions/ChFunctionSineStep.cpp::Eval
    if x <= x1:
        return y1
    if x >= x2:
        return y2
    xx = (x - x1) / (x2 - x1)
    return y1 + (y2 - y1) * (xx - wp.sin(2.0 * wp.pi * xx) / (2.0 * wp.pi))


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


@wp.func
def _wp_normal_stiffness_force(
    depth: float,
    has_table: int,
    normal_stiffness: float,
    xs: wp.array(dtype=float),
    ys: wp.array(dtype=float),
    n: int,
    max_depth: float,
    max_val: float,
    slope: float,
) -> float:
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/FialaTire.cpp::GetNormalStiffnessForce
    if has_table != 0:
        if depth > max_depth:
            return max_val + slope * (depth - max_depth)
        return _wp_interp_get_val(depth, xs, ys, n)
    return normal_stiffness * depth


@wp.func
def _wp_fiala_patch_forces(
    kappa: float,
    alpha: float,
    fz: float,
    mu: float,
    mu0: float,
    c_slip: float,
    c_alpha: float,
    u_min: float,
    u_max: float,
    width: float,
) -> wp.vec3:
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp::FialaPatchForces
    tan_alpha = wp.tan(alpha)
    SsA = wp.min(1.0, wp.sqrt(kappa * kappa + tan_alpha * tan_alpha))
    U = u_max - (u_max - u_min) * SsA
    S_critical = wp.abs(U * fz / (2.0 * c_slip))
    Alpha_critical = wp.atan(3.0 * U * fz / c_alpha)

    U = U * (mu / mu0)

    if wp.abs(kappa) < S_critical:
        fx = c_slip * kappa
    else:
        Fx1 = U * fz
        Fx2 = wp.abs((U * fz) * (U * fz) / (4.0 * kappa * c_slip))
        fx = _wp_signum(kappa) * (Fx1 - Fx2)

    if wp.abs(alpha) <= Alpha_critical:
        H = 1.0 - c_alpha * wp.abs(tan_alpha) / (3.0 * U * fz)
        H3 = H * H * H
        fy = -U * fz * (1.0 - H3) * _wp_signum(alpha)
        mz = U * fz * width * (1.0 - H) * H3 * _wp_signum(alpha)
    else:
        fy = -U * fz * _wp_signum(alpha)
        mz = 0.0

    return wp.vec3(fx, fy, mz)


@wp.kernel
def fiala_tire_advance_kernel(
    # Inputs per world:
    vx_in: wp.array(dtype=float),
    vy_in: wp.array(dtype=float),
    velz_in: wp.array(dtype=float),
    omega_in: wp.array(dtype=float),
    depth_in: wp.array(dtype=float),
    mu_in: wp.array(dtype=float),
    # Tire parameters (scalar):
    unloaded_radius: float,
    width: float,
    rolling_resistance: float,
    mu0: float,
    c_slip: float,
    c_alpha: float,
    u_min: float,
    u_max: float,
    normal_stiffness: float,
    normal_damping: float,
    has_vert_table: int,
    vert_xs: wp.array(dtype=float),
    vert_ys: wp.array(dtype=float),
    vert_n: int,
    vert_max_depth: float,
    vert_max_val: float,
    vert_slope: float,
    # Outputs per world:
    fx_out: wp.array(dtype=float),
    fy_out: wp.array(dtype=float),
    fz_out: wp.array(dtype=float),
    my_out: wp.array(dtype=float),
    mz_out: wp.array(dtype=float),
    kappa_out: wp.array(dtype=float),
    alpha_out: wp.array(dtype=float),
):
    # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.cpp::{Synchronize,Advance}
    tid = wp.tid()

    vx = vx_in[tid]
    vy = vy_in[tid]
    velz = velz_in[tid]
    omega = omega_in[tid]
    depth = depth_in[tid]

    mu = _wp_clamp(mu_in[tid], 0.1, 1.0)

    f_stiff = _wp_normal_stiffness_force(
        depth,
        has_vert_table,
        normal_stiffness,
        vert_xs,
        vert_ys,
        vert_n,
        vert_max_depth,
        vert_max_val,
        vert_slope,
    )
    f_damp = normal_damping * (-velz)
    fz = f_stiff + f_damp
    if fz < 0.0:
        fz = 0.0

    abs_vx = wp.abs(vx)
    if abs_vx != 0.0:
        vsx = vx - omega * (unloaded_radius - depth)
        kappa = -vsx / abs_vx
        alpha = wp.atan2(vy, abs_vx)
    else:
        kappa = 0.0
        alpha = 0.0

    f = _wp_fiala_patch_forces(kappa, alpha, fz, mu, mu0, c_slip, c_alpha, u_min, u_max, width)

    vx_min = 0.125
    vx_max = 0.5
    startup = _wp_sine_step_eval(abs_vx, vx_min, 0.0, vx_max, 1.0)
    my = -startup * rolling_resistance * fz * _wp_signum(omega)

    fx_out[tid] = f[0]
    fy_out[tid] = f[1]
    fz_out[tid] = fz
    my_out[tid] = my
    mz_out[tid] = f[2]
    kappa_out[tid] = kappa
    alpha_out[tid] = alpha


def run_fiala_tire_advance_batched(
    tire: FialaTire,
    vx: Sequence[float],
    vy: Sequence[float],
    vel_z: Sequence[float],
    omega: Sequence[float],
    depth: Sequence[float],
    mu: Sequence[float],
    device: str | wp.context.Device = "cpu",
) -> dict[str, list[float]]:
    n = len(vx)
    if not (len(vy) == len(vel_z) == len(omega) == len(depth) == len(mu) == n):
        raise ValueError("All input arrays must have the same length.")

    vx_wp = wp.array(vx, dtype=float, device=device)
    vy_wp = wp.array(vy, dtype=float, device=device)
    velz_wp = wp.array(vel_z, dtype=float, device=device)
    omega_wp = wp.array(omega, dtype=float, device=device)
    depth_wp = wp.array(depth, dtype=float, device=device)
    mu_wp = wp.array(mu, dtype=float, device=device)

    vert_pairs = tire.m_vert_map.table if tire.m_has_vert_table else []
    vert_xs = wp.array([p[0] for p in vert_pairs], dtype=float, device=device)
    vert_ys = wp.array([p[1] for p in vert_pairs], dtype=float, device=device)

    fx_out = wp.empty(n, dtype=float, device=device)
    fy_out = wp.empty(n, dtype=float, device=device)
    fz_out = wp.empty(n, dtype=float, device=device)
    my_out = wp.empty(n, dtype=float, device=device)
    mz_out = wp.empty(n, dtype=float, device=device)
    kappa_out = wp.empty(n, dtype=float, device=device)
    alpha_out = wp.empty(n, dtype=float, device=device)

    wp.launch(
        fiala_tire_advance_kernel,
        dim=n,
        inputs=[
            vx_wp,
            vy_wp,
            velz_wp,
            omega_wp,
            depth_wp,
            mu_wp,
            float(tire.m_unloaded_radius),
            float(tire.m_width),
            float(tire.m_rolling_resistance),
            float(tire.m_mu_0),
            float(tire.m_c_slip),
            float(tire.m_c_alpha),
            float(tire.m_u_min),
            float(tire.m_u_max),
            float(tire.m_normalStiffness),
            float(tire.m_normalDamping),
            int(tire.m_has_vert_table),
            vert_xs,
            vert_ys,
            int(len(vert_pairs)),
            float(tire.m_max_depth),
            float(tire.m_max_val),
            float(tire.m_slope),
        ],
        outputs=[fx_out, fy_out, fz_out, my_out, mz_out, kappa_out, alpha_out],
        device=device,
    )

    return {
        "fx": fx_out.numpy().tolist(),
        "fy": fy_out.numpy().tolist(),
        "fz": fz_out.numpy().tolist(),
        "my": my_out.numpy().tolist(),
        "mz": mz_out.numpy().tolist(),
        "kappa": kappa_out.numpy().tolist(),
        "alpha": alpha_out.numpy().tolist(),
    }
