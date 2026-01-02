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
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Keep Warp cache inside the repo by default (helps when running in sandboxed environments).
os.environ.setdefault("WARP_CACHE_PATH", str(_REPO_ROOT / ".warp_cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import warp as wp

# Import mujoco_warp as a package (not the workspace namespace dir).
sys.path.insert(0, str(_REPO_ROOT / "mujoco_warp"))
import mujoco_warp  # noqa: E402

import newton  # noqa: E402
from newton._src.vehicle.tires import MujocoFialaTireModule  # noqa: E402
from newton._src.vehicle.tires.disc_terrain_collision import CollisionType  # noqa: E402
from newton._src.vehicle.tires.fiala_tire import load_chrono_vehicle_json  # noqa: E402
from newton.solvers import SolverMuJoCo  # noqa: E402
from newton.tests.tires.chrono_gt import run_chrono_gt  # noqa: E402


def _repo_root() -> Path:
    return _REPO_ROOT


if wp.config.kernel_cache_dir != os.environ["WARP_CACHE_PATH"]:
    Path(os.environ["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]


@wp.kernel
def _set_rig_controls_kernel(
    joint_target_pos: wp.array(dtype=wp.float32),
    joint_target_vel: wp.array(dtype=wp.float32),
    dof_per_world: int,
    carrier_axis: int,
    slip_axis: int,
    wheel_axis: int,
    set_slip_pos: int,
    long_v: float,
    omega: float,
    slip_angle: float,
    slip_angle_vel: float,
):
    w = wp.tid()
    off = w * dof_per_world
    joint_target_vel[off + carrier_axis] = long_v
    joint_target_vel[off + wheel_axis] = omega
    if set_slip_pos != 0:
        joint_target_pos[off + slip_axis] = slip_angle
        joint_target_vel[off + slip_axis] = slip_angle_vel


def _chrono_data_path(rel: str) -> str:
    return str(_repo_root() / "chrono" / "build" / "data" / rel)


def _quat_from_angle_x(angle: float) -> tuple[float, float, float, float]:
    # Quaternion for rotation about +X by `angle` (Newton/Warp internal `xyzw` convention):
    # q_xyzw = (sin(angle/2), 0, 0, cos(angle/2)).
    #
    # Convention references:
    # - Newton internal quats are `xyzw`: `newton/newton/_src/sim/builder.py` (see docstrings mentioning `xyzw`).
    # - Chrono quats are stored as `wxyz` (`e0,e1,e2,e3` = real, i, j, k): `chrono/src/chrono/core/ChQuaternion.h`.
    #
    # Chrono formula references (wxyz storage):
    # - `chrono/src/chrono/core/ChRotation.cpp::QuatFromAngleAxis` (axis-angle: w=cos(a/2), v=axis*sin(a/2))
    # - `chrono/src/chrono/core/ChRotation.cpp::QuatFromAngleX` (calls QuatFromAngleAxis with axis=(1,0,0))


    s = math.sin(0.5 * angle)
    c = math.cos(0.5 * angle)
    return (s, 0.0, 0.0, c)  # xyzw


def _assert_allclose(testcase: unittest.TestCase, actual: np.ndarray, expected: np.ndarray, *, rtol: float, atol: float, msg: str) -> None:
    testcase.assertEqual(actual.shape, expected.shape, msg=f"{msg}: shape mismatch {actual.shape} vs {expected.shape}")
    diff = np.max(np.abs(actual - expected))
    tol = atol + rtol * np.max(np.abs(expected))
    testcase.assertLessEqual(diff, tol, msg=f"{msg}: max_abs_err={diff} tol={tol}")


def _collision_type_cases() -> list[tuple[CollisionType, str]]:
    return [
        (CollisionType.SINGLE_POINT, "single_point"),
        (CollisionType.FOUR_POINTS, "four_points"),
        (CollisionType.ENVELOPE, "envelope"),
    ]


def _plot_dir() -> Path:
    return Path(__file__).resolve().parent / "tire_rig_plots"


def _save_force_moment_plots(
    *,
    filename: str,
    t: np.ndarray,
    chrono_force: np.ndarray,
    chrono_moment: np.ndarray,
    newton_force: np.ndarray,
    newton_moment: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    out_dir = _plot_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    fig, ax = plt.subplots(3, 2, figsize=(11.0, 8.0), sharex=True)

    panels = [
        (ax[0, 0], "Longitudinal Tire Force in the Global ISO Frame", "Force (N)", chrono_force[:, 0], newton_force[:, 0]),
        (ax[0, 1], "Lateral Tire Force in the Global ISO Frame", "Force (N)", chrono_force[:, 1], newton_force[:, 1]),
        (ax[1, 0], "Vertical Tire Force in the Global ISO Frame", "Force (N)", chrono_force[:, 2], newton_force[:, 2]),
        (ax[1, 1], "X Tire Moment in the Global ISO Frame", "Moment (Nm)", chrono_moment[:, 0], newton_moment[:, 0]),
        (ax[2, 0], "Y Tire Moment in the Global ISO Frame", "Moment (Nm)", chrono_moment[:, 1], newton_moment[:, 1]),
        (ax[2, 1], "Z Tire Moment in the Global ISO Frame", "Moment (Nm)", chrono_moment[:, 2], newton_moment[:, 2]),
    ]

    for a, title, ylabel, y_chrono, y_newton in panels:
        a.plot(t, y_chrono, linestyle="--", marker="o", markersize=3.0, linewidth=1.2, label="Chrono")
        a.plot(t, y_newton, linestyle="-", marker="s", markersize=3.0, linewidth=1.2, label="Newton")
        a.set_title(title, fontsize=10)
        a.set_ylabel(ylabel)
        a.legend(fontsize=8)

    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_chrono_rig_samples(resp: dict[str, object]) -> dict[str, np.ndarray]:
    if not resp.get("ok", False):
        raise RuntimeError(resp.get("error", "Chrono rig request failed"))
    samples = resp.get("samples", None)
    if not isinstance(samples, list):
        raise RuntimeError("Chrono rig response missing samples")

    n = len(samples)
    t = np.zeros((n,), dtype=np.float64)
    slip = np.zeros((n,), dtype=np.float64)
    slip_angle = np.zeros((n,), dtype=np.float64)
    camber_angle = np.zeros((n,), dtype=np.float64)
    force = np.zeros((n, 3), dtype=np.float64)
    moment = np.zeros((n, 3), dtype=np.float64)
    pos = np.zeros((n, 3), dtype=np.float64)
    vel = np.zeros((n, 3), dtype=np.float64)
    omega_local = np.zeros((n, 3), dtype=np.float64)

    for i, s in enumerate(samples):
        assert isinstance(s, dict)
        t[i] = float(s["t"])
        slip[i] = float(s["slip"])
        slip_angle[i] = float(s["slip_angle"])
        camber_angle[i] = float(s["camber_angle"])

        tf = s["tire_force"]
        force[i, :] = np.asarray(tf["force"], dtype=np.float64)
        moment[i, :] = np.asarray(tf["moment"], dtype=np.float64)

        sp = s["spindle"]
        pos[i, :] = np.asarray(sp["pos"], dtype=np.float64)
        vel[i, :] = np.asarray(sp["vel"], dtype=np.float64)
        omega_local[i, :] = np.asarray(sp["omega_local"], dtype=np.float64)

    return {
        "t": t,
        "slip": slip,
        "slip_angle": slip_angle,
        "camber_angle": camber_angle,
        "force": force,
        "moment": moment,
        "pos": pos,
        "vel": vel,
        "omega_local": omega_local,
    }


def _build_newton_rig_model(
    *,
    tire_json: str,
    wheel_json: str,
    grav: float,
    normal_load: float,
    camber: float,
    mode: str,
    terrain_height: float,
    terrain_mu: float,
    nworld: int,
) -> tuple[newton.Model, dict[str, int]]:
    """Create a Newton model that mimics Chrono's ChTireTestRig mechanism (rig bodies + joints + global plane)."""
    dim = 0.1  # Chrono reference: ChTireTestRig.cpp::CreateMechanism

    wheel = load_chrono_vehicle_json(wheel_json)
    tire = load_chrono_vehicle_json(tire_json)

    wheel_mass = float(wheel["Mass"])
    wheel_inertia = [float(x) for x in wheel["Inertia"]]
    tire_mass = float(tire["Mass"])
    tire_inertia = [float(x) for x in tire["Inertia"]]

    mass_wt = wheel_mass + tire_mass
    inertia_wt = [wheel_inertia[i] + tire_inertia[i] for i in range(3)]

    total_mass = normal_load / grav if grav > 0.0 else 0.0
    other_mass = 2.0 * mass_wt  # slip + spindle (carrier does not contribute to normal load)
    chassis_mass_target = total_mass - other_mass
    chassis_mass = chassis_mass_target if chassis_mass_target > mass_wt else mass_wt

    base_spindle_inertia = (0.01, 0.02, 0.01)  # Chrono reference: ChTireTestRig.cpp::CreateMechanism
    spindle_inertia = [
        inertia_wt[0] + base_spindle_inertia[0],
        inertia_wt[1] + base_spindle_inertia[1],
        inertia_wt[2] + base_spindle_inertia[2],
    ]

    def _mat33_diag(ix: float, iy: float, iz: float) -> wp.mat33:
        return wp.mat33(ix, 0.0, 0.0, 0.0, iy, 0.0, 0.0, 0.0, iz)

    terrain_cfg = newton.ModelBuilder.ShapeConfig(mu=float(terrain_mu))

    # Chrono uses +Z up and gravity (0,0,-grav). Newton's builder expects the signed acceleration along up.
    main = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-float(grav))
    main.current_world = -1
    main.add_shape_plane(
        plane=(0.0, 0.0, 1.0, float(terrain_height)),
        width=0.0,
        length=0.0,
        body=-1,
        cfg=terrain_cfg,
        key="terrain",
    )

    def make_world() -> newton.ModelBuilder:
        b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-float(grav))
        # carrier, chassis, slip: mass=inertia=wheel+tire; spindle: wheel+tire plus base inertia
        carrier = b.add_link(
            xform=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            mass=mass_wt,
            I_m=_mat33_diag(*inertia_wt),
            armature=0.0,
            key="carrier",
        )
        # Ensure there is at least one world-local shape so SolverMuJoCo can pick the "first world" group when
        # operating in `separate_worlds=True` mode (it uses `shape_world` to determine the template world).
        # NOTE(separate_worlds): With `separate_worlds=True`, SolverMuJoCo converts one “template world” to a MuJoCo model
        # and then runs it batched via MuJoCo-Warp (`nworld > 1`). The template world is chosen by SolverMuJoCo as the
        # smallest non-negative id appearing in `model.shape_world` (geoms/sites). If the model has no shapes, this
        # heuristic can’t pick a proper template world, so we add a tiny `as_site=True` shape to ensure selection works.

        b.add_shape_sphere(body=carrier, radius=0.01, as_site=True, key="rig_site")
        chassis = b.add_link(
            xform=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            mass=chassis_mass,
            I_m=_mat33_diag(*inertia_wt),
            armature=0.0,
            key="chassis",
        )
        slip = b.add_link(
            xform=((0.0, 0.0, -4.0 * dim), (0.0, 0.0, 0.0, 1.0)),
            mass=mass_wt,
            I_m=_mat33_diag(*inertia_wt),
            armature=0.0,
            key="slip",
        )

        qc = _quat_from_angle_x(-camber)
        qc_inv = _quat_from_angle_x(+camber)
        wheel_body = b.add_link(
            xform=((0.0, 3.0 * dim, -4.0 * dim), qc),
            mass=mass_wt,
            I_m=_mat33_diag(*spindle_inertia),
            armature=0.0,
            key="wheel",
        )

        # Joint target gains (MuJoCo actuators created by SolverMuJoCo use these).
        if mode == "test":
            # Chrono uses ideal speed motors; use stiff velocity/position servos to approximate.
            carrier_kd = 1.0e7
            wheel_kd = 1.0e7
            slip_kp = 1.0e7
            slip_kd = 1.0e7
        else:
            carrier_kd = 0.0
            wheel_kd = 0.0
            slip_kp = 0.0
            slip_kd = 0.0

        j_carrier = b.add_joint_prismatic(
            -1,
            carrier,
            axis=(1.0, 0.0, 0.0),
            target_ke=0.0,
            target_kd=carrier_kd,
            armature=0.0,
            key="carrier_x",
        )

        j_chassis = b.add_joint_prismatic(
            carrier,
            chassis,
            axis=(0.0, 0.0, 1.0),
            target_ke=0.0,
            target_kd=0.0,
            armature=0.0,
            key="chassis_z",
        )

        joint_p = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        joint_c = ((0.0, 0.0, 4.0 * dim), (0.0, 0.0, 0.0, 1.0))
        # In Chrono, the slip body is connected through a lock joint. In Mode::TEST, the lock imposes a prescribed
        # rotation about +Z (slip angle). In DROP mode, this stays at 0 (no actuation, no lateral excitation).
        slip_limits = (None, None)
        j_slip = b.add_joint_revolute(
            chassis,
            slip,
            axis=(0.0, 0.0, 1.0),
            parent_xform=joint_p,
            child_xform=joint_c,
            target_ke=slip_kp,
            target_kd=slip_kd,
            limit_lower=slip_limits[0],
            limit_upper=slip_limits[1],
            armature=0.0,
            key="slip_yaw",
        )
        # NOTE (Lukas): with zero camber wheel spin is around Y-axis. 
        wheel_axis = (0.0, math.cos(camber), -math.sin(camber))
        j_wheel = b.add_joint_revolute(
            slip,
            wheel_body,
            axis=wheel_axis,
            parent_xform=((0.0, 3.0 * dim, 0.0), (0.0, 0.0, 0.0, 1.0)),
            child_xform=((0.0, 0.0, 0.0), qc_inv),
            target_ke=0.0,
            target_kd=wheel_kd,
            armature=0.0,
            key="wheel_spin",
        )

        b.add_articulation([j_carrier, j_chassis, j_slip, j_wheel], key="tire_test_rig")
        return b

    world_builder = make_world()
    for _ in range(int(nworld)):
        main.add_world(world_builder)

    model = main.finalize()
    axes = {
        "carrier_x": 0,
        "chassis_z": 1,
        "slip_yaw": 2,
        "wheel_spin": 3,
    }
    return model, axes


def _run_newton_rig(
    *,
    tire_json: str,
    wheel_json: str,
    mode: str,
    collision_type: CollisionType,
    dt: float,
    t_end: float,
    decimate: int,
    grav: float,
    normal_load: float,
    camber: float,
    time_delay: float,
    long_speed: float,
    ang_speed: float,
    sa_ampl: float,
    sa_freq: float,
    sa_phase: float,
    sa_shift: float,
    terrain_height: float,
    terrain_mu: float,
    nworld: int,
) -> dict[str, np.ndarray]:
    model, axes = _build_newton_rig_model(
        tire_json=tire_json,
        wheel_json=wheel_json,
        grav=grav,
        normal_load=normal_load,
        camber=camber,
        mode=mode,
        terrain_height=terrain_height,
        terrain_mu=terrain_mu,
        nworld=nworld,
    )

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    # Call Newton collision once only to get the container for stepping, we use Mujoco colission + tire module later. 
    contacts = model.collide(state_0)

    solver = SolverMuJoCo(model, use_mujoco_contacts=True)

    terrain_geom_name = None
    for gid in range(int(solver.mj_model.ngeom)):
        if int(solver.mj_model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            terrain_geom_name = mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            break
    if terrain_geom_name is None:
        raise RuntimeError("No plane geom found for tire test rig terrain.")

    module = MujocoFialaTireModule.from_mujoco_names(
        solver.mj_model,
        tire_json,
        wheel_body_names=["wheel"],
        terrain_geom_name=terrain_geom_name,
        collision_type=collision_type,
    )
    solver.add_tire_modules([module])

    wheel_id = int(mujoco.mj_name2id(solver.mj_model, mujoco.mjtObj.mjOBJ_BODY, "wheel"))
    disc_radius = float(module.tire.m_unloaded_radius)

    nsteps = int(round(t_end / dt))
    nsamp = (nsteps + decimate - 1) // decimate

    t = np.zeros((nsamp,), dtype=np.float64)
    slip = np.zeros((nworld, nsamp), dtype=np.float64)
    slip_angle = np.zeros((nworld, nsamp), dtype=np.float64)
    camber_angle = np.zeros((nworld, nsamp), dtype=np.float64)
    vx_cf_out = np.zeros((nworld, nsamp), dtype=np.float64)
    omega_out = np.zeros((nworld, nsamp), dtype=np.float64)
    depth_out = np.zeros((nworld, nsamp), dtype=np.float64)
    in_contact_out = np.zeros((nworld, nsamp), dtype=np.int32)
    force = np.zeros((nworld, nsamp, 3), dtype=np.float64)
    moment = np.zeros((nworld, nsamp, 3), dtype=np.float64)
    pos = np.zeros((nworld, nsamp, 3), dtype=np.float64)
    vel = np.zeros((nworld, nsamp, 3), dtype=np.float64)

    sample_idx = 0
    for step in range(nsteps):
        # Control based on simulation time at start-of-step (piecewise-constant over dt).
        t_now = float(step) * float(dt)
        t_eff = t_now - time_delay
        if mode == "test" and t_eff >= 0.0:
            long_v = long_speed
            omega = ang_speed
            # Chrono's ChLinkLockLock::SetMotionAng1 sign convention is opposite to MuJoCo hinge qpos.
            # Match Chrono's reported slip angle by negating the imposed slip joint angle.
            phase = 2.0 * math.pi * sa_freq * t_eff + sa_phase
            sa = -(sa_shift + sa_ampl * math.sin(phase))
            sa_dot = -(sa_ampl * (2.0 * math.pi * sa_freq) * math.cos(phase))
        else:
            long_v = 0.0
            omega = 0.0
            sa = 0.0
            sa_dot = 0.0

        dof_per_world = model.joint_dof_count // nworld
        wp.launch(
            _set_rig_controls_kernel,
            dim=nworld,
            inputs=[
                control.joint_target_pos,
                control.joint_target_vel,
                int(dof_per_world),
                int(axes["carrier_x"]),
                int(axes["slip_yaw"]),
                int(axes["wheel_spin"]),
                int(mode == "test"),
                float(long_v),
                float(omega),
                float(sa),
                float(sa_dot),
            ],
            device=model.device,
        )

        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        if (step % decimate) != 0:
            continue

        # Ensure derived fields (xipos, xmat, cvel) and tire forces are up-to-date at this state.
        mujoco_warp.forward(solver.mjw_model, solver.mjw_data)

        t_arr = solver.mjw_data.time.numpy()
        xipos = solver.mjw_data.xipos.numpy()
        xmat = solver.mjw_data.xmat.numpy()
        cvel_arr = solver.mjw_data.cvel.numpy()
        subtree_com = solver.mjw_data.subtree_com.numpy()
        body_rootid = solver.mjw_model.body_rootid.numpy()
        xfrc = solver.mjw_data.xfrc_applied.numpy()
        in_contact = module._in_contact.numpy()  # type: ignore[union-attr]
        depth = module._depth.numpy()  # type: ignore[union-attr]
        vx_cf = module._vx.numpy()  # type: ignore[union-attr]
        vy_cf = module._vy.numpy()  # type: ignore[union-attr]
        omega_w = module._omega.numpy()  # type: ignore[union-attr]
        cy = module._contact_y.numpy()  # type: ignore[union-attr]
        cz = module._contact_z.numpy()  # type: ignore[union-attr]

        t[sample_idx] = float(t_arr[0])

        for w in range(nworld):
            pos[w, sample_idx, :] = xipos[w, wheel_id, :]

            # MuJoCo `cvel` is com-based spatial velocity, expressed at the subtree COM of the kinematic tree root.
            # Convert to world linear velocity at the wheel center (Chrono spindle velocity).
            cvel = cvel_arr[w, wheel_id, :]
            omega_world = cvel[0:3]
            vel_com = cvel[3:6]
            rootid = int(body_rootid[wheel_id])
            dif = xipos[w, wheel_id, :] - subtree_com[w, rootid, :]
            vel_world = vel_com - np.cross(dif, omega_world)
            vel[w, sample_idx, :] = vel_world

            force[w, sample_idx, :] = xfrc[w, wheel_id, 0:3]
            moment[w, sample_idx, :] = xfrc[w, wheel_id, 3:6]
            vx_cf_out[w, sample_idx] = float(vx_cf[w])
            omega_out[w, sample_idx] = float(omega_w[w])
            depth_out[w, sample_idx] = float(depth[w])
            in_contact_out[w, sample_idx] = int(in_contact[w])

            # Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/test_rig/ChTireTestRig.cpp
            t_samp = float(t_arr[w])
            if t_samp < time_delay:
                slip[w, sample_idx] = 0.0
                slip_angle[w, sample_idx] = 0.0
                camber_angle[w, sample_idx] = 0.0
            else:
                # ChTireTestRig::GetLongitudinalSlip
                vx = float(math.sqrt(vel_world[0] * vel_world[0] + vel_world[1] * vel_world[1]))
                abs_vx = abs(vx)
                o = float(omega_w[w])
                slip[w, sample_idx] = ((disc_radius * o) - vx) / abs_vx if abs_vx > 1e-4 else 0.0

                # ChTireTestRig::GetSlipAngle / GetCamberAngle
                dir_y = xmat[w, wheel_id, :, 1] # get y-axis
                slip_angle[w, sample_idx] = math.atan(float(dir_y[0]) / float(dir_y[1]))
                # NOTE (Lukas): Shouldn't the camber angle here be gamma = math.atan2(-dir_y[2], math.sqrt(dir_y[0]**2 + dir_y[1]**2))?
                # Anyways, I leave it exactly as the chrono reference to exactly match the output.
                camber_angle[w, sample_idx] = math.atan(-float(dir_y[2]))

        sample_idx += 1

    return {
        "t": t,
        "slip": slip,
        "slip_angle": slip_angle,
        "camber_angle": camber_angle,
        "vx_cf": vx_cf_out,
        "omega": omega_out,
        "depth": depth_out,
        "in_contact": in_contact_out,
        "force": force,
        "moment": moment,
        "pos": pos,
        "vel": vel,
    }


class TestTireTestRig(unittest.TestCase):
    def tearDown(self) -> None:
        mujoco_warp.mj_resetCallbacks()

    def test_test_mode_matches_chrono_for_nworld_2(self):
        tire_json = _chrono_data_path("vehicle/generic/tire/FialaTire.json")
        wheel_json = _chrono_data_path("vehicle/generic/wheel/WheelSimple.json")

        dt = 1e-3 # chrono demo uses 2e-4 for NSG
        t_end = 2.0
        decimate = 10
        grav = 9.8
        normal_load = 3000.0
        camber = 0.0
        time_delay = 1.0
        long_speed = 0.2
        ang_speed = 10.0 * (2.0 * math.pi / 60.0)
        sa_ampl = 5.0 * math.pi / 180.0
        sa_freq = 0.2
        sa_phase = 0.0
        sa_shift = 0.0
        terrain_mu = 0.8

        disc_radius = float(load_chrono_vehicle_json(tire_json)["Fiala Parameters"]["Unloaded Radius"])
        terrain_height = -0.4 - disc_radius - 0.1  # Chrono reference: ChTireTestRig.cpp::CreateMechanism

        for coll_enum, coll_str in _collision_type_cases():
            with self.subTest(collision_type=coll_str):
                gt = run_chrono_gt(
                    {
                        "cmd": "tire_test_rig",
                        "mode": "test",
                        "wheel_json": wheel_json,
                        "tire_json": tire_json,
                        "dt": dt,
                        "t_end": t_end,
                        "decimate": decimate,
                        "grav": grav,
                        "normal_load": normal_load,
                        "camber": camber,
                        "time_delay": time_delay,
                        "collision_type": coll_str,
                        "long_speed": long_speed,
                        "ang_speed": ang_speed,
                        "slip_angle_ampl": sa_ampl,
                        "slip_angle_freq": sa_freq,
                        "slip_angle_phase": sa_phase,
                        "slip_angle_shift": sa_shift,
                        "terrain_mu": terrain_mu,
                    }
                )
                gt_np = _parse_chrono_rig_samples(gt)

                out = _run_newton_rig(
                    tire_json=tire_json,
                    wheel_json=wheel_json,
                    mode="test",
                    collision_type=coll_enum,
                    dt=dt,
                    t_end=t_end,
                    decimate=decimate,
                    grav=grav,
                    normal_load=normal_load,
                    camber=camber,
                    time_delay=time_delay,
                    long_speed=long_speed,
                    ang_speed=ang_speed,
                    sa_ampl=sa_ampl,
                    sa_freq=sa_freq,
                    sa_phase=sa_phase,
                    sa_shift=sa_shift,
                    terrain_height=terrain_height,
                    terrain_mu=terrain_mu,
                    nworld=2,
                )

                _save_force_moment_plots(
                    filename=f"test_mode_generic_{coll_str}_camber{camber * 180.0 / math.pi:.1f}deg.png",
                    t=gt_np["t"],
                    chrono_force=gt_np["force"],
                    chrono_moment=gt_np["moment"],
                    newton_force=out["force"][0],
                    newton_moment=out["moment"][0],
                )

                # Compare world 0 to Chrono, ensure world 1 matches world 0 (batched).
                _assert_allclose(self, out["t"], gt_np["t"], rtol=0.0, atol=1e-4, msg="time")
                # Newton uses stiff PD servos to approximate Chrono's ideal motors; compare after the system reaches steady state.
                compare_t0 = time_delay + 0.5
                mask = gt_np["t"] >= compare_t0
                _assert_allclose(self, out["slip"][0][mask], gt_np["slip"][mask], rtol=3.3e-3, atol=5e-4, msg="slip")
                _assert_allclose(
                    self, out["slip_angle"][0][mask], gt_np["slip_angle"][mask], rtol=5e-3, atol=5e-4, msg="slip_angle"
                )
                _assert_allclose(
                    self,
                    out["camber_angle"][0][mask],
                    gt_np["camber_angle"][mask],
                    rtol=5e-3,
                    atol=5e-4,
                    msg="camber_angle",
                )

                _assert_allclose(self, out["force"][0][mask], gt_np["force"][mask], rtol=3.1e-2, atol=2.0, msg="tire_force")
                _assert_allclose(self, out["moment"][0][mask], gt_np["moment"][mask], rtol=3e-2, atol=1.0, msg="tire_moment")

                _assert_allclose(self, out["slip"][1][mask], out["slip"][0][mask], rtol=0.0, atol=1e-6, msg="slip world1")
                _assert_allclose(self, out["force"][1][mask], out["force"][0][mask], rtol=0.0, atol=1e-6, msg="force world1")

    def test_drop_mode_matches_chrono_for_nworld_2(self):
        tire_json = _chrono_data_path("vehicle/generic/tire/FialaTire.json")
        wheel_json = _chrono_data_path("vehicle/generic/wheel/WheelSimple.json")

        dt = 1e-3
        t_end = 1.0
        decimate = 10
        grav = 9.8
        normal_load = 3000.0
        camber = 0.0
        time_delay = 0.0
        terrain_mu = 0.8

        disc_radius = float(load_chrono_vehicle_json(tire_json)["Fiala Parameters"]["Unloaded Radius"])
        terrain_height = -0.4 - disc_radius - 0.1  # Chrono reference: ChTireTestRig.cpp::CreateMechanism

        for coll_enum, coll_str in _collision_type_cases():
            with self.subTest(collision_type=coll_str):
                gt = run_chrono_gt(
                    {
                        "cmd": "tire_test_rig",
                        "mode": "drop",
                        "wheel_json": wheel_json,
                        "tire_json": tire_json,
                        "dt": dt,
                        "t_end": t_end,
                        "decimate": decimate,
                        "grav": grav,
                        "normal_load": normal_load,
                        "camber": camber,
                        "time_delay": time_delay,
                        "collision_type": coll_str,
                        "terrain_mu": terrain_mu,
                    }
                )
                gt_np = _parse_chrono_rig_samples(gt)

                out = _run_newton_rig(
                    tire_json=tire_json,
                    wheel_json=wheel_json,
                    mode="drop",
                    collision_type=coll_enum,
                    dt=dt,
                    t_end=t_end,
                    decimate=decimate,
                    grav=grav,
                    normal_load=normal_load,
                    camber=camber,
                    time_delay=time_delay,
                    long_speed=0.0,
                    ang_speed=0.0,
                    sa_ampl=0.0,
                    sa_freq=0.0,
                    sa_phase=0.0,
                    sa_shift=0.0,
                    terrain_height=terrain_height,
                    terrain_mu=terrain_mu,
                    nworld=2,
                )

                _save_force_moment_plots(
                    filename=f"drop_mode_generic_{coll_str}_camber{camber * 180.0 / math.pi:.1f}deg.png",
                    t=gt_np["t"],
                    chrono_force=gt_np["force"],
                    chrono_moment=gt_np["moment"],
                    newton_force=out["force"][0],
                    newton_moment=out["moment"][0],
                )

                _assert_allclose(self, out["t"], gt_np["t"], rtol=0.0, atol=1e-4, msg="time")

                # Compare summary vertical response metrics; expect looser agreement across engines.
                _assert_allclose(self, out["pos"][0][:, 2], gt_np["pos"][:, 2], rtol=3.3e-2, atol=1e-3, msg="spindle_z")
                self.assertLessEqual(
                    abs(float(np.min(out["pos"][0][:, 2])) - float(np.min(gt_np["pos"][:, 2]))),
                    2.3e-3,
                    msg="min spindle_z mismatch",
                )
                self.assertLessEqual(
                    abs(float(np.max(out["force"][0][:, 2])) - float(np.max(gt_np["force"][:, 2]))),
                    0.05 * float(np.max(gt_np["force"][:, 2])) + 10.0,
                    msg="peak normal_force mismatch",
                )

                _assert_allclose(self, out["pos"][1][:, 2], out["pos"][0][:, 2], rtol=0.0, atol=1e-6, msg="z world1")
                _assert_allclose(self, out["force"][1][:, 2], out["force"][0][:, 2], rtol=0.0, atol=1e-6, msg="fz world1")

    def test_test_mode_matches_chrono_hmmwv_for_nworld_2(self):
        tire_json = _chrono_data_path("vehicle/hmmwv/tire/HMMWV_FialaTire.json")
        wheel_json = _chrono_data_path("vehicle/hmmwv/wheel/HMMWV_Wheel.json")

        dt = 1e-3
        t_end = 2.0
        decimate = 10
        grav = 9.8
        normal_load = 3000.0
        camber = 0.0
        time_delay = 1.0
        long_speed = 0.2
        ang_speed = 10.0 * (2.0 * math.pi / 60.0)
        sa_ampl = 5.0 * math.pi / 180.0
        sa_freq = 0.2
        sa_phase = 0.0
        sa_shift = 0.0
        terrain_mu = 0.8

        disc_radius = float(load_chrono_vehicle_json(tire_json)["Fiala Parameters"]["Unloaded Radius"])
        terrain_height = -0.4 - disc_radius - 0.1  # Chrono reference: ChTireTestRig.cpp::CreateMechanism

        for coll_enum, coll_str in _collision_type_cases():
            with self.subTest(collision_type=coll_str):
                gt = run_chrono_gt(
                    {
                        "cmd": "tire_test_rig",
                        "mode": "test",
                        "wheel_json": wheel_json,
                        "tire_json": tire_json,
                        "dt": dt,
                        "t_end": t_end,
                        "decimate": decimate,
                        "grav": grav,
                        "normal_load": normal_load,
                        "camber": camber,
                        "time_delay": time_delay,
                        "collision_type": coll_str,
                        "long_speed": long_speed,
                        "ang_speed": ang_speed,
                        "slip_angle_ampl": sa_ampl,
                        "slip_angle_freq": sa_freq,
                        "slip_angle_phase": sa_phase,
                        "slip_angle_shift": sa_shift,
                        "terrain_mu": terrain_mu,
                    }
                )
                gt_np = _parse_chrono_rig_samples(gt)

                out = _run_newton_rig(
                    tire_json=tire_json,
                    wheel_json=wheel_json,
                    mode="test",
                    collision_type=coll_enum,
                    dt=dt,
                    t_end=t_end,
                    decimate=decimate,
                    grav=grav,
                    normal_load=normal_load,
                    camber=camber,
                    time_delay=time_delay,
                    long_speed=long_speed,
                    ang_speed=ang_speed,
                    sa_ampl=sa_ampl,
                    sa_freq=sa_freq,
                    sa_phase=sa_phase,
                    sa_shift=sa_shift,
                    terrain_height=terrain_height,
                    terrain_mu=terrain_mu,
                    nworld=2,
                )

                _save_force_moment_plots(
                    filename=f"test_mode_hmmwv_{coll_str}_camber{camber * 180.0 / math.pi:.1f}deg.png",
                    t=gt_np["t"],
                    chrono_force=gt_np["force"],
                    chrono_moment=gt_np["moment"],
                    newton_force=out["force"][0],
                    newton_moment=out["moment"][0],
                )

                _assert_allclose(self, out["t"], gt_np["t"], rtol=0.0, atol=1e-4, msg="time")
                compare_t0 = time_delay + 0.5
                mask = gt_np["t"] >= compare_t0
                _assert_allclose(self, out["slip"][0][mask], gt_np["slip"][mask], rtol=3e-3, atol=5e-4, msg="slip")
                _assert_allclose(
                    self, out["slip_angle"][0][mask], gt_np["slip_angle"][mask], rtol=5e-3, atol=5e-4, msg="slip_angle"
                )
                _assert_allclose(
                    self,
                    out["camber_angle"][0][mask],
                    gt_np["camber_angle"][mask],
                    rtol=5e-3,
                    atol=5e-4,
                    msg="camber_angle",
                )

                _assert_allclose(self, out["force"][0][mask], gt_np["force"][mask], rtol=2e-2, atol=3.0, msg="tire_force")
                _assert_allclose(self, out["moment"][0][mask], gt_np["moment"][mask], rtol=3e-2, atol=2.0, msg="tire_moment")

                _assert_allclose(self, out["slip"][1][mask], out["slip"][0][mask], rtol=0.0, atol=1e-6, msg="slip world1")
                _assert_allclose(self, out["force"][1][mask], out["force"][0][mask], rtol=0.0, atol=1e-6, msg="force world1")
                _assert_allclose(self, out["moment"][1][mask], out["moment"][0][mask], rtol=0.0, atol=1e-6, msg="moment world1")

    def test_test_mode_matches_chrono_with_camber_for_nworld_2(self):
        tire_json = _chrono_data_path("vehicle/generic/tire/FialaTire.json")
        wheel_json = _chrono_data_path("vehicle/generic/wheel/WheelSimple.json")

        dt = 1e-3
        t_end = 2.0
        decimate = 10
        grav = 9.8
        normal_load = 3000.0
        camber = 5.0 * math.pi / 180.0
        time_delay = 1.0
        long_speed = 0.2
        ang_speed = 10.0 * (2.0 * math.pi / 60.0)
        sa_ampl = 5.0 * math.pi / 180.0
        sa_freq = 0.2
        sa_phase = 0.0
        sa_shift = 0.0
        terrain_mu = 0.8

        disc_radius = float(load_chrono_vehicle_json(tire_json)["Fiala Parameters"]["Unloaded Radius"])
        terrain_height = -0.4 - disc_radius - 0.1  # Chrono reference: ChTireTestRig.cpp::CreateMechanism

        for coll_enum, coll_str in _collision_type_cases():
            with self.subTest(collision_type=coll_str):
                gt = run_chrono_gt(
                    {
                        "cmd": "tire_test_rig",
                        "mode": "test",
                        "wheel_json": wheel_json,
                        "tire_json": tire_json,
                        "dt": dt,
                        "t_end": t_end,
                        "decimate": decimate,
                        "grav": grav,
                        "normal_load": normal_load,
                        "camber": camber,
                        "time_delay": time_delay,
                        "collision_type": coll_str,
                        "long_speed": long_speed,
                        "ang_speed": ang_speed,
                        "slip_angle_ampl": sa_ampl,
                        "slip_angle_freq": sa_freq,
                        "slip_angle_phase": sa_phase,
                        "slip_angle_shift": sa_shift,
                        "terrain_mu": terrain_mu,
                    }
                )
                gt_np = _parse_chrono_rig_samples(gt)

                out = _run_newton_rig(
                    tire_json=tire_json,
                    wheel_json=wheel_json,
                    mode="test",
                    collision_type=coll_enum,
                    dt=dt,
                    t_end=t_end,
                    decimate=decimate,
                    grav=grav,
                    normal_load=normal_load,
                    camber=camber,
                    time_delay=time_delay,
                    long_speed=long_speed,
                    ang_speed=ang_speed,
                    sa_ampl=sa_ampl,
                    sa_freq=sa_freq,
                    sa_phase=sa_phase,
                    sa_shift=sa_shift,
                    terrain_height=terrain_height,
                    terrain_mu=terrain_mu,
                    nworld=2,
                )

                _save_force_moment_plots(
                    filename=f"test_mode_generic_{coll_str}_camber{camber * 180.0 / math.pi:.1f}deg.png",
                    t=gt_np["t"],
                    chrono_force=gt_np["force"],
                    chrono_moment=gt_np["moment"],
                    newton_force=out["force"][0],
                    newton_moment=out["moment"][0],
                )

                _assert_allclose(self, out["t"], gt_np["t"], rtol=0.0, atol=1e-4, msg="time")
                compare_t0 = time_delay + 0.5
                mask = gt_np["t"] >= compare_t0
                _assert_allclose(self, out["slip"][0][mask], gt_np["slip"][mask], rtol=5e-3, atol=1e-3, msg="slip")
                _assert_allclose(
                    self, out["slip_angle"][0][mask], gt_np["slip_angle"][mask], rtol=8e-3, atol=1e-3, msg="slip_angle"
                )
                _assert_allclose(
                    self,
                    out["camber_angle"][0][mask],
                    gt_np["camber_angle"][mask],
                    rtol=8e-3,
                    atol=1e-3,
                    msg="camber_angle",
                )

                _assert_allclose(self, out["force"][0][mask], gt_np["force"][mask], rtol=3.3e-2, atol=3.0, msg="tire_force")
                _assert_allclose(self, out["moment"][0][mask], gt_np["moment"][mask], rtol=5e-2, atol=3.0, msg="tire_moment")

                _assert_allclose(self, out["force"][1][mask], out["force"][0][mask], rtol=0.0, atol=1e-6, msg="force world1")
