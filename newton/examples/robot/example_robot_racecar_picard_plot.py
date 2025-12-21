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

###########################################################################
# Example Robot Racecar Picard Log Plot
#
# Plots per-wheel convergence across Picard iterations from the `.npz` log
# written by `example_robot_racecar_picard.py` (env `NEWTON_PICARD_LOG_PATH`).
#
# One figure is created per sampled time step (see `--plot-every`) with 5 subplots:
# - tire longitudinal force `f_x`
# - tire lateral force `f_y`
# - normal force `f_n`
# - rolling resistance moment `m_roll`
# - self-aligning moment `m_align`
#
# Each subplot uses:
# - x-axis: Picard iteration index
# - one line per wheel
#
# Command:
#   python -m newton.examples robot_racecar_picard_plot --log racecar_picard_log.npz
###########################################################################

from __future__ import annotations

import argparse
import os

import numpy as np


def _load_log(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default=os.getenv("NEWTON_PICARD_LOG_PATH", "racecar_picard_log.npz"),
        help="Path to the `.npz` file saved by `robot_racecar_picard`.",
    )
    parser.add_argument("--plot-every", type=int, default=10, help="Plot every Nth unique time step (default: 10).")
    parser.add_argument("--time-index", type=int, default=None, help="If set, plot only this unique time index.")
    parser.add_argument(
        "--stage",
        default="mean",
        help=(
            "Which RK stage to plot for each time step. Use 'mean' (default) to average across stages, "
            "or an integer index (e.g. 0..3 for RK4)."
        ),
    )
    parser.add_argument(
        "--time-round-decimals",
        type=int,
        default=9,
        help="Rounding decimals for grouping times (default: 9).",
    )
    parser.add_argument(
        "--outdir",
        default="picard_plots",
        help="Directory for saving plots (default: picard_plots).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively instead of only saving them.",
    )
    args = parser.parse_args()

    log_path = os.path.expanduser(args.log)
    log = _load_log(log_path)

    times = np.asarray(log["times"], dtype=np.float64)
    iters = np.asarray(log["iters"], dtype=np.int32)
    required = ("f_n", "fx", "fy", "m_roll", "m_align")
    missing = [k for k in required if k not in log]
    if missing:
        raise SystemExit(
            f"Log is missing required fields: {', '.join(missing)}. "
            "Re-run `robot_racecar_picard` to generate a new log."
        )

    f_n = np.asarray(log["f_n"], dtype=np.float64)          # (N, nwheel)
    fx = np.asarray(log["fx"], dtype=np.float64)            # (N, nwheel)
    fy = np.asarray(log["fy"], dtype=np.float64)            # (N, nwheel)
    m_roll = np.asarray(log["m_roll"], dtype=np.float64)    # (N, nwheel)
    m_align = np.asarray(log["m_align"], dtype=np.float64)  # (N, nwheel)
    wheel_names = np.asarray(log.get("wheel_names", np.arange(fx.shape[1]).astype(str)), dtype=str)

    if times.size == 0:
        raise SystemExit(f"No entries in log: {log_path}")

    time_keys = np.round(times, decimals=args.time_round_decimals)
    unique_times = np.unique(time_keys)
    if unique_times.size == 0:
        raise SystemExit(f"No unique times in log: {log_path}")

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        print(f"matplotlib is required for plotting ({e}); printing values instead.")
        selected = [int(args.time_index)] if args.time_index is not None else list(range(0, unique_times.size, max(1, int(args.plot_every))))
        for ti in selected:
            if ti < 0:
                ti += unique_times.size
            t_key = float(unique_times[ti])
            idx = np.where(time_keys == t_key)[0]
            if idx.size == 0:
                continue

            k = iters[idx]
            print("time:", t_key)
            for wi, name in enumerate(wheel_names.tolist()):
                print(f"wheel={name}")
                for kk in np.unique(k):
                    kk = int(kk)
                    m = k == kk
                    print(
                        f"  iter={kk:2d} f_n={float(np.mean(f_n[idx[m], wi])):.6g} "
                        f"fx={float(np.mean(fx[idx[m], wi])):.6g} fy={float(np.mean(fy[idx[m], wi])):.6g} "
                        f"m_roll={float(np.mean(m_roll[idx[m], wi])):.6g} m_align={float(np.mean(m_align[idx[m], wi])):.6g}"
                    )
        return

    plot_every = max(1, int(args.plot_every))
    if args.time_index is not None:
        selected_indices = [int(args.time_index)]
    else:
        selected_indices = list(range(0, unique_times.size, plot_every))

    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    for ti in selected_indices:
        if ti < 0:
            ti += unique_times.size
        if ti < 0 or ti >= unique_times.size:
            raise SystemExit(f"--time-index out of range (0..{unique_times.size-1}): {ti}")

        t_key = float(unique_times[ti])
        idx = np.where(time_keys == t_key)[0]
        if idx.size == 0:
            continue

        idx = np.asarray(idx, dtype=np.int64)
        k_all = iters[idx]

        picard_iterations = int(log.get("picard_iterations", 0) or 0)
        if picard_iterations <= 0:
            picard_iterations = int(np.max(k_all)) + 1 if k_all.size else 1

        stage_sel = args.stage
        if isinstance(stage_sel, str) and stage_sel.lower() == "mean":
            stage_sel = "mean"
        elif isinstance(stage_sel, str):
            try:
                stage_sel = int(stage_sel)
            except ValueError:
                raise SystemExit("--stage must be 'mean' or an integer index") from None

        stage_display = stage_sel
        if stage_sel != "mean":
            if idx.size % picard_iterations != 0:
                raise SystemExit(
                    f"Cannot select a single stage: entries-per-time={idx.size} not divisible by picard_iterations={picard_iterations}"
                )
            nstages = idx.size // picard_iterations
            stage = int(stage_sel)
            if stage < 0:
                stage += nstages
            if stage < 0 or stage >= nstages:
                raise SystemExit(f"--stage out of range for this log/time (0..{nstages-1}): {stage_sel}")
            idx = idx[stage * picard_iterations : (stage + 1) * picard_iterations]
            k_all = iters[idx]
            stage_display = stage

        k_unique = np.unique(k_all)
        k_unique.sort()

        fx_k = np.zeros((k_unique.size, fx.shape[1]), dtype=np.float64)
        fy_k = np.zeros((k_unique.size, fy.shape[1]), dtype=np.float64)
        fn_k = np.zeros((k_unique.size, f_n.shape[1]), dtype=np.float64)
        mr_k = np.zeros((k_unique.size, m_roll.shape[1]), dtype=np.float64)
        ma_k = np.zeros((k_unique.size, m_align.shape[1]), dtype=np.float64)
        for ii, kk in enumerate(k_unique.tolist()):
            m = k_all == int(kk)
            fn_k[ii, :] = np.mean(f_n[idx[m], :], axis=0)
            fx_k[ii, :] = np.mean(fx[idx[m], :], axis=0)
            fy_k[ii, :] = np.mean(fy[idx[m], :], axis=0)
            mr_k[ii, :] = np.mean(m_roll[idx[m], :], axis=0)
            ma_k[ii, :] = np.mean(m_align[idx[m], :], axis=0)

        fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 10))
        ax_fx, ax_fy = axes[0]
        ax_fn, ax_mr = axes[1]
        ax_ma, ax_unused = axes[2]
        ax_unused.axis("off")

        for wi, name in enumerate(wheel_names.tolist()):
            ax_fx.plot(k_unique, fx_k[:, wi], marker="o", label=name)
            ax_fy.plot(k_unique, fy_k[:, wi], marker="o", label=name)
            ax_fn.plot(k_unique, fn_k[:, wi], marker="o", label=name)
            ax_mr.plot(k_unique, mr_k[:, wi], marker="o", label=name)
            ax_ma.plot(k_unique, ma_k[:, wi], marker="o", label=name)

        ax_fx.set_title("f_x")
        ax_fy.set_title("f_y")
        ax_fn.set_title("f_n")
        ax_mr.set_title("m_roll")
        ax_ma.set_title("m_align")

        ax_fx.set_ylabel("Force [N]")
        ax_fn.set_ylabel("Force [N]")
        ax_mr.set_ylabel("Moment [Nm]")
        ax_ma.set_ylabel("Moment [Nm]")
        ax_ma.set_xlabel("Picard iteration")
        ax_unused.set_xlabel("Picard iteration")

        for ax in (ax_fx, ax_fy, ax_fn, ax_mr, ax_ma):
            ax.grid(True, alpha=0.3)

        ax_fx.legend(loc="best", fontsize=9)
        fig.suptitle(f"Picard convergence @ time={t_key} (stage={stage_display})")
        fig.tight_layout()

        stage_tag = f"s{stage_display}" if stage_display != "mean" else "mean"
        out_path = os.path.join(outdir, f"picard_t{t_key:.6f}_idx{ti:06d}_{stage_tag}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

        if args.show:
            img = plt.imread(out_path)
            plt.figure(figsize=(11, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
