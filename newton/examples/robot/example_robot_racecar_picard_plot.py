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
# Plots force/torque convergence across Picard iterations from the `.npz` log
# written by `example_robot_racecar_picard.py` (env `NEWTON_PICARD_LOG_PATH`).
#
# Command: python -m newton.examples robot_racecar_picard_plot --log racecar_picard_log.npz
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
    parser.add_argument(
        "--time-index",
        type=int,
        default=-1,
        help="Index into the list of unique (rounded) times to plot (default: last).",
    )
    parser.add_argument(
        "--time-round-decimals",
        type=int,
        default=9,
        help="Rounding decimals for grouping times (default: 9).",
    )
    parser.add_argument(
        "--save",
        default="",
        help="If set, save the figure to this path instead of showing it.",
    )
    args = parser.parse_args()

    log_path = os.path.expanduser(args.log)
    log = _load_log(log_path)

    times = np.asarray(log["times"], dtype=np.float64)
    iters = np.asarray(log["iters"], dtype=np.int32)
    forces = np.asarray(log["forces"], dtype=np.float64)   # (N, nwheel, 3)
    torques = np.asarray(log["torques"], dtype=np.float64)  # (N, nwheel, 3)
    wheel_names = np.asarray(log.get("wheel_names", np.arange(forces.shape[1]).astype(str)), dtype=str)

    if times.size == 0:
        raise SystemExit(f"No entries in log: {log_path}")

    time_keys = np.round(times, decimals=args.time_round_decimals)
    unique_times = np.unique(time_keys)
    if unique_times.size == 0:
        raise SystemExit(f"No unique times in log: {log_path}")

    time_index = int(args.time_index)
    if time_index < 0:
        time_index += unique_times.size
    if time_index < 0 or time_index >= unique_times.size:
        raise SystemExit(f"--time-index out of range (0..{unique_times.size-1}): {args.time_index}")

    t_key = float(unique_times[time_index])
    mask = time_keys == t_key
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise SystemExit("No samples for selected time index.")

    order = np.argsort(iters[idx])
    idx = idx[order]
    k = iters[idx]
    f_norm = np.linalg.norm(forces[idx], axis=2)   # (K, nwheel)
    tau_norm = np.linalg.norm(torques[idx], axis=2)  # (K, nwheel)

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        print(f"matplotlib is required for plotting ({e}); printing values instead.")
        print("time:", t_key)
        for wi, name in enumerate(wheel_names.tolist()):
            print(f"wheel={name}")
            for ii, kk in enumerate(k.tolist()):
                print(f"  iter={kk:2d} |F|={f_norm[ii, wi]:.6g} |tau|={tau_norm[ii, wi]:.6g}")
        return

    fig, (ax_f, ax_t) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for wi, name in enumerate(wheel_names.tolist()):
        ax_f.plot(k, f_norm[:, wi], marker="o", label=f"{name}")
        ax_t.plot(k, tau_norm[:, wi], marker="o", label=f"{name}")

    ax_f.set_title(f"Picard convergence at time={t_key}")
    ax_f.set_ylabel("||F_tire|| (world)")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc="best")

    ax_t.set_xlabel("Picard iteration index")
    ax_t.set_ylabel("||tau_tire|| (world)")
    ax_t.grid(True, alpha=0.3)

    fig.tight_layout()

    if args.save:
        out = os.path.expanduser(args.save)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved plot to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

