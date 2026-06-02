import os
import csv
import time
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from metzler_shape_setup import (
    TesseractOpenGL,
    shape_generator,
    mirror_shape_generator,
    t_to_deg_int,
)

# experiment config
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from exp_config import (  # noqa: E402
    SUBJECT_NAME,
    EXP_TYPE,
    TRIAL_TYPE,
    TRIAL_NUM,
    SAVE_DIR_IMG,
    SAVE_DIR_CSV,
    MIRROR_AXES,
    PATHS_RANDOM,
)

CSV_PATH = os.path.join(
    SAVE_DIR_CSV, f"{SUBJECT_NAME}_{EXP_TYPE}_{TRIAL_TYPE}_{TRIAL_NUM}.csv"
)

PATHS = PATHS_RANDOM

T_VALUES = [
    0.000, 0.667, 1.333, 2.000, 2.667, 3.333, 4.000, 4.667,
    5.333, 6.000, 6.667, 7.333, 8.000, 8.667, 9.333, 10.000,
    10.667, 11.333, 12.000, 12.667, 13.333, 14.000, 14.667, 15.333, 16.000,
]

ROTATE_DELTA_T = 0.667
ROTATE_N_FRAMES = 30
ROTATE_INTERVAL_MS = 33


def prepare_trial(trial_idx):
    path = random.choice(PATHS)

    # build shape and mirror
    S = shape_generator(path, start=(0, 0, 0, 0),
                        step=1.0, include_origin=True)
    S = S - np.mean(S, 0)
    S_m = mirror_shape_generator(S, axes=MIRROR_AXES)

    # 4D centers
    S = S + np.array([0.5, 0.5, 0.5, 0.5])
    S_m = S_m + np.array([0.5, 0.5, 0.5, 0.5])

    # two distinct random angles
    t1, t2 = random.sample(T_VALUES, k=2)
    t1 += 0.311
    t2 += 0.311
    deg1, deg2 = t_to_deg_int(t1), t_to_deg_int(t2)

    is_mirrored = random.choice([0, 1])

    # jittering t values for animation (same as tesseract experiment)
    left_ts = np.linspace(t1 - ROTATE_DELTA_T, t1 +
                          ROTATE_DELTA_T, ROTATE_N_FRAMES)
    right_ts = np.linspace(t2 - ROTATE_DELTA_T, t2 +
                           ROTATE_DELTA_T, ROTATE_N_FRAMES)

    right_centers = S_m if is_mirrored else S

    # pre-render frames for left and right
    proj = TesseractOpenGL()

    left_frames = [proj.get_projected_centers_2d(
        S, t, d=10.0) for t in left_ts]
    right_frames = [proj.get_projected_centers_2d(
        right_centers, t, d=10.0) for t in right_ts]

    return {
        "trial": trial_idx,
        "path": path,
        "deg1": deg1,
        "deg2": deg2,
        "is_mirrored": is_mirrored,
        "left_frames": left_frames,
        "right_frames": right_frames,
    }


def show_trial_and_capture(trial_data):
    mpl.rcParams["toolbar"] = "None"

    left_frames = trial_data["left_frames"]
    right_frames = trial_data["right_frames"]
    n_frames = len(left_frames)

    # compute global axis limits so shape doesn't jump around
    all_pts = np.concatenate(left_frames + right_frames, axis=0)
    margin = 0.5
    x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    left_line,  = axes[0].plot(
        [], [], 'o-', color='red', linewidth=2, markersize=5)
    right_line, = axes[1].plot(
        [], [], 'o-', color='red', linewidth=2, markersize=5)

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.axis('off')

    fig.suptitle(
        "Press SPACE/BAR if MIRRORED   |   Press ENTER if SAME",
        fontsize=12,
    )

    key_holder = {"key": None, "rt": None, "t_start": None}

    def on_key(event):
        if key_holder["key"] is not None:
            return
        if event.key in ("enter", "return", " ", "space"):
            if key_holder["t_start"] is not None:
                key_holder["rt"] = time.perf_counter() - key_holder["t_start"]
            key_holder["key"] = "enter" if event.key in (
                "enter", "return") else "space"
            plt.close(fig)

    def on_draw(event):
        if key_holder["t_start"] is None:
            key_holder["t_start"] = time.perf_counter()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("draw_event", on_draw)

    cycle = 2 * (n_frames - 1)

    def update(i):
        pos = i % cycle
        idx = pos if pos < n_frames else cycle - pos

        lp = left_frames[idx]
        rp = right_frames[idx]

        left_line.set_data(lp[:, 0], lp[:, 1])
        right_line.set_data(rp[:, 0], rp[:, 1])

        return [left_line, right_line]

    anim = FuncAnimation(
        fig,
        update,
        interval=ROTATE_INTERVAL_MS,
        blit=True,
        cache_frame_data=False,
        save_count=cycle,
    )

    plt.show()
    del anim

    return key_holder["key"] or "", key_holder["rt"]


def run_gui_experiment(n_trials: int):
    os.makedirs(SAVE_DIR_CSV, exist_ok=True)
    csv_path = CSV_PATH

    fields = [
        "trial",
        "path",
        "deg1",
        "deg2",
        "delta_deg",
        "response_time_ms",
        "is_mirrored",
        "user_answer",
        "is_correct",
    ]

    print(f"Preparing {n_trials} trials...")
    trials = [prepare_trial(i) for i in range(1, n_trials + 1)]
    print("Done. Starting experiment.\n")

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()

        for trial in trials:
            user_key, rt = show_trial_and_capture(trial)

            user_answer_is_mirror = user_key == "space"
            is_correct = int(user_answer_is_mirror ==
                             bool(trial["is_mirrored"]))
            response_time_ms = int(round(rt * 1000)) if rt is not None else 0

            writer.writerow({
                "trial": trial["trial"],
                "path": trial["path"],
                "deg1": trial["deg1"],
                "deg2": trial["deg2"],
                "delta_deg": abs(trial["deg2"] - trial["deg1"]),
                "response_time_ms": response_time_ms,
                "is_mirrored": trial["is_mirrored"],
                "user_answer": int(user_answer_is_mirror),
                "is_correct": is_correct,
            })

    print(f"\nExperiment finished. CSV saved to: {csv_path}")


if __name__ == "__main__":
    N_TRIALS = 3
    run_gui_experiment(N_TRIALS)
