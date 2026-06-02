import os
import csv
import time
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio.v3 as iio

# importing from main setup file
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

# specific angle (time to degree) to show in the experiment (randomized)
T_VALUES = [
    0.000, 0.667, 1.333, 2.000, 2.667, 3.333, 4.000, 4.667,
    5.333, 6.000, 6.667, 7.333, 8.000, 8.667, 9.333, 10.000,
    10.667, 11.333, 12.000, 12.667, 13.333, 14.000, 14.667, 15.333, 16.000,
]

# Video rotate settings
ROTATE_DELTA_T = 0.667
ROTATE_N_FRAMES = 30
ROTATE_INTERVAL_MS = 33

OBJECT_SETTINGS = dict(
    width=1800,
    height=1800,
    opacity=0.80,
    color_r=1.0,
    color_g=0.0,
    color_b=0.0,
)


def prepare_trial(trial_idx):
    path = random.choice(PATHS)

    # Build original shape and mirror
    S = shape_generator(path, start=(0, 0, 0, 0),
                        step=1.0, include_origin=True)
    S = S - np.mean(S, 0)
    S_m = mirror_shape_generator(S, axes=MIRROR_AXES)

    # Two distinct times for different angles
    t1, t2 = random.sample(T_VALUES, k=2)

    # add a small angle to avoid 0 degree
    t1 = t1+0.311
    t2 = t2+0.311
    deg1, deg2 = t_to_deg_int(t1), t_to_deg_int(t2)
    is_mirrored = random.choice([0, 1])

    # Render images/videos

    # Left side: rotate from t1 - ROTATE_DELTA_T to t1 + ROTATE_DELTA_T (30 degree in total: 15 degree each side of t1)
    left_rotate_ts = np.linspace(
        t1 - ROTATE_DELTA_T, t1 + ROTATE_DELTA_T, ROTATE_N_FRAMES)

    left_shape = S

    left_frames = TesseractOpenGL(left_shape).draw_video(
        t_values=left_rotate_ts.tolist(), **OBJECT_SETTINGS)

    left_png = os.path.join(SAVE_DIR_IMG, f"{path}_{deg1}.png")
    iio.imwrite(left_png, left_frames[-1])

    # Right side: rotate from t2 - ROTATE_DELTA_T to t2 + ROTATE_DELTA_T (30 degree in total: 15 degree each side of t2)
    right_rotate_ts = np.linspace(
        t2 - ROTATE_DELTA_T, t2 + ROTATE_DELTA_T, ROTATE_N_FRAMES)

    right_shape = S_m if is_mirrored else S

    right_frames = TesseractOpenGL(right_shape).draw_video(
        t_values=right_rotate_ts.tolist(), **OBJECT_SETTINGS)

    right_png = os.path.join(SAVE_DIR_IMG, f"{path}_{deg2}.png")
    iio.imwrite(right_png, right_frames[-1])

    return {
        "trial": trial_idx,
        "path": path,
        "t1": t1,
        "t2": t2,
        "deg1": deg1,
        "deg2": deg2,
        "is_mirrored": is_mirrored,
        "left_frames": left_frames,
        "right_frames": right_frames,
        "left_png_name": os.path.basename(left_png),
        "right_png_name": os.path.basename(right_png),
    }


def show_trial_and_capture(trial_data):
    mpl.rcParams["toolbar"] = "None"

    left_frames = trial_data["left_frames"]
    right_frames = trial_data["right_frames"]
    n_frames = len(right_frames)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    left_video = ax[0].imshow(left_frames[0])
    ax[0].axis("off")

    right_video = ax[1].imshow(right_frames[0])
    ax[1].axis("off")

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
            if event.key in ("enter", "return"):
                key_holder["key"] = "enter"
            else:
                key_holder["key"] = "space"
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Animation
    cycle = 2 * (n_frames - 1)

    def update(i):
        pos = i % cycle
        if pos < n_frames:
            idx = pos
        else:
            idx = cycle - pos

        left_video.set_data(left_frames[idx])
        right_video.set_data(right_frames[idx])
        return [left_video, right_video]

    def on_draw(event):
        if key_holder["t_start"] is None:
            key_holder["t_start"] = time.perf_counter()

    fig.canvas.mpl_connect("draw_event", on_draw)

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
    os.makedirs(SAVE_DIR_IMG, exist_ok=True)
    os.makedirs(SAVE_DIR_CSV, exist_ok=True)
    csv_path = CSV_PATH

    fields = [
        "trial",
        "path",
        "left_img",
        "right_img",
        "deg1",
        "deg2",
        "delta_deg",
        "response_time_ms",
        "is_mirrored",
        "user_answer",
        "is_correct",
    ]

    # Pre-render every trial upfront
    print(f"Pre-rendering {n_trials} trials")
    t_prep_start = time.perf_counter()
    trials = []

    for trial_idx in range(1, n_trials + 1):
        trial = prepare_trial(trial_idx)
        trials.append(trial)
        print(f"  trial {trial_idx}/{n_trials} ready")
    t_prep_total = time.perf_counter() - t_prep_start
    print(f"Pre-render done in {t_prep_total:.1f}s. Starting experiment.\n")

    # Run trials, capture responses, write CSV
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()

        for trial in trials:
            user_key, rt = show_trial_and_capture(trial)

            user_answer_is_mirror = user_key == "space"
            is_correct = int(user_answer_is_mirror == trial["is_mirrored"])
            response_time_ms = int(round(rt * 1000)) if rt is not None else 0

            writer.writerow({
                "trial": trial["trial"],
                "path": trial["path"],
                "left_img": trial["left_png_name"],
                "right_img": trial["right_png_name"],
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
