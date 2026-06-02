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
    deg1, deg2 = t_to_deg_int(t1), t_to_deg_int(t2)
    is_mirrored = random.choice([0, 1])

    # Render images/videos

    # Left side: static render of S at t1 (always the original, never mirrored)
    left_png = os.path.join(SAVE_DIR_IMG, f"{path}_{deg1}.png")
    TesseractOpenGL(S).draw_image(filename=left_png, t=t1, **OBJECT_SETTINGS)
    left_img = iio.imread(left_png)

    # Right side: rotate from t2 - ROTATE_DELTA_T to t2 + ROTATE_DELTA_T (30 degree in total: 15 degree each side of t2)
    rotate_ts = np.linspace(t2 - ROTATE_DELTA_T, t2 +
                            ROTATE_DELTA_T, ROTATE_N_FRAMES)

    right_shape = S_m if is_mirrored else S

    # Render all rotate frames
    right_frames = TesseractOpenGL(right_shape).draw_video(
        t_values=rotate_ts.tolist(), **OBJECT_SETTINGS)

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
        "left_img": left_img,
        "right_frames": right_frames,
        "left_png_name": os.path.basename(left_png),
        "right_png_name": os.path.basename(right_png),
    }


def show_trial_and_capture(trial_data):
    mpl.rcParams["toolbar"] = "None"

    left_img = trial_data["left_img"]
    right_frames = trial_data["right_frames"]
    n_frames = len(right_frames)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].imshow(left_img)
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

        right_video.set_data(right_frames[idx])
        return [right_video]

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


def show_feedback(trial_data, user_answer_is_mirror, is_correct):
    mpl.rcParams["toolbar"] = "None"

    left_img = trial_data["left_img"]
    # Freeze on the last frame (shape rotated all the way to t2)
    right_img = trial_data["right_frames"][-1]

    feedback_text = "Correct" if is_correct else "Incorrect"
    feedback_color = "green" if is_correct else "red"
    answer_word = "MIRRORED" if user_answer_is_mirror else "SAME"
    correct_word = "MIRRORED" if trial_data["is_mirrored"] else "SAME"

    fig_fb, ax_fb = plt.subplots(1, 2, figsize=(10, 6))
    ax_fb[0].imshow(left_img)
    ax_fb[1].imshow(right_img)
    ax_fb[0].axis("off")
    ax_fb[1].axis("off")

    # "You answered: X" on the left, "Correct answer: Y" on the right
    fig_fb.text(0.15, 0.90, "You answered:",
                fontsize=16, ha="left", va="center")
    fig_fb.text(0.315, 0.90, answer_word,
                fontsize=20, fontweight="bold", ha="left", va="center")
    fig_fb.text(0.55, 0.90, "Correct answer:",
                fontsize=16, ha="left", va="center")
    fig_fb.text(0.73, 0.90, correct_word,
                fontsize=20, fontweight="bold", ha="left", va="center")

    # Big correct/incorrect text, fades in
    feedback_artist = fig_fb.text(
        0.5, 0.75, feedback_text,
        fontsize=40, color=feedback_color, fontweight="bold",
        ha="center", va="center", alpha=0.0,
    )

    fig_fb.text(0.5, 0.16, "Press ENTER to continue",
                fontsize=14, ha="center", va="center")

    def on_key_fb(event):
        if event.key in ("enter", "return"):
            plt.close(fig_fb)

    fig_fb.canvas.mpl_connect("key_press_event", on_key_fb)

    plt.show(block=False)

    # Fade in the big feedback text
    for alpha in np.linspace(0.0, 1.0, 20):
        if not plt.fignum_exists(fig_fb.number):
            break
        feedback_artist.set_alpha(alpha)
        fig_fb.canvas.draw_idle()
        plt.pause(0.03)

    # Wait for ENTER
    while plt.fignum_exists(fig_fb.number):
        plt.pause(0.05)


# run experiment
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

    # Run trials, capture responses, show feedback, write CSV
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()

        for trial in trials:
            user_key, rt = show_trial_and_capture(trial)

            user_answer_is_mirror = user_key == "space"
            is_correct = int(user_answer_is_mirror == trial["is_mirrored"])
            response_time_ms = int(round(rt * 1000)) if rt is not None else 0

            show_feedback(trial, user_answer_is_mirror, is_correct)

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
