import os
import csv
import time
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from metzler_shape_setup import (
    TesseractOpenGL,
    shape_generator,
    mirror_shape_generator,
    t_to_deg_int,
)

SUBJECT_NAME = "raihan"
TRIAL_TYPE = "paperclip"
TRIAL_NUM = 10
SAVE_DIR_CSV = f"exp-results/{SUBJECT_NAME}/{TRIAL_TYPE}/{TRIAL_NUM}/result"
MIRROR_AXES = "x"

# PATHS_RANDOM = [
#     "UFFFLLDDDOO",
#     "UUULLLDDLLO",
#     "RRRUUURRRDD",
#     "OOORRRUURRR",
#     "RRRDDDRRRRU",
#     "LLLUUUOORRR",
#     "LLLLIIUUUUO",
#     "DDDLLLUUOLL",
#     "LLDDDDDLLII",
#     "OOORRRRIIID",
# ]

PATHS_RANDOM = ["DRRRUUUBBBO"]

PATHS = PATHS_RANDOM

T_VALUES = [
    0.000, 0.667, 1.333, 2.000, 2.667, 3.333, 4.000, 4.667,
    5.333, 6.000, 6.667, 7.333, 8.000, 8.667, 9.333, 10.000,
    10.667, 11.333, 12.000, 12.667, 13.333, 14.000, 14.667, 15.333, 16.000,
]

OBJECT_SETTINGS = dict(
    width=1200,
    height=1200,
    opacity=0.90,
    color_r=1.0,
    color_g=0.0,
    color_b=0.0,
)


def prepare_trial(trial_idx):
    path = random.choice(PATHS)

    S = shape_generator(path, start=(0, 0, 0, 0),
                        step=1.0, include_origin=True)
    S = S - np.mean(S, 0)
    S_m = mirror_shape_generator(S, axes=MIRROR_AXES)

    S = S + np.array([0.5, 0.5, 0.5, 0.5])
    S_m = S_m + np.array([0.5, 0.5, 0.5, 0.5])

    # one fixed angle per side
    t1, t2 = random.sample(T_VALUES, k=2)
    t1 += 0.311
    t2 += 0.311
    deg1, deg2 = t_to_deg_int(t1), t_to_deg_int(t2)

    is_mirrored = random.choice([0, 1])

    S_m = S_m if is_mirrored else S

    # single static projection per side
    left_pts = TesseractOpenGL(S).get_projected_centers_2d(S, t1, 7.0)
    right_pts = TesseractOpenGL(
        S_m).get_projected_centers_2d(S_m, t2, 7.0)

    return {
        "trial": trial_idx,
        "path": path,
        "deg1": deg1,
        "deg2": deg2,
        "is_mirrored": is_mirrored,
        "left_pts": left_pts,
        "right_pts": right_pts,
    }


def show_trial_and_capture(trial_data):
    mpl.rcParams["toolbar"] = "None"

    left_pts = trial_data["left_pts"]
    right_pts = trial_data["right_pts"]

    all_pts = np.concatenate([left_pts, right_pts], axis=0)
    margin = 0.5
    x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].plot(left_pts[:, 0],  left_pts[:, 1],  'o-',
                 color='red', linewidth=2, markersize=5)
    axes[1].plot(right_pts[:, 0], right_pts[:, 1], 'o-',
                 color='red', linewidth=2, markersize=5)

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

    plt.show()

    return key_holder["key"] or "", key_holder["rt"]


def show_instant_feedback(trial_data, user_answer_is_mirror, is_correct):
    mpl.rcParams["toolbar"] = "None"

    left_pts = trial_data["left_pts"]
    right_pts = trial_data["right_pts"]

    all_pts = np.concatenate([left_pts, right_pts], axis=0)
    margin = 0.5
    x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

    fig_fb, axes_fb = plt.subplots(1, 2, figsize=(10, 6))

    axes_fb[0].plot(left_pts[:, 0], left_pts[:, 1], 'o-',
                    color='red', linewidth=2, markersize=5)
    axes_fb[1].plot(right_pts[:, 0], right_pts[:, 1], 'o-',
                    color='red', linewidth=2, markersize=5)

    for ax in axes_fb:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.axis('off')

    answer_word = "MIRRORED" if user_answer_is_mirror else "SAME"
    correct_word = "MIRRORED" if trial_data["is_mirrored"] else "SAME"
    feedback_text = "Correct" if is_correct else "Incorrect"
    feedback_color = "green" if is_correct else "red"

    fig_fb.text(0.15, 0.90, "You answered:", fontsize=16, ha="left", va="center")
    fig_fb.text(0.315, 0.90, answer_word, fontsize=20, fontweight="bold", ha="left", va="center")
    fig_fb.text(0.55, 0.90, "Correct answer:", fontsize=16, ha="left", va="center")
    fig_fb.text(0.73, 0.90, correct_word, fontsize=20, fontweight="bold", ha="left", va="center")

    feedback_artist = fig_fb.text(
        0.5, 0.75, feedback_text,
        fontsize=40, color=feedback_color, fontweight="bold",
        ha="center", va="center", alpha=0.0,
    )

    fig_fb.text(0.5, 0.16, "Press ENTER to continue", fontsize=14, ha="center", va="center")

    def on_key_fb(event):
        if event.key in ("enter", "return"):
            plt.close(fig_fb)

    fig_fb.canvas.mpl_connect("key_press_event", on_key_fb)

    plt.show(block=False)

    for alpha in np.linspace(0.0, 1.0, 20):
        if not plt.fignum_exists(fig_fb.number):
            break
        feedback_artist.set_alpha(alpha)
        fig_fb.canvas.draw_idle()
        plt.pause(0.03)

    while plt.fignum_exists(fig_fb.number):
        plt.pause(0.05)


def run_gui_experiment(n_trials: int):
    os.makedirs(SAVE_DIR_CSV, exist_ok=True)
    csv_path = os.path.join(
        SAVE_DIR_CSV, f"{SUBJECT_NAME}_{TRIAL_TYPE}_{TRIAL_NUM}.csv")

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

            show_instant_feedback(trial, user_answer_is_mirror, is_correct)

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
