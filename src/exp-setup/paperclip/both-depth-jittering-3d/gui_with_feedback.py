import os
import csv
import time
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from metzler_shape_setup import (
    MetzlerShape3D,
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
    PATHS_RANDOM_3D,
)

CSV_PATH = os.path.join(
    SAVE_DIR_CSV, f"{SUBJECT_NAME}_{EXP_TYPE}_{TRIAL_TYPE}_{TRIAL_NUM}.csv"
)

PATHS = PATHS_RANDOM_3D

T_VALUES = [
    0.000, 0.667, 1.333, 2.000, 2.667, 3.333, 4.000, 4.667,
    5.333, 6.000, 6.667, 7.333, 8.000, 8.667, 9.333, 10.000,
    10.667, 11.333, 12.000, 12.667, 13.333, 14.000, 14.667, 15.333, 16.000,
]

ROTATE_DELTA_T = 0.667
ROTATE_N_FRAMES = 30
ROTATE_INTERVAL_MS = 33

# depth cue: marker area for the farthest vs nearest point
MARKER_SIZE_MIN = 10
MARKER_SIZE_MAX = 300


def depth_to_sizes(depth, d_min, d_max):
    """Map per-point depth (larger = nearer) to scatter marker areas."""
    if d_max <= d_min:
        return np.full_like(np.asarray(depth, dtype=float),
                            (MARKER_SIZE_MIN + MARKER_SIZE_MAX) / 2)
    return np.interp(depth, (d_min, d_max),
                     (MARKER_SIZE_MIN, MARKER_SIZE_MAX))


def prepare_trial(trial_idx):
    path = random.choice(PATHS)

    # build shape and mirror
    S = shape_generator(path, start=(0, 0, 0),
                        step=1.0, include_origin=True)
    S = S - np.mean(S, 0)
    S_m = mirror_shape_generator(S, axes=MIRROR_AXES)

    # 3D centers
    S = S + np.array([0.5, 0.5, 0.5])
    S_m = S_m + np.array([0.5, 0.5, 0.5])

    # two distinct random angles
    t1, t2 = random.sample(T_VALUES, k=2)
    t1 += 0.311
    t2 += 0.311
    deg1, deg2 = t_to_deg_int(t1), t_to_deg_int(t2)

    is_mirrored = random.choice([0, 1])

    # jittering t values for animation (same as tesseract experiment)
    left_ts = np.linspace(t1, t1, ROTATE_N_FRAMES)
    right_ts = np.linspace(t2 - ROTATE_DELTA_T, t2 +
                           ROTATE_DELTA_T, ROTATE_N_FRAMES)

    right_centers = S_m if is_mirrored else S

    # pre-render frames for left and right (with per-point depth)
    proj = MetzlerShape3D()

    left = [proj.get_projected_centers_2d(S, t, d=10.0, return_depth=True)
            for t in left_ts]
    right = [proj.get_projected_centers_2d(right_centers, t, d=10.0,
                                           return_depth=True)
             for t in right_ts]

    left_frames = [pts for pts, _ in left]
    left_depths = [dep for _, dep in left]
    right_frames = [pts for pts, _ in right]
    right_depths = [dep for _, dep in right]

    return {
        "trial": trial_idx,
        "path": path,
        "deg1": deg1,
        "deg2": deg2,
        "is_mirrored": is_mirrored,
        "left_frames": left_frames,
        "right_frames": right_frames,
        "left_depths": left_depths,
        "right_depths": right_depths,
    }


def show_trial_and_capture(trial_data):
    mpl.rcParams["toolbar"] = "None"

    left_frames = trial_data["left_frames"]
    right_frames = trial_data["right_frames"]
    left_depths = trial_data["left_depths"]
    right_depths = trial_data["right_depths"]
    n_frames = len(left_frames)

    # compute global axis limits so shape doesn't jump around
    all_pts = np.concatenate(left_frames + right_frames, axis=0)
    margin = 0.5
    x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

    # global depth range so marker sizes stay stable across the animation
    all_depths = np.concatenate(left_depths + right_depths)
    d_min, d_max = all_depths.min(), all_depths.max()

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # line draws the connections; scatter draws depth-sized markers on top
    left_line,  = axes[0].plot([], [], '-', color='red', linewidth=2)
    right_line, = axes[1].plot([], [], '-', color='red', linewidth=2)
    left_scatter = axes[0].scatter([], [], color='black', zorder=3)
    right_scatter = axes[1].scatter([], [], color='black', zorder=3)

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

        left_scatter.set_offsets(lp)
        right_scatter.set_offsets(rp)
        left_scatter.set_sizes(depth_to_sizes(left_depths[idx], d_min, d_max))
        right_scatter.set_sizes(depth_to_sizes(
            right_depths[idx], d_min, d_max))

        return [left_line, right_line, left_scatter, right_scatter]

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


def show_instant_feedback(trial_data, user_answer_is_mirror, is_correct):
    mpl.rcParams["toolbar"] = "None"

    left_frames = trial_data["left_frames"]
    right_frames = trial_data["right_frames"]
    left_depths = trial_data["left_depths"]
    right_depths = trial_data["right_depths"]

    # use middle frame as the static representative view
    mid = len(left_frames) // 2
    left_pts = left_frames[mid]
    right_pts = right_frames[mid]

    all_pts = np.concatenate(left_frames + right_frames, axis=0)
    margin = 0.5
    x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

    all_depths = np.concatenate(left_depths + right_depths)
    d_min, d_max = all_depths.min(), all_depths.max()

    fig_fb, axes_fb = plt.subplots(1, 2, figsize=(10, 6))

    axes_fb[0].plot(left_pts[:, 0], left_pts[:, 1], '-',
                    color='red', linewidth=2)
    axes_fb[1].plot(right_pts[:, 0], right_pts[:, 1], '-',
                    color='red', linewidth=2)
    axes_fb[0].scatter(left_pts[:, 0], left_pts[:, 1], color='black', zorder=3,
                       s=depth_to_sizes(left_depths[mid], d_min, d_max))
    axes_fb[1].scatter(right_pts[:, 0], right_pts[:, 1], color='black', zorder=3,
                       s=depth_to_sizes(right_depths[mid], d_min, d_max))

    for ax in axes_fb:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.axis('off')

    answer_word = "MIRRORED" if user_answer_is_mirror else "SAME"
    correct_word = "MIRRORED" if trial_data["is_mirrored"] else "SAME"
    feedback_text = "Correct" if is_correct else "Incorrect"
    feedback_color = "green" if is_correct else "red"

    fig_fb.text(0.15, 0.90, "You answered:",
                fontsize=16, ha="left", va="center")
    fig_fb.text(0.315, 0.90, answer_word, fontsize=20,
                fontweight="bold", ha="left", va="center")
    fig_fb.text(0.55, 0.90, "Correct answer:",
                fontsize=16, ha="left", va="center")
    fig_fb.text(0.73, 0.90, correct_word, fontsize=20,
                fontweight="bold", ha="left", va="center")

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
    N_TRIALS = 10
    run_gui_experiment(N_TRIALS)
