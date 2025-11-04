import os, csv, time, random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio.v3 as iio

# importing from main setup file
from metzler_shape_setup import (
    TesseractOpenGL,
    shape_generator,
    mirror_shape_generator,
    t_to_deg_int,
)


SUBJECT_NAME = "raihan"
TRIAL_TYPE = "practice"
TRIAL_NUM = 1
SAVE_DIR_IMG = f"exp_results/{SUBJECT_NAME}/{TRIAL_TYPE}/{TRIAL_NUM}/images"
SAVE_DIR_CSV = f"exp_results/{SUBJECT_NAME}/{TRIAL_TYPE}/{TRIAL_NUM}/result"
MIRROR_AXES = "x"

# paths to generate shape (randomized)
PATHS_RANDOM = [
    "UFFFLLDDDOO",
    "UUULLLDDLLO",
    "RRRUUURRRDD",
    "OOORRRUURRR",
    "RRRDDDRRRRU",
    "LLLUUUOORRR",
    "LLLLIIUUUUO",
    "DDDLLLUUOLL",
    "LLDDDDDLLII",
    "OOORRRRIIID",
]

PATHS = PATHS_RANDOM

# specific angle (time to degree) to show in the experiment (randomized)
T_VALUES = [
    0.000,
    0.667,
    1.333,
    2.000,
    2.667,
    3.333,
    4.000,
    4.667,
    5.333,
    6.000,
    6.667,
    7.333,
    8.000,
    8.667,
    9.333,
    10.000,
    10.667,
    11.333,
    12.000,
    12.667,
    13.333,
    14.000,
    14.667,
    15.333,
    16.000,
]

OBJECT_SETTINGS = dict(
    width=1200,
    height=1200,
    opacity=0.90,
    color_r=1.0,
    color_g=0.0,
    color_b=0.0,
)


# run experiment
def run_gui_experiment(n_trials: int):
    os.makedirs(SAVE_DIR_IMG, exist_ok=True)
    os.makedirs(SAVE_DIR_CSV, exist_ok=True)
    csv_path = os.path.join(
        SAVE_DIR_CSV, f"{SUBJECT_NAME}_{TRIAL_TYPE}_{TRIAL_NUM}.csv"
    )

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

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()

        for trial in range(1, n_trials + 1):
            path = random.choice(PATHS)

            # build original shape + mirror
            S = shape_generator(path, start=(0, 0, 0, 0), step=1.0, include_origin=True)
            S = S - np.mean(S, 0)
            S_m = mirror_shape_generator(S, axes=MIRROR_AXES)

            # two distinct times for different angles
            t1, t2 = random.sample(T_VALUES, k=2)
            deg1, deg2 = t_to_deg_int(t1), t_to_deg_int(t2)

            is_mirrored = random.choice([0, 1])

            # render images
            left_png = os.path.join(SAVE_DIR_IMG, f"{path}_{deg1}.png")
            right_png = os.path.join(SAVE_DIR_IMG, f"{path}_{deg2}.png")

            if is_mirrored:
                TesseractOpenGL(S).draw_image(
                    filename=left_png, t=t1, **OBJECT_SETTINGS
                )
                TesseractOpenGL(S_m).draw_image(
                    filename=right_png, t=t2, **OBJECT_SETTINGS
                )
            else:
                TesseractOpenGL(S).draw_image(
                    filename=left_png, t=t1, **OBJECT_SETTINGS
                )
                TesseractOpenGL(S).draw_image(
                    filename=right_png, t=t2, **OBJECT_SETTINGS
                )

            # show two shapes side by side
            # only Enter/Space allowed
            mpl.rcParams["toolbar"] = "None"

            imgL = iio.imread(left_png)
            imgR = iio.imread(right_png)
            fig, ax = plt.subplots(1, 2, figsize=(10, 6))
            ax[0].imshow(imgL)
            ax[0].axis("off")
            ax[0]
            ax[1].imshow(imgR)
            ax[1].axis("off")
            ax[1]
            fig.suptitle(
                "Press SPACE/BAR if MIRRORED   |   Press ENTER if NOT MIRRORED",
                fontsize=12,
            )

            key_holder = {"key": None, "rt": None}

            t_start = time.perf_counter()

            def on_key(event):
                if (
                    event.key in ("enter", "return", " ", "space")
                    and key_holder["key"] is None
                ):
                    key_holder["rt"] = time.perf_counter() - t_start
                    if event.key in ("enter", "return"):
                        key_holder["key"] = "enter"
                    elif event.key in (" ", "space"):
                        key_holder["key"] = "space"
                    plt.close(fig)
                # ignore everything else

            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()

            user_key = key_holder["key"] or ""
            user_answer_is_mirror = user_key == "space"
            is_correct = int(user_answer_is_mirror == is_mirrored)

            if key_holder["rt"] is not None:
                response_time_ms = int(round(key_holder["rt"] * 1000))
            else:
                response_time_ms = 0

            writer.writerow(
                {
                    "trial": trial,
                    "path": path,
                    "left_img": os.path.basename(left_png),
                    "right_img": os.path.basename(right_png),
                    "deg1": deg1,
                    "deg2": deg2,
                    "delta_deg": abs(deg2 - deg1),
                    "response_time_ms": response_time_ms,
                    "is_mirrored": is_mirrored,
                    "user_answer": int(user_answer_is_mirror),
                    "is_correct": is_correct,
                }
            )

    print(f"Experiment finished. CSV saved to: {csv_path}")


if __name__ == "__main__":
    N_TRIALS = 10
    run_gui_experiment(N_TRIALS)
