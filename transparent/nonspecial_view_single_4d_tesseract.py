import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import multiprocessing
import ipywidgets as widgets
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.fx.all import crop

A = [0, 0, 0, 0]
B = [0, 0, 0, 1]
C = [0, 0, 1, 0]
D = [0, 1, 0, 0]
E = [1, 0, 0, 0]
F = [0, 0, 1, 1]
G = [0, 1, 0, 1]
H = [1, 0, 0, 1]
I = [0, 1, 1, 0]
J = [1, 0, 1, 0]
K = [1, 1, 0, 0]
L = [0, 1, 1, 1]
M = [1, 0, 1, 1]
N = [1, 1, 0, 1]
O = [1, 1, 1, 0]
P = [1, 1, 1, 1]

S = np.array(
    [
        [A, B],
        [A, D],
        [B, G],
        [D, G],
        [D, K],
        [E, K],
        [E, H],
        [B, H],
        [A, E],
        [G, N],
        [H, N],
        [K, N],
        #
        [A, C],
        [B, F],
        [D, I],
        [G, L],
        [E, J],
        [H, M],
        [N, P],
        [K, O],
        #
        [F, L],
        [F, M],
        [I, L],
        [I, O],
        [J, M],
        [J, O],
        [L, P],
        [M, P],
        [O, P],
        [C, I],
        [C, J],
        [C, F],
    ]
) - 0.5 * np.array(P)


# projection function
# This function is similar to the previous functions used for perspective
# Those performed projection in two stages (4D -> 3D -> 2D)
# while this performs the same thing in a single step (4D -> 2D)
#
def perspective_proj_4d_to_2d(v, d=5.0):
    z, w = v[2], v[3]
    factor = d**2 / ((d * d) - (d * w) - (d * z))
    return v[:2] * factor


# rotation function
def rotation_matrix_4d(i, j, theta):
    R = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


# plotting
fig, ax = plt.subplots(figsize=(5, 5))


def tesseract_anim(t):
    ax.clear()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    plt.axis("off")

    th = 2 * np.pi * t / 8
    R = rotation_matrix_4d(0, 2, th)

    Srot = S @ R.T
    Srot_proj = []

    for seg in Srot:
        # nonspecial viewpoint - different from regular shaped tesseract
        theta = np.pi / 6
        r1 = rotation_matrix_4d(0, 1, 0)
        r2 = rotation_matrix_4d(0, 2, theta)
        r3 = rotation_matrix_4d(0, 3, theta)
        r4 = rotation_matrix_4d(1, 2, theta)
        r5 = rotation_matrix_4d(1, 3, 0)
        r6 = rotation_matrix_4d(2, 3, 0)

        p0_4d = seg[0]
        p1_4d = seg[1]

        p0_4d = p0_4d @ r1.T @ r2.T @ r3.T @ r4.T @ r5.T @ r6.T
        p1_4d = p1_4d @ r1.T @ r2.T @ r3.T @ r4.T @ r5.T @ r6.T

        # 4d to 2d
        p0_2d = perspective_proj_4d_to_2d(p0_4d, d=3)
        p1_2d = perspective_proj_4d_to_2d(p1_4d, d=3)
        Srot_proj.append([p0_2d, p1_2d])

    for idx, s in enumerate(Srot_proj):
        if idx < 12:
            color = "black"
        elif idx < 20:
            color = "blue"
        else:
            color = "red"  # alternating colors (better to understand)
        ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], "-", color=color, linewidth=1)
    return mplfig_to_npimage(fig)


duration = 10
fps = 30
name = "nonspecial_transparent_tesseract_1.mp4"

animation = VideoClip(lambda t: tesseract_anim(t), duration=duration)
animation.ipython_display(fps=fps, loop=True, autoplay=True)
# animation.write_videofile(name, fps=fps)
