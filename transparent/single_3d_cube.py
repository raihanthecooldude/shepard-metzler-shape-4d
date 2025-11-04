import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import multiprocessing
import ipywidgets as widgets
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.fx.all import crop

# %matplotlib inline # if you use this code in jupyter notebook (.ipynb) instead of a python (.py) file 

A = [0,0,0]
B = [0,0,1]
C = [0,1,0]
D = [1,0,0]
E = [0,1,1]
F = [1,1,0]
G = [1,0,1]
H = [1,1,1]

S = np.array([[A,B],[A,C],[A,D],[B,E],[B,G],[C,E],[C,F],[D,F],[D,G],[E,H],[F,H],[G,H]])-0.5*np.array(H)
# S.shape

# perspective projection function
def perspective_proj(point, d=3.0):
    x, y, z = point
    return np.array([d * x / (d - z), d * y / (d - z)])


fig, axs = plt.subplots(1, 1, figsize=(5,5))
axs.set_xlim([-2,2])
axs.axis('equal')

def plt_anim_proj(t):
    t1 = 2 * np.pi / 8
    R1 = np.array([
        [np.cos(t1), -np.sin(t1), 0],
        [np.sin(t1),  np.cos(t1), 0],
        [0, 0, 1]
    ])
    t2 = 2 * np.pi / 8
    R2 = np.array([
        [np.cos(t2), 0, -np.sin(t2)],
        [0, 1, 0],
        [np.sin(t2), 0,  np.cos(t2)]
    ])
    t3 = 2 * np.pi / 8 * t
    R3 = np.array([
        [1, 0, 0],
        [0, np.cos(t3), -np.sin(t3)],
        [0, np.sin(t3),  np.cos(t3)]
    ])

    axs.clear()
    axs.axis('equal')
    axs.set_xlim([-2, 2])
    axs.set_ylim([-2, 2])
    plt.axis('off')

    d = 4.0
    
    Srot = [S[i] @ R1 @ R2 @ R3 for i in range(12)]
    
    for p in Srot:
        p0 = perspective_proj(p[0], d)
        p1 = perspective_proj(p[1], d)
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], '-o', color='black')

    return mplfig_to_npimage(fig)

duration = 10
fps = 15
name = 'cube_3d.mp4'

animation = VideoClip(lambda t: plt_anim_proj(t), duration=duration)

animation.ipython_display(fps=fps, loop=True, autoplay=True)

# animation.write_videofile(name, fps=fps)
