import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import multiprocessing
import ipywidgets as widgets
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.fx.all import crop

A = [0,0,0,0]
B = [0,0,0,1]
C = [0,0,1,0]
D = [0,1,0,0]
E = [1,0,0,0]
F = [0,0,1,1]
G = [0,1,0,1]
H = [1,0,0,1]
I = [0,1,1,0]
J = [1,0,1,0]
K = [1,1,0,0]
L = [0,1,1,1]
M = [1,0,1,1]
N = [1,1,0,1]
O = [1,1,1,0]
P = [1,1,1,1]

S = np.array([
    [A,B],[A,D],
    [B,G],[D,G],
    [D,K],[E,K],
    [E,H],[B,H],
    [A,E],[G,N],
    [H,N],[K,N],
    #
    [A,C],[B,F],
    [D,I],[G,L],
    [E,J],[H,M],
    [N,P],[K,O],
    #
    [F,L],[F,M],
    [I,L],[I,O],
    [J,M],[J,O],
    [L,P],
    [M,P],
    [O,P],
    [C,I],
    [C,J],[C,F]
])-0.5 * np.array(P)


# projection functions (4D -> 3D -> 2D)
def perspective_proj_4d_to_3d(v, d=7.0):
    w = v[3]
    factor = d / (d - w)
    return v[:3] * factor

def perspective_proj_3d_to_2d(v, d=5.0):
    z = v[2]
    factor = d / (d - z)
    return v[:2] * factor


# rotation functions
def rotation_matrix_4d(i, j, theta):
    R = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    R[[i,i,j,j],[i,j,i,j]] = [c,-s,s,c]
    return R

def rotation_matrix_3d_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c]
    ])

def rotation_matrix_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0,  0],
        [ 0,c, s],
        [ 0,-s, c]  
    ])


# plotting
fig, ax = plt.subplots(figsize=(5, 5))

def tesseract_anim(t):
    ax.clear()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    plt.axis('off')
    
    # 4D rotation
    th = 2 * np.pi * t / 8
    R2 = rotation_matrix_4d(2, 3, th) # rotating around z-w plane
    R = R2
    
    Srot = S @ R.T
    Srot_proj = []
    
    for seg in Srot:
        # 4d to 3d
        p0_3d = perspective_proj_4d_to_3d(seg[0], d=3)
        p1_3d = perspective_proj_4d_to_3d(seg[1], d=3)
        
        # special viewpoint - can see shape as regular tesseract
        theta3d_y = np.pi / 6
        R3d_y = rotation_matrix_3d_y(theta3d_y)
        theta3d_x = -np.pi / 10
        R3d_x = rotation_matrix_3d_x(theta3d_x)
        p0_3d = p0_3d @ R3d_y.T @ R3d_x.T
        p1_3d = p1_3d @ R3d_y.T @ R3d_x.T
        
        # 3d to 2d
        p0_2d = perspective_proj_3d_to_2d(p0_3d, d=3)
        p1_2d = perspective_proj_3d_to_2d(p1_3d, d=3)
        Srot_proj.append([p0_2d, p1_2d])
    
    for idx, s in enumerate(Srot_proj):
        if idx < 12:
            color = 'black' 
        elif idx<20: 
            color = 'blue' 
        else: 
            color = 'red' # alternating colors (better to understand)
        ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], '-', color = color, linewidth = 1)
    return mplfig_to_npimage(fig)


duration = 10
fps = 30
name = "transparent_tesseract_1.mp4"

animation = VideoClip(lambda t: tesseract_anim(t), duration=duration)
animation.ipython_display(fps=fps, loop=True, autoplay=True)
# animation.write_videofile(name + ".mp4", fps=fps)
