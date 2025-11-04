import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import imageio.v3 as iio
import random


# shape generator
def shape_generator(
    path: str, start=(0, 0, 0, 0), step: float = 1.0, include_origin: bool = True
) -> np.ndarray:
    # update coordinates based on the direction
    dirs = {
        "R": np.array([+1, 0, 0, 0], float),
        "L": np.array([-1, 0, 0, 0], float),
        "U": np.array([0, +1, 0, 0], float),
        "D": np.array([0, -1, 0, 0], float),
        "F": np.array([0, 0, +1, 0], float),
        "B": np.array([0, 0, -1, 0], float),
        "O": np.array([0, 0, 0, +1], float),
        "I": np.array([0, 0, 0, -1], float),
    }

    pos = np.array(start, float).copy()
    shifts = []
    if include_origin:
        shifts.append(pos.copy())

    for ch in path:
        v = dirs.get(ch.upper())
        if v is None:
            continue
        pos = pos + step * v
        shifts.append(pos.copy())

    return np.array(shifts, dtype=float)


# mirror shape generator
def mirror_shape_generator(shape: np.ndarray, axes="x") -> np.ndarray:
    axis_map = {"x": 0, "y": 1, "z": 2, "w": 3}

    if isinstance(axes, str):
        axes = [a for a in axes.lower() if a in axis_map]

    sign = np.ones(4, dtype=float)
    for a in axes:
        sign[axis_map[a]] *= -1.0

    return shape * sign


# build S
def build_S(shift4: np.ndarray) -> np.ndarray:
    return np.array(shift4, dtype=float)


# get degree with respect to time
def t_to_deg_int(t: float) -> int:
    return int(round(22.5 * t)) % 360


class TesseractOpenGL:
    def __init__(self, shifts=None):
        if shifts is None:
            shifts = [np.zeros(4, float)]
        self.shifts = [np.array(s, float) for s in shifts]

        self.vertices_4d = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=float,
        )
        self.center_shift = 0.5 * np.array([1, 1, 1, 1])

        # add faces
        cube_faces = [
            # w = 0
            [
                [[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                [[0, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0]],
                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]],
                [[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0]],
                [[0, 1, 0, 0], [0, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]],
            ],
            # w = 1
            [
                [[0, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1]],
                [[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1]],
                [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 0, 1]],
                [[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]],
                [[0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]],
                [[0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]],
            ],
            # z = 0
            [
                [[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                [[0, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1]],
                [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]],
                [[1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1], [1, 1, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]],
                [[0, 1, 0, 0], [0, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 0]],
            ],
            # z = 1
            [
                [[0, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0]],
                [[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1]],
                [[0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]],
                [[1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]],
                [[0, 0, 1, 0], [0, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 0]],
                [[0, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]],
            ],
            # y = 0
            [
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1]],
                [[0, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 1], [0, 0, 1, 1]],
                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 0, 1]],
                [[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0]],
                [[0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]],
            ],
            # y = 1
            [
                [[0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [0, 1, 0, 1]],
                [[0, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 1]],
                [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 0, 1]],
                [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 1]],
                [[0, 1, 0, 0], [0, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]],
                [[0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]],
            ],
            # x = 0
            [
                [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1]],
                [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]],
                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 0, 1]],
                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]],
                [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 0, 1]],
            ],
            # x = 1
            [
                [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [1, 0, 0, 1]],
                [[1, 0, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 1, 1]],
                [[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 0, 1]],
                [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 1]],
                [[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]],
                [[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]],
            ],
        ]
        self.faces_idx = []
        for group in cube_faces:
            for face in group:
                idxs = []
                for v in face:
                    idx = np.where((self.vertices_4d == v).all(axis=1))[0][0]
                    idxs.append(idx)
                self.faces_idx.append(idxs)

        # add edges
        A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P = range(16)
        self.edges_idx = [
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
            [A, C],
            [B, F],
            [D, I],
            [G, L],
            [E, J],
            [H, M],
            [N, P],
            [K, O],
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

    # projection function
    @staticmethod
    def perspective_proj_4d_to_3d(v, d=3.0):
        w = v[3]
        factor = d / (d - w)
        return v[:3] * factor

    # this function works similar to the previous one
    # but more unbiased since this zz is dependent on both w and z (not only w)
    #
    # unbiased projection function
    @staticmethod
    def perspective_proj_4d_to_3d_prime(v, d=3.0):
        w = v[3]
        z = v[2]
        zz = w + z
        factor = d / (d - zz)
        new_v = np.array((v[0] * factor, v[1] * factor, zz))
        return new_v

    # rotation function
    @staticmethod
    def rotation_matrix_4d(i, j, theta):
        R = np.eye(4)
        c, s = np.cos(theta), np.sin(theta)
        R[[i, i, j, j], [i, j, i, j]] = [c, -s, s, c]
        return R

    # get new projected vertices after rotation and dimensional transformation
    def get_projected_vertices(self, t, shift4):
        th = 2 * np.pi * t / 16
        R4d = self.rotation_matrix_4d(0, 2, th)

        v4d = (self.vertices_4d + shift4) - self.center_shift
        v4d_rot = v4d @ R4d.T

        theta = np.pi / 6
        r1 = self.rotation_matrix_4d(0, 1, 0.0)
        r2 = self.rotation_matrix_4d(0, 2, theta)
        r3 = self.rotation_matrix_4d(0, 3, theta)
        r4 = self.rotation_matrix_4d(1, 2, theta)
        r5 = self.rotation_matrix_4d(1, 3, 0.0)
        r6 = self.rotation_matrix_4d(2, 3, 0.0)

        v4d_rot = v4d_rot @ r1.T @ r2.T @ r3.T @ r4.T @ r5.T @ r6.T

        v3d = np.array([self.perspective_proj_4d_to_3d(v, d=7.0) for v in v4d_rot])

        return v3d

    # main draw image function
    def draw_image(
        self,
        filename="tesseracts.png",
        width=1200,
        height=1200,
        t=0.0,
        opacity=1.00,
        color_r=1,
        color_g=0,
        color_b=0,
    ):
        if not glfw.init():
            raise Exception("Could not initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        win = glfw.create_window(width, height, "4D Tesseracts (Still)", None, None)
        if not win:
            glfw.terminate()
            raise Exception("Could not create window")
        glfw.make_context_current(win)

        glViewport(0, 0, width, height)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(1, 1, 1, 1)

        # set perspective and viewpoint
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(80, 1, 1.0, 20)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 8, 0, 0, 0, 0, 1, 0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for shift4 in self.shifts:
            # draw faces
            verts = self.get_projected_vertices(t, shift4)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(2.0, 2.0)
            glColor4f(color_r, color_g, color_b, opacity)
            for idxs in self.faces_idx:
                glBegin(GL_QUADS)
                for i in idxs:
                    glVertex3f(*verts[i])
                glEnd()
            glDisable(GL_POLYGON_OFFSET_FILL)

            # add edges
            glLineWidth(1.0)
            glColor4f(0, 0, 0, 1.0)
            glBegin(GL_LINES)
            for i1, i2 in self.edges_idx:
                glVertex3f(*verts[i1])
                glVertex3f(*verts[i2])
            glEnd()

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        image = np.flipud(image)
        iio.imwrite(filename, image)

        glfw.terminate()
        print(f"Saved image to {filename}")
