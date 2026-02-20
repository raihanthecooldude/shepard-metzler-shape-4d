import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import imageio.v3 as iio


def shape_generator(
    path: str, start=(0, 0, 0, 0), step: float = 1.0, include_origin: bool = True
) -> np.ndarray:
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


def build_S(shift4: np.ndarray) -> np.ndarray:
    return np.array(shift4, dtype=float)


class CubePile4D:
    def __init__(self, shifts=None):
        if shifts is None:
            shifts = [np.zeros(4, float)]
        self.shifts = [np.array(s, float) for s in shifts]

        self.vertices_4d = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 0],
            ],
            dtype=float,
        )
        self.center_shift = 0.5 * np.array([1, 1, 1, 0], float)

        self.faces_idx = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],  # back/front
            [0, 3, 7, 4],
            [1, 2, 6, 5],  # left/right
            [0, 1, 5, 4],
            [3, 2, 6, 7],  # bottom/top
        ]
        self.edges_idx = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

    @staticmethod
    def rotation_matrix_4d(i, j, theta):
        R = np.eye(4)
        c, s = np.cos(theta), np.sin(theta)
        R[[i, i, j, j], [i, j, i, j]] = [c, -s, s, c]
        return R

    @staticmethod
    def perspective_proj_4d_to_3d(v, d=4.0):
        w = v[3]
        factor = d / (d - w)
        return v[:3] * factor

    @staticmethod
    def perspective_proj_4d_to_3d_zz(v, d=3.0):
        w = v[3]
        z = v[2]
        zz = w + z
        new_v = np.array((v[0], v[1], zz))
        return new_v

    def project_one(self, t, shift4):
        th2 = 2 * np.pi * t / 20
        R4d = self.rotation_matrix_4d(0, 3, th2)

        v4d = (self.vertices_4d + shift4) - self.center_shift
        v4d_rot = v4d @ R4d.T

        theta = np.pi / 6
        r1 = self.rotation_matrix_4d(0, 1, 0.0)
        r2 = self.rotation_matrix_4d(0, 2, theta)
        r3 = self.rotation_matrix_4d(0, 3, theta)
        r4 = self.rotation_matrix_4d(1, 2, 0.0)
        r5 = self.rotation_matrix_4d(1, 3, 0.0)
        r6 = self.rotation_matrix_4d(2, 3, 0.0)

        v4d_rot = v4d_rot @ r1.T @ r2.T @ r3.T @ r4.T @ r5.T @ r6.T

        v3d = np.array([self.perspective_proj_4d_to_3d_zz(v, d=5.0) for v in v4d_rot])
        return v3d

    def draw(self, save_video=False, filename="mirror_pile.mp4", duration=12, fps=60):
        if not glfw.init():
            raise Exception("Could not initialize GLFW")
        win = glfw.create_window(
            900, 900, "Pile of 3D Cubes Mirroring in 4D", None, None
        )
        if not win:
            glfw.terminate()
            raise Exception("Could not create window")
        glfw.make_context_current(win)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(1, 1, 1, 1)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, 1, 0.1, 20)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(5, 5, 10, 0, 0, 0, 0, 1, 0)

        t0 = time.time()
        frames = []
        max_frames = int(duration * fps) if save_video else None

        while not glfw.window_should_close(win):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            t = time.time() - t0

            # cube
            for shift4 in self.shifts:
                verts = self.project_one(t, shift4)

                # faces
                glEnable(GL_POLYGON_OFFSET_FILL)
                glPolygonOffset(2.0, 2.0)
                glColor4f(1, 0, 0, 0.95)
                for f in self.faces_idx:
                    glBegin(GL_QUADS)
                    for i in f:
                        glVertex3f(*verts[i])
                    glEnd()
                glDisable(GL_POLYGON_OFFSET_FILL)

                # edges
                glLineWidth(1.4)
                glColor4f(0, 0, 0, 1)
                glBegin(GL_LINES)
                for i, j in self.edges_idx:
                    glVertex3f(*verts[i])
                    glVertex3f(*verts[j])
                glEnd()

            if save_video:
                w, h = glfw.get_framebuffer_size(win)
                glPixelStorei(GL_PACK_ALIGNMENT, 1)
                data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
                img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                frames.append(np.flipud(img))
                if len(frames) >= max_frames:
                    break

            glfw.swap_buffers(win)
            glfw.poll_events()

        glfw.terminate()

        if save_video:
            print(f"Saving {len(frames)} frames to {filename} ...")
            iio.imwrite(filename, frames, fps=fps)
            print("Done.")


if __name__ == "__main__":
    path = "UUBBBLLLD"
    S = shape_generator(path, start=(0, 0, 0, 0), step=1.0, include_origin=True)
    S = S - np.mean(S, 0)

    app = CubePile4D(S)
    # app.draw(save_video=True, filename="3d_in_4d.mp4", duration=90, fps=90)
    app.draw(fps=60)
