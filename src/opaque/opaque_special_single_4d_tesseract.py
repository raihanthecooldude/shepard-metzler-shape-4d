import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import imageio.v3 as iio


class TesseractOpenGL:
    def __init__(self):
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

    # rotation functions
    @staticmethod
    def rotation_matrix_4d(i, j, theta):
        R = np.eye(4)
        c, s = np.cos(theta), np.sin(theta)
        R[[i, i, j, j], [i, j, i, j]] = [c, -s, s, c]
        return R

    @staticmethod
    def rotation_matrix_3d_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def rotation_matrix_3d_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

    # get new projected vertices after rotation and dimensional transformation
    def get_projected_vertices(self, t):
        th = 2 * np.pi * t / 8
        R4d = self.rotation_matrix_4d(2, 3, th)
        v4d = self.vertices_4d - self.center_shift
        v4d_rot = v4d @ R4d.T

        v3d = np.array([self.perspective_proj_4d_to_3d(v, d=3.0) for v in v4d_rot])
        R3d_y = self.rotation_matrix_3d_y(np.pi / 6)
        R3d_x = self.rotation_matrix_3d_x(-np.pi / 10)
        v3d = v3d @ R3d_y.T @ R3d_x.T

        return v3d

    # main draw function
    def draw(self, save_video=False, filename="tesseract.mp4", duration=10, fps=30):
        if not glfw.init():
            raise Exception("Could not initialize GLFW")
        win = glfw.create_window(800, 800, "4D Tesseract", None, None)
        if not win:
            glfw.terminate()
            raise Exception("Could not create window")
        glfw.make_context_current(win)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(1, 1, 1, 1)

        def set_perspective():
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(35, 1, 0.1, 100)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(0, 0, 8, 0, 0, 0, 0, 1, 0)

        set_perspective()
        t0 = time.time()

        if save_video:
            frames = []
            max_frames = int(duration * fps)
        frame_count = 0

        while not glfw.window_should_close(win):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            t = time.time() - t0

            # draw faces
            verts = self.get_projected_vertices(t)
            glColor4f(1, 0, 0, 0.80)
            for idxs in self.faces_idx:
                pts = [verts[i] for i in idxs]
                glBegin(GL_QUADS)
                for pt in pts:
                    glVertex3f(*pt)
                glEnd()

            # draw edges
            glLineWidth(2.0)
            glColor4f(0, 0, 0, 1.0)
            glBegin(GL_LINES)
            for i1, i2 in self.edges_idx:
                glVertex3f(*verts[i1])
                glVertex3f(*verts[i2])
            glEnd()

            # video saving frames
            if save_video:
                width, height = glfw.get_framebuffer_size(win)
                glPixelStorei(GL_PACK_ALIGNMENT, 1)
                data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
                image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                image = np.flipud(image)
                frames.append(image)
                frame_count += 1
                if frame_count >= max_frames:
                    break

            glfw.swap_buffers(win)
            glfw.poll_events()
        glfw.terminate()

        # save video
        if save_video:
            print(f"Saving {len(frames)} frames to {filename} ...")
            iio.imwrite(filename, frames, fps=fps)
            print("Done")


if __name__ == "__main__":
    tesseract = TesseractOpenGL()
    # tesseract.draw(save_video=True, filename="opauqe_spacial_single_tesseract.mp4", duration=30, fps=90)
    tesseract.draw(duration=30, fps=30)
