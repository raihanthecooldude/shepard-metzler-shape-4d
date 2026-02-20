import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import imageio.v3 as iio


class LShape3D:
    def __init__(self):
        self.vertices_3d = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.7, 0.0, 0.0],
                [0.7, 0.2, 0.0],
                [0.2, 0.2, 0.0],
                [0.2, 1.5, 0.0],
                [0.0, 1.5, 0.0],
            ],
            float,
        )

        center = np.mean(self.vertices_3d, axis=0)
        self.vertices_3d -= center

        self.edges_idx = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

    @staticmethod
    def rotation_matrix_y(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array(
            [
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c],
            ],
            float,
        )

    def project_one(self, t: float) -> np.ndarray:
        theta = t * 0.7
        R = self.rotation_matrix_y(theta)
        return self.vertices_3d @ R.T

    def draw(
        self,
        save_video: bool = False,
        filename: str = "L_rotate.mp4",
        duration: float = 3.0,
        fps: int = 60,
    ):

        window_w, window_h = 900, 900
        frames = []
        max_frames = int(duration * fps) if save_video else None

        if not glfw.init():
            raise Exception("Could not initialize GLFW")

        if save_video:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        win = glfw.create_window(window_w, window_h, "2D L Rotating in 3D", None, None)
        if not win:
            glfw.terminate()
            raise Exception("Could not create window")

        glfw.make_context_current(win)

        glEnable(GL_DEPTH_TEST)
        glClearColor(1, 1, 1, 1)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, 1, 0.1, 20)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 4, 0, 0, 0, 0, 1, 0)

        t0 = time.time()

        while not glfw.window_should_close(win):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if save_video:
                t = len(frames) / fps
            else:
                t = time.time() - t0

            verts = self.project_one(t)

            glLineWidth(4)
            glColor3f(0, 0, 0)
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

        if save_video and frames:
            print(f"Saving {len(frames)} frames to {filename} ...")
            iio.imwrite(filename, frames, fps=fps)
            print("Done.")


if __name__ == "__main__":
    app = LShape3D()

    app.draw()
    # app.draw(save_video=True, filename="2d_in_3d.mp4", duration=10.0, fps=60)
