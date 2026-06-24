import numpy as np


# shape generator (3D)
def shape_generator(
    path: str, start=(0, 0, 0), step: float = 1.0, include_origin: bool = True
) -> np.ndarray:
    # update coordinates based on the direction
    dirs = {
        "R": np.array([+1, 0, 0], float),
        "L": np.array([-1, 0, 0], float),
        "U": np.array([0, +1, 0], float),
        "D": np.array([0, -1, 0], float),
        "F": np.array([0, 0, +1], float),
        "B": np.array([0, 0, -1], float),
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


# mirror shape generator (3D)
def mirror_shape_generator(shape: np.ndarray, axes="x") -> np.ndarray:
    axis_map = {"x": 0, "y": 1, "z": 2}

    if isinstance(axes, str):
        axes = [a for a in axes.lower() if a in axis_map]

    sign = np.ones(3, dtype=float)
    for a in axes:
        sign[axis_map[a]] *= -1.0

    return shape * sign


# get degree with respect to time
def t_to_deg_int(t: float) -> int:
    return int(round(22.5 * t)) % 360


class MetzlerShape3D:
    # 3d to 2d conversion
    @staticmethod
    def perspective_proj_3d_to_2d(v, d=7.0, return_depth=False):
        x, y, z = v[0], v[1], v[2]

        factor = d / (d - z)
        x2 = x * factor
        y2 = y * factor

        if return_depth:
            # perspective magnification: larger => nearer the camera
            return np.array([x2, y2]), factor
        return np.array([x2, y2])

    # rotation function
    @staticmethod
    def rotation_matrix_3d(i, j, theta):
        R = np.eye(3)
        c, s = np.cos(theta), np.sin(theta)
        R[[i, i, j, j], [i, j, i, j]] = [c, -s, s, c]
        return R

    def get_projected_centers_2d(self, centers_3d, t, d=7.0, return_depth=False):
        th = 2 * np.pi * t / 16
        R_time = self.rotation_matrix_3d(0, 2, th)

        theta = np.pi / 6
        pitch = self.rotation_matrix_3d(1, 2, theta)

        rotated = centers_3d @ R_time.T @ pitch.T

        if return_depth:
            projected = [
                self.perspective_proj_3d_to_2d(c, d=d, return_depth=True)
                for c in rotated
            ]
            pts = np.array([p for p, _ in projected])
            depth = np.array([dep for _, dep in projected])
            return pts, depth

        return np.array([self.perspective_proj_3d_to_2d(c, d=d) for c in rotated])
