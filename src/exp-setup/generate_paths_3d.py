import os
import random

LENGTH = 12          # number of moves in each path
N = 20               # how many paths to generate
# number of bends (axis-changes); 3 elbows -> 4 straight arms
N_ELBOWS = 3
MIN_ARM_LEN = 3      # minimum cubes per straight arm
SEED = 0             # for reproducible output, or None for fresh randomness
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "generated_paths_3d.txt")

PREVIOUS_PATHS = [
]

# 3D only: x/y/z moves (no 4th-dimension O/I)
_AXIS_DELTA = {
    "R": (+1, 0, 0), "L": (-1, 0, 0),
    "U": (0, +1, 0), "D": (0, -1, 0),
    "F": (0, 0, +1), "B": (0, 0, -1),
}
_OPPOSITE = {"R": "L", "L": "R", "U": "D", "D": "U", "F": "B", "B": "F"}
_AXIS_OF = {"R": "x", "L": "x", "U": "y", "D": "y", "F": "z", "B": "z"}
_ALL_MOVES = list(_AXIS_DELTA)
_N_DIMS = 3


def _build_one_path(length, rng, n_elbows, min_arm_len):
    visited = {(0, 0, 0)}
    moves = []
    used_axes = set()

    def backtrack(pos, prev_move, elbows, run_len):
        if len(moves) == length:
            # the final arm must also be long enough
            return (len(used_axes) == _N_DIMS and elbows == n_elbows
                    and run_len >= min_arm_len)

        remaining = length - len(moves)
        missing = _N_DIMS - len(used_axes)
        if remaining < missing:
            return False
        # prune: already turned too many times, or can't reach n_elbows in time
        if elbows > n_elbows or elbows + remaining < n_elbows:
            return False

        candidates = _ALL_MOVES[:]
        rng.shuffle(candidates)
        for mv in candidates:
            if prev_move is not None and mv == _OPPOSITE[prev_move]:
                continue
            nxt = tuple(p + d for p, d in zip(pos, _AXIS_DELTA[mv]))
            if nxt in visited:
                continue

            # an elbow is a turn onto a different axis
            is_elbow = prev_move is not None and _AXIS_OF[mv] != _AXIS_OF[prev_move]
            # can only turn once the current arm is long enough
            if is_elbow and run_len < min_arm_len:
                continue
            new_elbows = elbows + (1 if is_elbow else 0)
            if new_elbows > n_elbows:
                continue

            newly = _AXIS_OF[mv] not in used_axes
            if remaining == missing and not newly:
                continue

            new_run_len = 1 if is_elbow else run_len + 1

            visited.add(nxt)
            moves.append(mv)
            if newly:
                used_axes.add(_AXIS_OF[mv])

            if backtrack(nxt, mv, new_elbows, new_run_len):
                return True

            visited.discard(nxt)
            moves.pop()
            if newly:
                used_axes.discard(_AXIS_OF[mv])
        return False

    if backtrack((0, 0, 0), None, 0, 0):
        return "".join(moves)
    return None


def generate_random_paths(previous_path, length, n, n_elbows=3, min_arm_len=1,
                          seed=None, max_attempts=100000):
    if length < _N_DIMS:
        raise ValueError(
            f"length must be >= {_N_DIMS} to visit all 3 dimensions")
    # visiting 3 dimensions needs at least 2 axis-changes
    if n_elbows < _N_DIMS - 1:
        raise ValueError(
            f"n_elbows must be >= {_N_DIMS - 1} to visit all 3 dimensions")
    # n_elbows bends -> n_elbows + 1 arms, each at least min_arm_len long
    if length < (n_elbows + 1) * min_arm_len:
        raise ValueError(
            f"length {length} too short for {n_elbows + 1} arms of "
            f"min_arm_len {min_arm_len} (need >= {(n_elbows + 1) * min_arm_len})"
        )

    rng = random.Random(seed)
    seen = set(previous_path)
    result = []
    attempts = 0
    while len(result) < n:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"only generated {len(result)}/{n} unique paths in "
                f"{max_attempts} attempts (try a larger `length`)"
            )
        attempts += 1
        path = _build_one_path(length, rng, n_elbows, min_arm_len)
        if path is None or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def write_paths_file(paths, output_file):
    with open(output_file, "w") as f:
        f.write("PATHS_RANDOM_3D = [\n")
        for p in paths:
            f.write(f'    "{p}",\n')
        f.write("]\n")


if __name__ == "__main__":
    paths = generate_random_paths(
        PREVIOUS_PATHS, length=LENGTH, n=N, n_elbows=N_ELBOWS,
        min_arm_len=MIN_ARM_LEN, seed=SEED)
    write_paths_file(paths, OUTPUT_FILE)
    print(f"Wrote {len(paths)} paths to {OUTPUT_FILE}")
