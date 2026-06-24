SUBJECT_NAME = "raihan"
EXP_TYPE = "practice"
TRIAL_TYPE = "still-vs-depth-jittering"
TRIAL_NUM = 1
SAVE_DIR_IMG = f"exp-results/{SUBJECT_NAME}/{EXP_TYPE}/{TRIAL_TYPE}/{TRIAL_NUM}/images"
SAVE_DIR_CSV = f"exp-results/{SUBJECT_NAME}/{EXP_TYPE}/{TRIAL_TYPE}/{TRIAL_NUM}/result"
MIRROR_AXES = "x"

# paths to generate shape (randomized) - 4D (uses O/I 4th-dim moves)
PATHS_RANDOM = [
    # "UFFFLLDDDOO",
    "UUULLLDDLLOFFF",
    # "RRRUUURRRDD",
    "OOORRRUUFFFRRR",
    # "RRRDDDRRRRU",
    "LLLUUUBBOOORRR",
    "LLLLIIBBBUUUUO",
    "DDDFFFLLLUUOLL",
    "LLDDDDDLLIIBBB",
    "OOORRRRIIIDDFF",
]

# PATHS_RANDOM = [
#     "UFFFLLDDDOO",
# ]

# PATHS_RANDOM = ["DRRRUUUBBBO"]

# paths to generate shape (randomized) - 3D (only R/L/U/D/F/B moves)
PATHS_RANDOM_3D = [
    "RRRUUUFFFLL",
    "UUURRRDDDFF",
    "FFFLLLUUURR",
    "LLLDDDFFFUU",
    "RRRFFFDDDLL",
    "UUUFFFLLLDD",
    "FFFRRRUUUBB",
    "DDDRRRBBBUU",
]
