SUBJECT_NAME = "raihan"
EXP_TYPE = "practice"
# PHASE = "phase-two"
TRIAL_TYPE = "non-jittering-4d"
TRIAL_NUM = 1
SAVE_DIR_IMG = f"exp-results/{SUBJECT_NAME}/{EXP_TYPE}/{TRIAL_TYPE}/{TRIAL_NUM}/images"
SAVE_DIR_CSV = f"exp-results/{SUBJECT_NAME}/{EXP_TYPE}/{TRIAL_TYPE}/{TRIAL_NUM}/result"
MIRROR_AXES = "x"

# paths to generate shape (randomized) - 4D (uses O/I 4th-dim moves)
PATHS_RANDOM = [
    # "UFFFLLDDDOO",
    "UULLLDDLLOOFFF",
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
#     "DDDFFFLLIIIILL",
#     "OOUUFFRRRDDDRR",
#     "RRBBLLLLOOORRU",
#     "OOORRRRIIIILLL",
#     "IIRRDDDBBBBUUU",
#     "RRRIIIDDIIIDDD",
#     "DDDOOLLLLIIIIU",
#     "IIILLUUULLLLUU",
#     "RRUUUUIIRRRROO",
#     "OOOORRRUUFFFFI",
# ]


# PATHS_RANDOM = [
#     "UFFFLLDDDOO",
# ]

# PATHS_RANDOM = ["DRRRUUUBBBO"]

# paths to generate shape (randomized) - 3D (only R/L/U/D/F/B moves)
PATHS_RANDOM_3D = [
    "RRRUUUFFL",
    "UUURRDDFF",
    "FFFLUUURR",
    "LLLDDFFUU",
    "RRRFFDDDL",
    "UUFFFLLLD",
    "FFRRRUUBB",
    "DDDRRBBUU",
    "UUUBBLLLD",
    "UUUFFRRRD",
]
