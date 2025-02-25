import numpy as np

# Hand-picked / Known coordinates
WORLD_CENTER = np.array([
    [0],
    [0],
    [0]
])

WORLD_ROTATION = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

WORLD_LANE_CORNERS = np.array([
    [0, 0, 0],
    [12, 0, 0],
    [12, -1.07, 0],
    [0, -1.07, 0]
], dtype=np.float32)

# The vertices of the bowling lane in the N2A image reference frame
N2A_LANE_CORNERS = np.array([
    [20, 625],
    [283, 299],
    [321, 298],
    [282, 640]
], dtype=np.float32)


# CAM_LANE_CORNERS = np.array([
# ..
# ], dtype=np.float32)