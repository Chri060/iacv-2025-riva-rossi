import numpy as np

# World translation matrix (known)
WORLD_CENTER = np.array([
    [0],
    [0],
    [0]
])

# World rotation matrix (known)
WORLD_ROTATION = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Lane coordinates in the world (manually selected)
WORLD_LANE_CORNERS = np.array([
    [0, 0, 0],
    [12, 0, 0],
    [12, -1.07, 0],
    [0, -1.07, 0]
], dtype=np.float32)

# The vertices of the bowling lane in the N2A image reference frame
N2A_LANE_CORNERS = np.array([
    [82.0, 974.0],
    [478.0, 486.0],
    [538.0, 486.0],
    [474.0, 1002.0]
], dtype=np.float32)

# The vertices of the bowling lane in the Lumix image reference frame
LUM_LANE_CORNERS = np.array([
    [1436.0, 676.0],
    [596.0, 330.0],
    [686.0, 328.0],
    [1800.0, 642.0]
], dtype=np.float32)