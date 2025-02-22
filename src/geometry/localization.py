import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the vertices of the bowling lane in the 3d space (in meters)
object_points = np.array([
    [0, 0, 0],
    [18, 0, 0],
    [18, 0.91, 0],
    [0, 0.91, 0]
], dtype=np.float32)

# Define the vertices of the bowling lane in the image (in pixels)
image_points = np.array([
    [21, 624],
    [284, 299],
    [320, 299],
    [282, 638]
], dtype=np.float32)

# Import the calibration matrix
K = np.array([
    [1492.0959,     0.0000,         468.8429],
    [0.0000,        1494.3482,      942.5052],
    [0.0000 ,       0.0000 ,        1.0000]
], dtype=np.float32)

# Distorsion coefficients
D = np.array([ 0.0837,    -2.2290,    -0.0180,     0.0005,    14.8918 ])

# Solve PnP to estimate the camera pose (rotation and translation vectors)
_, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, K, D)

# Convert rotation vector to rotation matrix
rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

# Plotting the 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the lane (object points)
lane_x, lane_y, lane_z = object_points[:, 0], object_points[:, 1], object_points[:, 2]
ax.scatter(lane_x, lane_y, lane_z, color='blue', label='Lane Vertices')

# Draw lines to connect the lane vertices
ax.plot([object_points[0, 0], object_points[1, 0]], [object_points[0, 1], object_points[1, 1]], [object_points[0, 2], object_points[1, 2]], color='blue')
ax.plot([object_points[1, 0], object_points[2, 0]], [object_points[1, 1], object_points[2, 1]], [object_points[1, 2], object_points[2, 2]], color='blue')
ax.plot([object_points[2, 0], object_points[3, 0]], [object_points[2, 1], object_points[3, 1]], [object_points[2, 2], object_points[3, 2]], color='blue')
ax.plot([object_points[3, 0], object_points[0, 0]], [object_points[3, 1], object_points[0, 1]], [object_points[3, 2], object_points[0, 2]], color='blue')

# Plot the camera's position (translation vector)
camera_position = translation_vector.flatten()

print(f"Camera Position: {camera_position}")

# Represent the camera as a red sphere for better visibility
ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='red', s=100, label='Camera Position')

# Optionally, add an arrow to represent the camera's orientation
# The camera is looking in the direction opposite to the rotation axis
camera_direction = rotation_matrix @ np.array([0, 0, 1])  # Assuming the camera is initially facing along the z-axis
ax.quiver(camera_position[0], camera_position[1], camera_position[2], camera_direction[0], camera_direction[1], camera_direction[2], color='red', length=2)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set limits for better visualization
ax.set_xlim([min(lane_x) - 10, max(lane_x) + 10])
ax.set_ylim([min(lane_y) - 10, max(lane_y) + 10])
ax.set_zlim([min(lane_z) - 10, max(lane_z) + 10])

# Set the viewing angle
ax.view_init(elev=30, azim=30)

# Add a legend
ax.legend()

# Show the plot
plt.show()