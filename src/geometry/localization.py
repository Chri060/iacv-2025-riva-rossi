import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define the vertices of the bowling lane in the 3D space (in meters)
object_points = np.array([
    [0, 0, 0],
    [18, 0, 0],
    [18, -1.07, 0],
    [0, -1.07, 0]
], dtype=np.float32)

# Define the vertices of the bowling lane in the image (in pixels)
image_points = np.array([
    [20, 625],
    [283, 299],
    [321, 298],
    [282, 640]
], dtype=np.float32)

# Import the calibration matrix
# TODO make it automatic
K = np.array([
    [1492.0959, 0.0000, 468.8429],
    [0.0000, 1494.3482, 942.5052],
    [0.0000, 0.0000, 1.0000]
], dtype=np.float32)

# Distortion coefficients (reshaped to column vector)
D = np.array([0.0837, -2.2290, -0.0180, 0.0005, 14.8918], dtype=np.float32).reshape(-1, 1)

# Remove the distortion from the points
undistorted_points = cv2.undistortPoints(image_points.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)

# Find rotation and translation vectors with PnP without distorsion
_, rotation_vector, translation_vector = cv2.solvePnP(object_points, undistorted_points, K, None, flags=cv2.SOLVEPNP_ITERATIVE)

# Convert rotation vector to rotation matrix
rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

# Compute camera position in world coordinates
camera_position = -rotation_matrix.T @ translation_vector

# Ensure correct orientation of rotation matrix
camera_orientation = rotation_matrix.T @ np.array([0, 0, 1])

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

# Plot the camera's position (red sphere)
ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='red', s=200, label='Camera Position')

# Represent camera orientation with an arrow
camera_orientation /= np.linalg.norm(camera_orientation)
ax.quiver(camera_position[0], camera_position[1], camera_position[2], camera_orientation[0], -camera_orientation[1], camera_orientation[2], color='red', length=5, linewidth=2)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set limits for better visualization
ax.set_xlim([min(lane_x) - 10, max(lane_x) + 10])
ax.set_ylim([min(lane_y) - 10, max(lane_y) + 10])
ax.set_zlim([min(lane_z) - 10, max(lane_z) + 10])

# Set the viewing angle
ax.view_init(elev=45, azim=15)

# Add a legend
ax.legend()

# Show the plot
plt.show()

# Show the camera position in command line interface
print(f"Camera Position: {camera_position.flatten()}")
