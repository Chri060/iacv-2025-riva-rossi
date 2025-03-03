import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate
import data.coords as coords

def plot3dTrajectory(image_points):
    # Define the world and image coordinates
    WORLD_LANE_CORNERS = coords.WORLD_LANE_CORNERS
    LUM_LANE_CORNERS = coords.LUM_LANE_CORNERS

    # Compute the homography matrix between image and world
    H, _ = cv2.findHomography(LUM_LANE_CORNERS, WORLD_LANE_CORNERS[:, :2])

    # Reshape the 2D points for transformation
    image_points_reshaped = image_points.reshape(-1, 1, 2)

    # Apply the homography
    world_points = cv2.perspectiveTransform(image_points_reshaped, H)

    # Extract world coordinates
    world_points = world_points.reshape(-1, 2)
    world_points = np.column_stack((world_points, np.zeros(world_points.shape[0])))

    # Fit a spline curve with a smoothing factor and no closure (open curve)
    smoothing_factor = 2
    tck, u = interpolate.splprep([world_points[:, 0], world_points[:, 1]], s=smoothing_factor, per=False)

    # Evaluate the spline at more points for a smoother curve
    unew = np.linspace(0, 1, 1000)
    out_points = interpolate.splev(unew, tck)

    # Plot the world coordinates in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the transformed points in 3D
    # ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], c='r', marker='o')

    # Plot the fitted curve in 3D
    ax.plot(out_points[0], out_points[1], np.zeros_like(out_points[0]), color='g', linestyle='-', linewidth=2)

    # Plot the lane in the world coordinates
    lane = np.vstack((WORLD_LANE_CORNERS, WORLD_LANE_CORNERS[0]))
    ax.plot(lane[:, 0], lane[:, 1], lane[:, 2], color='b', linestyle='-', linewidth=2)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bowling ball trajectory')

    # Set equal scaling
    max_range = np.max([np.ptp(world_points[:, 0]), np.ptp(world_points[:, 1]), np.ptp(world_points[:, 2])])
    mid_x = np.mean(world_points[:, 0])
    mid_y = np.mean(world_points[:, 1])
    mid_z = np.mean(world_points[:, 2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


if __name__ == "__main__":
    points = np.array([
        [1771, 600],
        [1755, 601],
        [1734, 609],
        [1706, 620],
        [1659, 617],
        [1613, 607],
        [1580, 591],
        [1544, 579],
        [1509, 568],
        [1477, 558],
        [1442, 546],
        [1412, 536],
        [1383, 526],
        [1356, 517],
        [1330, 508],
        [1305, 501],
        [1282, 494],
        [1259, 489],
        [1238, 482],
        [1219, 478],
        [1201, 472],
        [1184, 468],
        [1169, 463],
        [1154, 460],
        [1138, 455],
        [1126, 451],
        [1112, 449],
        [1100, 445],
        [1087, 442],
        [1076, 439],
        [1064, 436],
        [1053, 433],
        [1043, 430],
        [1033, 428],
        [1023, 425],
        [1014, 422],
        [1004, 419],
        [996, 418],
        [987, 414],
        [979, 412],
        [970, 409],
        [963, 408],
        [955, 407],
        [947, 410],
        [940, 408],
        [934, 406],
        [927, 406],
        [921, 402],
        [914, 404],
        [909, 402],
        [902, 403],
        [896, 400],
        [890, 398],
        [884, 399],
        [879, 400],
        [873, 400],
        [867, 399],
        [863, 398],
        [857, 397],
        [853, 396],
        [847, 396],
        [843, 396],
        [839, 394],
        [834, 392],
        [829, 390],
        [825, 388],
        [820, 388],
        [817, 385],
        [813, 385],
        [810, 382],
        [805, 381],
        [802, 379],
        [798, 379],
        [795, 375],
        [791, 375],
        [788, 375],
        [783, 370],
        [781, 370],
        [777, 371],
        [775, 368],
        [770, 367],
        [768, 366],
        [765, 368],
        [763, 364],
        [760, 365],
        [757, 364],
        [754, 360],
        [752, 362],
        [747, 362],
        [745, 360],
        [742, 362],
        [740, 360],
        [736, 359],
        [734, 359],
        [731, 358],
        [730, 358],
        [726, 358],
        [725, 358],
        [722, 358],
        [720, 357],
        [717, 356],
        [716, 357],
        [713, 354],
        [711, 356],
        [709, 354],
        [707, 352],
        [704, 346],
        [702, 349],
        [700, 348],
        [698, 348],
        [694, 347],
        [694, 348],
        [691, 346],
        [690, 345],
        [687, 344],
        [686, 344],
        [684, 343],
        [683, 344],
        [681, 345],
        [680, 345],
        [678, 343],
        [677, 343],
        [675, 343],
        [674, 342],
        [672, 342],
        [671, 340],
        [668, 341],
        [668, 340],
        [665, 341],
        [663, 339],
        [660, 333],
        [659, 333],
        [657, 332],
        [656, 332],
        [654, 331],
        [653, 331],
        [651, 331],
        [650, 331],
        [647, 330],
        [646, 330],
        [644, 329],
        [642, 329],
        [640, 329],
        [638, 328],
        [637, 328],
        [636, 328]
    ], dtype=np.float32)

    plot3dTrajectory(points)
