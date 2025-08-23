import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes

from pipeline.environment import BallTrajectory3d, Camera


def get_3d_plot(name: str = "3D Plot") -> Axes:
    """
   Create and return a 3D Matplotlib axes object for plotting.

   This function initializes a Matplotlib figure with a 3D subplot,
   sets axis labels, defines an equal aspect ratio for the 3D box,
   and sets an initial view angle.

   Args:
       name (str): The name of the figure window. Defaults to "3D Plot".

   Returns:
       matplotlib.axes._subplots.Axes3DSubplot: The 3D axes object ready
       for plotting 3D data.
   """
    fig = plt.figure(name)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=45, azim=15)

    return ax


def bowling_lane(ax: Axes, corners: NDArray, min_offset: float = 3, max_offset: float = 3):
    """
    Plots a 3D representation of a bowling lane on the provided axes.

    The lane is drawn by connecting the four corner points with lines
    and filling the surface with a semi-transparent color. The view angle
    and axis limits are also set.

    Args:
        ax (Axes): A Matplotlib 3D axes object on which to plot the lane.
        corners (NDArray): A 4x3 array with the 3D coordinates of the lane corners
            in the order [lower-left, upper-left, upper-right, lower-right].
        min_offset (float): Minimum offset to extend the plot limits below the
            corner points. Defaults to 3.
        max_offset (float): Maximum offset to extend the plot limits above the
            corner points. Defaults to 3.

    Returns:
        None. The function modifies the provided axes in place.
    """

    set_limits(ax, np.min(corners[:, :]) - min_offset, np.max(corners[:, :]) + max_offset)
    ax.view_init(elev=45, azim=15)

    ll, ul, ur, lr = corners[0], corners[1], corners[2], corners[3]
    ax.plot([ll[0], ul[0]], [ll[1], ul[1]], [ll[2], ul[2]], color="blue")
    ax.plot([ul[0], ur[0]], [ul[1], ur[1]], [ul[2], ur[2]], color="red")
    ax.plot([ur[0], lr[0]], [ur[1], lr[1]], [ur[2], lr[2]], color="blue")
    ax.plot([lr[0], ll[0]], [lr[1], ll[1]], [lr[2], ll[2]], color="blue")

    ax.plot_trisurf(
        [corners[0][0], corners[1][0], corners[2][0], corners[3][0]],
        [corners[0][1], corners[1][1], corners[2][1], corners[3][1]],
        [corners[0][2], corners[1][2], corners[2][2], corners[3][2]],
        color="cyan",
        alpha=0.3,
    )


def reference_frame(ax: Axes, pos: list[int], rot: list[list[int]], colors=None, names=None, length: int = 1,
                    label: str = None, line_color: str = None):
    """
    Plots a 3D reference frame on the provided axes.

    The function draws the three axes of a reference frame at a specified
    position, using the provided rotation matrix to determine the directions
    of the axes. Optional labels, colors, and a legend can also be displayed.

    Args:
        ax (Axes): Matplotlib 3D axes to plot the reference frame on.
        pos (list[int]): 3D coordinates of the reference frame origin.
        rot (list[list[int]]): 3x3 rotation matrix specifying the orientation
            of the frame axes.
        colors (list[str], optional): Colors for the x, y, z axes. Defaults to ["r", "g", "b"].
        names (list[str], optional): Names/labels for the axes. Defaults to ["x", "y", "z"].
        length (int, optional): Length of axes arrows. Defaults to 1.
        label (str, optional): Label for the reference frame origin point. Adds a legend if provided.
        line_color (str, optional): Color of the origin marker. Defaults to a random RGB color.

    Returns:
        None. The function modifies the provided axes in place.
    """

    if names is None:
        names = ["x", "y", "z"]
    if colors is None:
        colors = ["r", "g", "b"]
    if line_color is None:
        line_color = np.random.rand(3)

    if label:
        ax.scatter(*pos, color=line_color, s=10, label=label)
        ax.legend()

    pos = np.array(pos).reshape(1, 3)
    rot = np.array(rot)

    for i in range(3):
        ax.quiver(*pos[0], *rot[:, i], color=colors[i], length=length)
        text_pos = pos + rot[:, i] * length
        ax.text(*text_pos[0], s=names[i], color=colors[i])


def trajectory(ax: Axes, traj: BallTrajectory3d):
    """
    Plots a 3D trajectory of a ball on the provided axes.

    The function extracts the x, y, z coordinates from a BallTrajectory3d object
    and plots the path in 3D space using a green line.

    Args:
        ax (Axes): A Matplotlib 3D axes object on which to plot the trajectory.
        traj (BallTrajectory3d): A BallTrajectory3d object containing the trajectory data.

    Returns:
        None. The function modifies the provided axes in place.
    """

    coords = traj.get_coords()
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    ax.plot3D(x, y, z, "green")


def set_limits(ax: Axes, minimum: float, maximum: float):
    """
    Set equal limits for all three axes of a 3D plot.

    This function sets the x, y, and z axis limits to the same range,
    ensuring that the 3D plot has a uniform scale.

    Args:
        ax (Axes): A Matplotlib 3D axes object on which to set the limits.
        minimum (float): Minimum value for the axes.
        maximum (float): Maximum value for the axes.

    Returns:
        None. The function modifies the provided axes in place.
    """

    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
    ax.set_zlim(minimum, maximum)


def camera(ax: Axes, camera_data: Camera, line_color: str = None):
    """
    Plots a camera as a reference frame in 3D space on the provided axes.

    This function visualizes the camera's position and orientation using
    a reference frame. The camera's rotation and position are taken from
    the provided Camera object. The axes are labeled with the camera's name.

    Args:
        ax (Axes): A Matplotlib 3D axes object on which to plot the camera.
        camera_data (Camera): A Camera object containing the following attributes:
            - position (list or array-like): 3D coordinates of the camera.
            - rotation (list[list] or array-like): 3x3 rotation matrix defining the camera orientation.
            - name (str): Name of the camera for labeling the reference frame.
        line_color (str, optional): Color of the reference frame axes. Defaults to None,
            in which case default colors are used.

    Returns:
        None. The function modifies the provided axes in place.
    """

    reference_frame(ax, camera_data.position, camera_data.rotation, length=3, label=camera_data.name, colors=line_color)
