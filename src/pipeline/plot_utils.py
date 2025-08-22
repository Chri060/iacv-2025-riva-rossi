import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes

from pipeline.environment import BallTrajectory3d, Camera


def get_3d_plot(name: str = "3D Plot") -> Axes:
    """
    Returns the figure and axes for a 3d plot.
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
    Plots a bowling lane.
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
    Plots a reference frame in the given 3d axes.
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
        ax.text(*text_pos[0], names[i], color=colors[i])


def trajectory(ax: Axes, traj: BallTrajectory3d):
    """
    Plot a trajectory in 3D.
    """
    coords = traj.get_coords()
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    ax.plot3D(x, y, z, "green")


def set_limits(ax: Axes, minimum: float, maximum: float):
    """
    Sets the limits of the three-dimensional axes.
    """
    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
    ax.set_zlim(minimum, maximum)


def camera(ax: Axes, camera_data: Camera, line_color: str = None):
    """
    Plots a camera in the given axes.
    """
    reference_frame(ax, camera_data.position, camera_data.rotation, length=3, label=camera_data.name, colors=line_color)
