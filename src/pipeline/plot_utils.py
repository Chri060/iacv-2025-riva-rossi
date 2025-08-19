import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray
from matplotlib.axes import Axes

from pipeline.environment import Camera, BallTrajectory3d


def plot_regression(
    x: NDArray,
    y_train: NDArray,
    y_pred: NDArray,
    title: str = "Regression",
    xlabel: str = "X",
    ylabel: str = "Y",
):
    plt.figure(title)
    plt.scatter(x, y_train)
    plt.plot(x, y_pred, 'y')
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(title)
    plt.show()


def plot_3d_spline_interpolation(
    t: NDArray,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    newx: NDArray,
    newy: NDArray,
    newz: NDArray,
    name: str = "3D Spline Interpolation"
):
    f = plt.figure(name)

    ax1 = f.add_subplot(1, 4, 1)
    ax1.plot(t, newx, "y--")
    ax1.scatter(t, x)
    ax1.set_title("X spline")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("X")

    ax2 = f.add_subplot(1, 4, 2)
    ax2.plot(t, newy, "y--")
    ax2.scatter(t, y)
    ax2.set_title("Y spline")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Y")

    ax3 = f.add_subplot(1, 4, 3)
    ax3.plot(t, newz, "y--")
    ax3.scatter(t, z)
    ax3.set_title("Z spline")
    ax3.set_xlabel("Frames")
    ax3.set_ylabel("Z")

    ax4 = f.add_subplot(1, 4, 4, projection="3d")
    ax4.plot(newx, newy, newz, "y--")
    ax4.scatter(x, y, z)
    ax4.scatter([newx[0]], [newy[0]], newz[0], c="g", label="Start")
    ax4.scatter([newx[0]], [newy[0]], newz[0], c="r", label="End")
    ax4.set_title("3D trajectory")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.legend()

    plt.show()


def plot_2d_spline_interpolation(
    t: NDArray, x: NDArray, y: NDArray, newx: NDArray, newy: NDArray, name : str = "2D Spline Interpolation"
):
    f = plt.figure(name)

    ax1 = f.add_subplot(1, 3, 1)
    ax1.plot(t, newx, "y--")
    ax1.scatter(t, x)
    ax1.set_title("X spline")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("X")

    ax2 = f.add_subplot(1, 3, 2)
    ax2.plot(t, newy, "y--")
    ax2.scatter(t, y)
    ax2.set_title("Y spline")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Y")

    ax3 = f.add_subplot(1, 3, 3)
    ax3.plot(newx, newy, "y--")
    ax3.scatter(x, y)
    ax3.scatter([newx[0]], [newy[0]], c="g", label="Start")
    ax3.scatter([newx[-1]], [newy[-1]], c="r", label="End")
    ax3.set_title("2D trajectory")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.legend()
    plt.show()


# Returns figure and axes for a 3d plot
def get_3d_plot(name: str = "3D Plot") -> Axes:
    fig = plt.figure(name)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Set axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=45, azim=15)

    return ax


# Plots a bowling lane
def bowling_lane(ax: Axes, corners: NDArray, min_offset : float = 3, max_offset : float = 3):
    min = np.min(corners[:, :]) - min_offset
    max = np.max(corners[:, :]) + max_offset
    set_limits(ax, min, max)
    view_angle(ax, elev=45, azim=15)

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


# Plots a reference frame in a given 3d axes
def reference_frame(
    ax: Axes,
    pos: NDArray,
    rot: NDArray,
    colors: list[str] = ["r", "g", "b"],
    names: list[str] = ["x", "y", "z"],
    length: int = 1,
    label: str = None,
    lcolor: str = None,
):
    if lcolor is None:
        lcolor = np.random.rand(
            3,
        )

    if label:
        ax.scatter(*pos, color=lcolor, s=10, label=label)
        ax.legend()
    pos = np.array(pos).reshape(1, 3)
    rot = np.array(rot)
    for i in range(3):
        ax.quiver(*pos[0], *rot[:, i], color=colors[i], length=length)
        text_pos = pos + rot[:, i] * length
        ax.text(*text_pos[0], names[i], color=colors[i])


# Plots a camera in a given 3d axes
def camera(ax : Axes, camera : Camera, lcolor : str = None):
    reference_frame(
        ax, camera.position, camera.rotation, length=3, label=camera.name, lcolor=lcolor
    )


def trajectory(ax : Axes, trajectory : BallTrajectory3d):
    coords = trajectory.get_coords()
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    ax.plot3D(x, y, z, "green")


# Sets the limits of a 3D axes
def set_limits(ax : Axes, min : float, max: float):
    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.set_zlim(min, max)


# Sets the view angle
def view_angle(ax : Axes, elev : float, azim : float):
    ax.view_init(elev=elev, azim=azim)


# Pyplot wrapper for show()
def show():
    plt.show(block=True)
