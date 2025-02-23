import matplotlib.pyplot as plt

# Returns figure and axes for a 3d plot
def get_3d_plot():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=45, azim=15)

    return fig, ax

# Plots a bowling lane
def bowling_lane(ax, corners):
    ll, ul, ur, lr = corners[0], corners[1], corners[2], corners[3]
    ax.plot([ll[0], ul[0]], [ll[1], ul[1]], [ll[2], ul[2]], color='blue')
    ax.plot([ul[0], ur[0]], [ul[1], ur[1]], [ul[2], ur[2]], color='red')
    ax.plot([ur[0], lr[0]], [ur[1], lr[1]], [ur[2], lr[2]], color='blue')
    ax.plot([lr[0], ll[0]], [lr[1], ll[1]], [lr[2], ll[2]], color='blue')

    ax.plot_trisurf([corners[0, 0], corners[1, 0], corners[2, 0], corners[3, 0]],
                [corners[0, 1], corners[1, 1], corners[2, 1], corners[3, 1]],
                [corners[0, 2], corners[1, 2], corners[2, 2], corners[3, 2]],
                color='cyan', alpha=0.3)
    
# Plots a reference frame in a given 3d axes
def reference_frame(ax, pos, rot, colors=['r', 'g', 'b'], names=['x', 'y', 'z'], length=1, label=None, lcolor='r'):
    if label:
        print(*pos)
        ax.scatter(*pos, color=lcolor, s=10*length, label=label)
        ax.legend()
    pos = pos.reshape(1,3)
    for i in range(3):
        ax.quiver(*pos[0], *rot[:, i], color=colors[i], length=length)
        text_pos = (pos+rot[:,i]*length)[0]
        ax.text(text_pos[0], text_pos[1], text_pos[2], names[i], color=colors[i])

# Plots a camera in a given 3d axes
def camera(ax, camera_position, camera_orientation, label=None, lcolor='r'):
    reference_frame(ax, camera_position, camera_orientation, length=3, label=label, lcolor=lcolor)

# Sets the limits of a 3D axes
def set_limits(ax, min, max):
    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.set_zlim(min, max)

# Sets the view angle
def view_angle(ax, elev, azim):
    ax.view_init(elev=elev, azim=azim)

# Pyplot wrapper for show()
def show():
    plt.show()