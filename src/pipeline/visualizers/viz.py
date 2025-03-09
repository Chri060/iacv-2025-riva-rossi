from pipeline.pipe import Pipe
from pipeline.environment import Environment
from pipeline.visualizers import plot_utils
import numpy as np

class Camera_Localization_Viz(Pipe):
    def execute(self, params):
        fig, ax = plot_utils.get_3d_plot('Camera placement : 3D Visualization')

        lane_coords = np.array(Environment.coords['world_lane'])
        min_offset = 3
        max_offset = 3
        min = np.min(lane_coords[:,:]) - min_offset
        max = np.max(lane_coords[:,:]) + max_offset
        plot_utils.set_limits(ax, min, max)
        plot_utils.view_angle(ax, elev=45, azim=15)

        plot_utils.bowling_lane(ax, lane_coords)

        plot_utils.reference_frame(ax, [0, 0, 0], [[1,0,0],[0,1,0],[0,0,1]], label='World reference frame', lcolor='cyan')

        for name in Environment.camera_names:
            view = Environment.get(name)
            plot_utils.camera(ax, view.camera)

        # Show the plot
        plot_utils.show()