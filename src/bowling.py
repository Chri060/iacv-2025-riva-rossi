import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import geometry.calibration as cal
import geometry.localization as loc
import geometry.flow as flow
import utils.display as display
import utils.plot as plot
from data.camera import Camera
import data.datamanager as datamanager
import data.coords as coords

# Path where the checkerboard images are stored
n2a_checkerboard_path = "resources/calibration/nothing_2a_checkerboards" # Nothing Phone 2a
cam_checkerboard_path = "resources/calibration/cam_checkerboards" # Other camera

n2a_lane_path = "resources/localization/nothing_2a/lane.png" # Path for a n2a bowling lane sample file
cam_lane_path = "..." # Path for a cam bowling lane sample file

CALIBRATION = True
SHOW_LOCALIZATION = True

if __name__ == "__main__":
    if CALIBRATION: 
        display.title_print("Calibration")
        int_n2a, dist_n2a = cal.intrinsic(images_path=n2a_checkerboard_path, checkerboard_size=(9,6), show_detection=False)
        int_cam, dist_cam = cal.intrinsic(images_path=cam_checkerboard_path)
        int_n2a = cal.intrinsic_distortion_fix(int_n2a, dist_n2a, n2a_lane_path,
                                               show_distortion=True, 
                                               corners=np.array([coords.N2A_LANE_CORNERS[0], coords.N2A_LANE_CORNERS[1]]))
        # TODO intrinsic for cam
        # int_cam = cal.intrinsic_distortion_fix(int_cam, dist_cam, cam_lane_path,
        #                                        show_distortion=True, 
        #                                        corners=np.array([coords.N2A_LANE_CORNERS[0], coords.N2A_LANE_CORNERS[1]]))
        
        ext_n2a, pos_n2a, rot_n2a = cal.extrinsic(int_n2a, coords.WORLD_LANE_CORNERS, coords.N2A_LANE_CORNERS)
        # TODO extrinsic for cam
        # ext_cam, pos_cam, rot_cam = cal.extrinsic(int_cam, coords.WORLD_LANE_CORNERS, coords.CAM_LANE_CORNERS)

        # Store the results
        n2a_cam = Camera("Nothing 2A", int_n2a, ext_n2a, dist_n2a, pos_n2a, rot_n2a)
        cam2 = Camera("Camera", int_cam, None, dist_cam, None, None)

        print(n2a_cam)
        print(cam2)

        datamanager.save(n2a_cam, "n2a_cam")
        datamanager.save(cam2, "cam2")
    else : 
        n2a_cam = datamanager.load("n2a_cam")
        cam2 = datamanager.load("cam2")

    if SHOW_LOCALIZATION:
        display.title_print("Showing Localization Results")
        # Plotting the 3D visualization
        fig, ax = plot.get_3d_plot()

        plot.reference_frame(ax, coords.WORLD_CENTER, coords.WORLD_ROTATION, label='World reference frame', lcolor='cyan')
        plot.bowling_lane(ax, coords.WORLD_LANE_CORNERS)

        plot.camera(ax, n2a_cam, label="N2A camera")
        plot.set_limits(ax, -10, 20)
        plot.view_angle(ax, elev=45, azim=15)
        # Show the plot
        plot.show()
  
           

