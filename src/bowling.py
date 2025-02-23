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

INTRINSIC = False
EXTRINSIC = True

if __name__ == "__main__":
    if INTRINSIC: 
        mtx_n2a, dist_n2a = cal.intrinsic(images_path=n2a_checkerboard_path)
        display.mat(mtx_n2a, "N2A Intrinsic calibration matrix :")
        display.mat(dist_n2a, "N2A Distortion matrix :")
        mtx_cam, dist_cam = cal.intrinsic(images_path=cam_checkerboard_path)
        display.mat(mtx_n2a, "Cam Intrinsic calibration matrix :")
        display.mat(dist_n2a, "Cam Distortion matrix :")

        cam1 = Camera(mtx_n2a, None, dist_n2a)
        cam2 = Camera(mtx_cam, None, dist_cam)

        datamanager.save(cam1, "cam1")
        datamanager.save(cam2, "cam2")
    else : 
        cam1 = datamanager.load("cam1")
        cam2 = datamanager.load("cam2")

    if EXTRINSIC:
        pos, rot = cal.extrinsic(cam1, coords.WORLD_LANE_CORNERS, coords.N2A_LANE_CORNERS)

        # Plotting the 3D visualization
        fig, ax = plot.get_3d_plot()

        plot.reference_frame(ax, coords.WORLD_CENTER, coords.WORLD_ROTATION, label='World reference frame', lcolor='cyan')
        plot.bowling_lane(ax, coords.WORLD_LANE_CORNERS)

        plot.camera(ax, pos, rot, label="N2A camera")
        plot.set_limits(ax, -10, 20)
        plot.view_angle(ax, elev=45, azim=15)
        # Show the camera position in command line interface
        print(f"Camera Position: {pos.flatten()}")
        # Show the plot
        plot.show()
           

