import geometry.calibration as cal
import utils.display as display
import utils.plot as plot
from data.camera import Camera
import data.datamanager as datamanager
import data.coords as coords

# Path where the checkerboard images are stored
n2a_checkerboard_path = "resources/calibration/nothing_2a_checkerboards" # Nothing Phone 2a
lum_checkerboard_path = "resources/calibration/lumix_checkerboards" # Other camera

n2a_lane_path = "resources/localization/nothing_2a/lane.png" # Path for a n2a bowling lane sample file
lum_lane_path = "resources/localization/lumix/lane.png" # Path for a lum bowling lane sample file

CALIBRATION = True
SHOW_LOCALIZATION = True

if __name__ == "__main__":
    if CALIBRATION: 
        display.title_print("Calibration")
        int_n2a, dist_n2a = cal.intrinsic(images_path=n2a_checkerboard_path, checkerboard_size=(9,6), show_detection=False)
        int_n2a, undist_corners_n2a = cal.intrinsic_distortion_fix(int_n2a, dist_n2a, n2a_lane_path,
                                               show_distortion=True, 
                                               corners=coords.N2A_LANE_CORNERS)

        int_lum, dist_lum = cal.intrinsic(images_path=lum_checkerboard_path, checkerboard_size=(9,6), show_detection=False)
        int_lum, undist_corners_lum = cal.intrinsic_distortion_fix(int_lum, dist_lum, lum_lane_path,
                                               show_distortion=True, 
                                               corners=coords.LUM_LANE_CORNERS)
        print(undist_corners_lum)
        
        ext_n2a, pos_n2a, rot_n2a = cal.extrinsic(int_n2a, coords.WORLD_LANE_CORNERS, undist_corners_n2a)
        ext_lum, pos_lum, rot_lum = cal.extrinsic(int_lum, coords.WORLD_LANE_CORNERS, undist_corners_lum)

        # Store the results
        n2a_cam = Camera("Nothing 2A", int_n2a, ext_n2a, dist_n2a, pos_n2a, rot_n2a)
        lum_cam = Camera("Camera", int_lum, ext_lum, dist_lum, pos_lum, rot_lum)

        print(n2a_cam)
        print(lum_cam)

        datamanager.save(n2a_cam, "n2a_cam")
        datamanager.save(lum_cam, "cam2")
    else : 
        n2a_cam = datamanager.load("n2a_cam")
        lum_cam = datamanager.load("cam2")

    if SHOW_LOCALIZATION:
        display.title_print("Showing Localization Results")
        # Plotting the 3D visualization
        fig, ax = plot.get_3d_plot()

        #plot.reference_frame(ax, coords.WORLD_CENTER, coords.WORLD_ROTATION, label='World reference frame', lcolor='cyan')
        plot.bowling_lane(ax, coords.WORLD_LANE_CORNERS)

        plot.camera(ax, n2a_cam, label="N2A camera")
        plot.camera(ax, lum_cam, label="Lumix camera")
        plot.set_limits(ax, -10, 20)
        plot.view_angle(ax, elev=45, azim=15)
        # Show the plot
        plot.show()
  
           

