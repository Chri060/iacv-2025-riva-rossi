import calibration
import utils.display as display

# Path where the checkerboard images are stored
n2a_checkerboard_path = "resources\\calibration\\nothing_2a_checkerboards" # Nothing Phone 2a
cam_checkerboard_path = "resources\\calibration\\cam_checkerboards" # Other camera


if __name__ == "__main__":
    mtx_n2a, dist_n2a = calibration.intrinsic(images_path=n2a_checkerboard_path)
    display.mat(mtx_n2a, "N2A Intrinsic calibration matrix :")
    display.mat(dist_n2a, "N2A Distortion matrix :")
    mtx_cam, dist_cam = calibration.intrinsic(images_path=cam_checkerboard_path)


