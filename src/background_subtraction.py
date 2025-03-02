import cv2 as cv
import numpy as np
import data.coords as coords

def create_polygonal_mask(corners, padding, size):
    # Create a mask based on the ROI specified before
    polygon = np.array(corners, dtype=np.int32)

    # Create a black mask
    poly_mask = np.zeros(size[:2], dtype=np.uint8)
    cv.fillPoly(poly_mask, [polygon], 255)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (padding * 1, padding * 5))

    # Dilate the mask to expand the region
    poly_mask = cv.dilate(poly_mask, kernel, iterations=1)

    if size[2] == 3: # makes the mask 3 channel if the input size was a 3 channel
        poly_mask = cv.cvtColor(poly_mask, cv.COLOR_GRAY2BGR)
    return poly_mask

def process_video(background_path, video_path, output_path, roi_points, 
                  threshold=15, kernel_size=10, tracking_tolerance=100,
                  ball_color_range=[(0,0,0),(255,255,255)],start_second=0, end_second=None, 
                  stabilization=False, playback_scaling=0.5):
    interactive = False
    # Load the background image 
    background = cv.imread(background_path)
    if background is None:
        print("Could not load background image!")
        return

    # Open the video
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        print("Could not open the video!")
        return

    # Get video properties
    frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv.CAP_PROP_FPS))
    ms_per_frame = int(100/fps)

    # Define codec and create VideoWriter
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Get the polygonal bounding box and apply it to the background image
    poly_mask = create_polygonal_mask(roi_points, 20, background.shape)
    background = cv.bitwise_and(background, poly_mask)

    if stabilization:
        # Create Shi-Tomasi corner detector parameters for feature detection
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Initialize the previous frame and points for optical flow
        ret, prev_frame = video.read()
        if not ret:
            print("Could not read the first frame.")
            return
        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

        # Detect features in the first frame
        prev_pts = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Initialize video playback
    frame_counter = start_second*fps
    video.set(cv.CAP_PROP_POS_FRAMES, int(start_second*fps-1))
    pos = None
    missed = 0
    position_detected=False

    # Process each frame
    while True:
        to_plot = []
        tracking_radius = tracking_tolerance * (missed+1)^3
        ret, frame = video.read()
        frame_counter += 1
        frame = cv.bitwise_and(frame, poly_mask)

        if end_second!=None and frame_counter > end_second*fps:
            break

        to_plot.append(frame)

        # Check if a frame was found
        if not ret:
            break

        if stabilization: 
            # Convert frame to grayscale
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Calculate optical flow to track features
            next_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None)

            # Select good points based on status
            good_new = next_pts[status == 1]
            good_old = prev_pts[status == 1]

            # Estimate the motion using the good points
            if len(good_new)>0 and stabilization:
                # Calculate the transformation matrix to align the frames
                transformation_matrix, _ = cv.estimateAffinePartial2D(good_old, good_new)

                # Warp the frame to stabilize it
                frame = cv.warpAffine(frame, transformation_matrix, (frame_width, frame_height))
                
            # Update previous frame and points
            prev_gray = frame_gray.copy()
            prev_pts = good_new.reshape(-1, 1, 2)

        # Apply Gaussian blur to smooth the image
        frame = cv.GaussianBlur(frame, (5, 5), 0)

        # Compute absolute difference between stabilized frame and background
        difference = cv.absdiff(frame, background)

        # Convert to grayscale
        difference_gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)

        # Threshold the difference
        _, diff_thresholded = cv.threshold(difference_gray, threshold, 255, cv.THRESH_BINARY)

        # Compute morphological mask
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        morph_mask = cv.morphologyEx(diff_thresholded, cv.MORPH_OPEN, kernel)
        morph_mask = cv.cvtColor(morph_mask, cv.COLOR_GRAY2BGR)

        # Apply mask to the frame
        frame_morph = cv.bitwise_and(frame, morph_mask)

        # Compute color mask
        hsv = cv.cvtColor(frame_morph, cv.COLOR_BGR2HSV)
        color_mask = cv.inRange(hsv, ball_color_range[0], ball_color_range[1])
        color_mask = np.array(cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel))

        # Compute the average position
        indices = np.argwhere(color_mask == 255)
        # Considering only the indices that are close to the previously computer point
        if len(indices) != 0 and not np.any(pos==None):
            indices = indices[np.linalg.norm(indices-pos, axis=1)<=tracking_radius]

        # Computing the new point
        if len(indices) != 0:
            pos = np.astype(indices.mean(axis=0), np.int16)
            pos_text = f"{pos[0]}, {pos[1]}"
            position_detected = True
            missed = 0
        else :
            missed+=1

        # Compute resulting video frame
        color_mask = cv.cvtColor(color_mask, cv.COLOR_GRAY2BGR)
        result = cv.bitwise_and(frame_morph, color_mask)

        if position_detected:
            print_pos = [pos[1], pos[0]]
            result = cv.putText(result, pos_text, print_pos, cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=1, lineType=cv.LINE_AA)
            cv.circle(result, print_pos, 5, (0,255,0),-1)
            cv.circle(result, print_pos, tracking_radius, (0,255,0), 2)

        to_plot.append(result)

        # Write the processed frame to output video
        out.write(result)

        # resizing
        for i, img in enumerate(to_plot):
            to_plot[i] = cv.resize(img, dsize=(0,0), fx=playback_scaling, fy=playback_scaling)        

        frame = np.hstack((to_plot[0],to_plot[1]))
        frame = cv.putText(frame, str(frame_counter), (30,30), cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2, lineType=cv.LINE_AA)

        
        cv.imshow('Before and After', frame)

        if interactive:
            key = cv.waitKey(0) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == 77:  # Right Arrow
                cv.waitKey(ms_per_frame)
            elif key == 75:  # Left Arrow
                cv.waitKey(ms_per_frame)
                frame_counter = max(start_second * fps, frame_counter - 10)
                video.set(cv.CAP_PROP_POS_FRAMES, frame_counter)
            elif key == ord('s'):  # Move backward
                cv.waitKey(ms_per_frame)
                frame_counter = max(start_second * fps, frame_counter - 2)
                video.set(cv.CAP_PROP_POS_FRAMES, frame_counter)
            elif key == ord(' '): # Space key to remove interactive mode
                interactive = False
        else: 
            key = cv.waitKey(ms_per_frame) & 0xFF
            if key == ord(' '): 
                interactive = True

    # Release resources
    video.release()
    out.release()
    cv.destroyAllWindows()
    print("Processing complete. Video saved at:", output_path)


if __name__ == "__main__":
    camera = 1
    video_name = 'opt_7.MP4'
    cameras = ['nothing_2a', 'lumix']
    coordinates = [coords.N2A_LANE_CORNERS, coords.LUM_LANE_CORNERS]
    sel_cam = cameras[camera]
    background_path = f"resources/localization/{sel_cam}/lane.png"
    video_path = f"resources/video/{sel_cam}/{video_name}"
    roi = coordinates[camera]
    output_path = f"resources/video_modified/{sel_cam}/{video_name}_mod.mp4"

    if camera == 0: # nothing
        hue_range = 40
        sat_range = 100
        value_range = 100
        hue = 20
        sat = 78.2*2.55
        value = 70*2.55
        color_range = [(hue-hue_range, sat-sat_range, value-value_range),
                        (hue+hue_range, sat+sat_range, value+value_range)]
        process_video(background_path, video_path, output_path, roi,threshold=20, kernel_size=5, tracking_tolerance=60,
                      ball_color_range=color_range, start_second=1, end_second=6, stabilization=False, playback_scaling=0.6) 
    else : # lumix
        hue_range = 12
        sat_range = 100
        value_range = 100
        hue = 12
        sat = 94.5*2.55
        value = 63.9*2.55
        color_range = [(hue-hue_range, sat-sat_range, value-value_range),
                        (hue+hue_range, sat+sat_range, value+value_range)]
        process_video(background_path, video_path, output_path, roi,threshold=15, kernel_size=8, tracking_tolerance=50,
                      ball_color_range=color_range, start_second=5, end_second=9, stabilization=False, playback_scaling=0.6)