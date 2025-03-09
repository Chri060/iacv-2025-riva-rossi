import cv2 as cv
import numpy as np
from scipy import interpolate
from pipeline.pipe import Pipe
from pipeline.environment import Environment
from pipeline.environment import DataManager

class LaneDetector(Pipe):
    def execute(self, params):
        scales = params.get('scale', [0.7, 0.7])
        detection_results = {}
        for i, view in enumerate(Environment.get_views()):
            _, duration, _ = view.video.get_video_properties()
            capture = view.video.capture
            # reads the frame in the middle of the video
            capture.set(cv.CAP_PROP_POS_MSEC, duration*1000/2)
            ret, frame = capture.read()
            points = self.__manual_point_selection(frame, scale=scales[i]) 
            capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            view.lane.corners = points
            detection_results.update({view.camera.name: points})
        DataManager.save(detection_results, self.save_name)

    
    def load(self, params):
        try: visualization = params['visualization']
        except: visualization = Environment.visualization

        detection_results = DataManager.load(self.save_name)
        for view in Environment.get_views():
            view.lane.corners = detection_results[view.camera.name]
            if visualization:
                capture = view.video.capture
                _, duration, _ = view.video.get_video_properties()
                # reads the frame in the middle of the video
                capture.set(cv.CAP_PROP_POS_MSEC, duration*1000/2)
                ret, frame = capture.read()
                capture.set(cv.CAP_PROP_POS_MSEC, 0)
                for point in view.lane.corners:
                    frame = cv.circle(frame, (int(point[0]), int(point[1])), 2, (255,0,0), 3)
                cv.imshow(f"Lane : {view.camera.name}", frame)
                cv.waitKey(0)
                cv.destroyAllWindows()
                

    def __manual_point_selection(self, image, scale=0.5):
        selected_points = []
        upscale = 1/scale

        def select_point(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                cv.circle(image, (x, y), 3, (0, 0, 255), -1)
                cv.imshow("Image", image)
                x = x*upscale
                y = y*upscale
                selected_points.append([x, y])

        image = cv.resize(image, None, fx=scale, fy=scale)
        cv.imshow("Image", image)
        cv.setMouseCallback("Image", select_point)

        cv.waitKey(0)
        cv.destroyAllWindows()

        return selected_points

class ObjectTracker(Pipe):
    def execute(self, params):
        try: save_path = params.get('save_path')
        except: raise Exception("Missing required parameter : save_path")

        for i, view in enumerate(Environment.get_views()):
            video = view.video
            roi = view.lane.corners
            output_path = f"{save_path}/{view.camera.name}/{Environment.savename}_{Environment.video_names[i]}"
            localized_points = self.track_moving_object(video, output_path, roi,threshold=20, kernel_size=5, tracking_tolerance=60,
                    playback_scaling=0.6) 


        ## Nothing2A parameters
        # hue_range = 40
        # sat_range = 100
        # value_range = 100
        # hue = 20
        # sat = 78.2*2.55
        # value = 70*2.55
        # color_range = [(hue-hue_range, sat-sat_range, value-value_range),
        #                 (hue+hue_range, sat+sat_range, value+value_range)]
        # threshold = 20
        # kernel_size = 5
        # tracking_tolerance = 60

        ## Lumix parameters
        # hue_range = 12
        # sat_range = 100
        # value_range = 100
        # hue = 12
        # sat = 94.5*2.55
        # value = 63.9*2.55
        # color_range = [(hue-hue_range, sat-sat_range, value-value_range),
        #                 (hue+hue_range, sat+sat_range, value+value_range)]
        # threshold = 15
        # kernel_size = 8
        # tracking_tolerance = 50

    
    def load(self, params):
        return super().load(params)
    
    def track_moving_object(self, video, output_path, roi_points, 
                    threshold=15, kernel_size=10, tracking_tolerance=100,
                    color_range=[(0,0,0),(255,255,255)],start_second=0, end_second=None, 
                    stabilization=False, playback_scaling=0.5):
        interactive = False

        # Open the video
        video = video.capture
        if not video.isOpened():
            print("Could not open the video!")
            return
        
        background = self.__estimate_background(video)

        # Get video properties
        frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv.CAP_PROP_FPS))
        ms_per_frame = int(100/fps)

        # Define codec and create VideoWriter
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Get the polygonal bounding box and apply it to the background image
        poly_mask = self.__create_polygonal_mask(roi_points, 20, background.shape)
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

        image_points = []

        # Process each frame
        while True:
            to_plot = []
            tracking_radius = tracking_tolerance * (missed+1)^3
            ret, frame = video.read()
            if not ret:
                break
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
            color_mask = cv.inRange(hsv, color_range[0], color_range[1])
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
            
            image_points.append(pos)

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
        return image_points

    # creates a binary polygonal mask of a given size
    def __create_polygonal_mask(self, corners, padding, size):
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

    # Estimates a background image by averaging n frames
    def __estimate_background(self, video_capture, n=50 ,show=False):
        print(f"Estimating Background for {video_capture}")
        ids = video_capture.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=n)
        frames = []
        for id in ids:
            video_capture.set(cv.CAP_PROP_POS_FRAMES, id)
            ret, frame = video_capture.read()
            frames.append(frame)
        
        background = np.astype(np.median(frames, axis=0), np.uint8)
        if show:
            background_toshow = cv.resize(background, dsize=(0,0), fx=0.5, fy=0.5)
            cv.imshow('background', background_toshow)
            cv.waitKey(0)
        return background

class Triangulator(Pipe):
    def execute(self, params):
        return None
    
    def load(self, params):
        return None

class Single_View_Trajectory_Estimator(Pipe):
    def execute(self, params):
        return None
    
    def load(self, params):
        return None

    # Estimates a trajectory in world reference given image to world point correspondences and trajectory_points
    def single_view_trajectory_estimation(self, trajectory_points, from_image_points, to_world_points):
        trajectory_points = np.array(list(filter(lambda i: i is not None, trajectory_points)), dtype=np.float32)
        # Compute the homography matrix between image and world
        H, _ = cv.findHomography(from_image_points, to_world_points[:, :2])

        # Reshape the 2D points for transformation
        image_points_reshaped = trajectory_points.reshape(-1, 1, 2)

        # Apply the homography
        world_points = cv.perspectiveTransform(image_points_reshaped, H)

        # Extract world coordinates
        world_points = world_points.reshape(-1, 2).astype(np.float64)

        # Fit a spline curve with a smoothing factor and no closure (open curve)
        smoothing_factor = 2
        tck, u = interpolate.splprep([world_points[:, 0], world_points[:, 1]], s=smoothing_factor, per=False)
        # Evaluate the spline at more points for a smoother curve
        unew = np.linspace(0, 1, 1000)
        out_points = interpolate.splev(unew, tck)
        return out_points

