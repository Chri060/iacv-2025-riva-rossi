import cv2
import numpy as np

def track_ball(video_path, output_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Create the CSRT tracker
    tracker = cv2.TrackerCSRT_create()

    # Initialize variable to check if the ball has been tracked
    initialized = False
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(1*fps-1))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold the image to isolate the ball (assuming the ball is brighter)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (which should be the ball)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box for the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Initialize the tracker with the first frame and bounding box
            if not initialized:
                tracker.init(frame, (x, y, w, h))
                initialized = True
            else:
                # Update the tracker
                success, bbox = tracker.update(frame)
                if success:
                    # Get updated bounding box
                    x, y, w, h = [int(v) for v in bbox]

                    # Calculate the center of the bounding box (center of the ball)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    print(f"[{center_x}, {center_y}],")

                    # Draw the bounding box and center point on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot for center

                else:
                    # If tracking fails, print a message
                    cv2.putText(frame, "Tracking failed", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the frame with the tracked object to the output video
        out.write(frame)

        # Display the frame with tracking
        cv2.imshow('Tracking', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Tracking complete. Video saved at:", output_path)

if __name__ == "__main__":
    video_path = "resources/video_modified/mod.mp4"
    output_path = "resources/video_modified/tracked_ball.mp4"
    track_ball(video_path, output_path)
