# Program execution

Before running the application, you need to install the required dependencies and system packages.

### Dependencies

First, install all Python dependencies from the requirements.txt file:

```bash
   pip install -r requirements.txt
```

FFmpeg is required for video processing and audio synchronization.
Install it based on your operating system:

```bash
   # Linux
   sudo apt update 
   sudo apt install ffmpeg
   
   # MacOS
   brew install ffmpeg
   
   # Windows
   choco install ffmpeg
```

OpenH264 is required for video encoding/decoding.
Install it based on your operating system:

```bash
   # Linux
   wget https://github.com/cisco/openh264/releases/download/v2.3.1/libopenh264-2.3.1-linux64.7z
   7z x libopenh264-2.3.1-linux64.7z
   sudo cp libopenh264-2.3.1-linux64/libopenh264.so.2 /usr/lib/
   
   # MacOS
   brew install openh264
   
   # Windows: download from the website
```

### Run the program

To start the bowling ball tracking pipeline, run the main Python script with the path to your configuration file as an
argument:

```bash
   python3 main.py ./config/dev.yml
```

## Configuration file

The configuration file is structured as follows:

- save_name: suffix given to the file saved
- global: global variables common to all parts of the pipelines
- pipeline: local variables for each part of the pipeline

Each part is explained in detail in the following part.

### Save name

```yml
save_name: default
```

The pipeline save name.
All components from the pipeline package have the ability to execute their code and store its results, or to load them
by skipping execution.
This name will be used to retrieve past computations that used the same savename.

### Global

``` yml
global:
  video_name: "video.mp4"
  camera_names: [ "camera1", "camera2" ]
  paths:
    originals: "resources/videos/originals"
  coords:
    world_lane: [ [ 0, 0, 0 ], [ 19, 0, 0 ], [ 19, -1.07, 0 ], [ 0, -1.07, 0 ] ]
  ball_radius: 0.1091
  visualization: True
```

Some necessary global configuration options specifying the videos to analyze, camera names, paths and known coordinates:

- video_name: the name of the original stereo videos we are referring to
- camera_names: the names of the cameras we are using (for visualization and organization purposes)
- paths: the global paths that we need in our code
    - originals: ath to original videos
- coords: set of known configurable coordinates
    - world_lane: the lane corners expressed in the world reference system ((0,0,0) in the bottom left corner, X towards
      the pins, Y away from the lane, Z towards the sky)
- ball_radius: the real bowling ball radius
- visualization: enables visualization for the entire pipeline

### Pipeline

```yml
pipeline:
  - name: module_name
    type: load | execute
    params: { ... }
```

The pipeline represents the ordered list of operations that are going to be executed on our stereo videos provided in
the global configuration.
Each element of the pipeline is called a Pipe and provides different functionalities.
Each pipe has a `name` referring to its name and a `type` that can be either `execute` or `load` depending on the wanted
configuration.

The pipeline consists of the following operations:

1. _Intrinsic calibration_: the camera's intrinsic parameters are determined using a checkerboard pattern, enabling
   accurate mapping between the 2D image plane and 3D space
    ```yml
   - name: intrinsic
     type: _
     params: {
       images_path: "resources/calibration"           # Calibration images path
     }
     ```
2. _Video synchronization_: two video sources are synchronized using audio correlation techniques to ensure temporal
   alignment across frames
    ```yml
    - name: video_synchronization
      type: _
      params: {
        save_path: "resources/videos/sync"            # Synchronized video save folder
      }
     ```
3. _Video undistortion_: lens distortions are corrected using the previously computed intrinsic parameters, providing a
   geometrically accurate view of the lane
    ```yml
       - name: video_undistortion
         type: _
         params: {
           save_path: "resources/videos/undistorted"  # Undistorted video save folder
         }
    ```
4. _Lane detection_: the lane edges are manually identified to establish a reference frame for the ball localization
   ```yml
      - name: lane_detection
        type: _
        params: { 
            save_path: "resources/images/lane_detection"    # Lane detected image save path
        }
    ```
5. _Extrinsic calibration_: the camera’s position and orientation relative to the lane are computed from the lane
   geometry and intrinsic calibration, allowing the projection of 2D detections into 3D space
   ```yml
      - name: extrinsic
        type: _
        params: { 
            save_path: "resources/graphs/extrinsic"         # Cameras position graphs save path
        }
    ```
6. _Ball tracking_: the ball is detected in each frame using YOLO, and missing detections are estimated through temporal
   extrapolation
   ```yml
      - name: ball_tracker
        type: _
        params: { 
            save_path: "resources/images/tracking"         # Ball detected image save path
        }
    ```
7. _Ball localization_: detected 2D positions are transformed into 3D coordinates using the extrinsic calibration
   ```yml
      - name: ball_localization
        type: _
        params: { 
            save_path: "resources/graphs/trajectory"         # 3D ball position save path
        }
    ```
8. _Ball rotation estimation_: optical flow and frame-to-frame analysis are applied to estimate the ball’s spin and the
   orientation of its rotation axis
   ```yml
      - name: ball_rotation
        type: _
        params: { 
            save_path: "resources/graphs/"         # Ball angular speed and axis graphs save path
        }
    ```