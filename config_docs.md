# Config Documentation
A detailed explaination of the various options for the config file

## savename

```yml
savename: default
```

The pipeline save name. 

All the components from the pipeline package have the ability to execute their code and store its results, or to load them by skipping execution. 
This name will be used to retrieve past computations that used the same savename. 

E.g. Assume we first run an X module in "execute" mode using the "my_savename" savename. We'll be able to load the previously computed results only by keeping the same savename in the config, and setting the X module to "load" instead of "execute". This is done to prevent save conflicts on different savenames.

## global
Some necessary global configuration options specifying the videos to analyze, camera names, paths and known coordinates.

``` yml
global:
  video_names: ["video1.MP4", "video2.MP4"]
  camera_names: ["cam1", "cam2"]
  paths:
    originals: "resources/videos/originals"
  coords:
    world_lane: [[0, 0, 0], [12, 0, 0], [12, -1.07, 0], [0, -1.07, 0]]
  visualization: True
```

- video_names : the names of the original stereo videos we are referring to 
- camera_names : the names of the cameras we are using (just for visualization purposes)
- paths : the global paths that we need in our code
  - originals : the path to original videos
- coords : a set of known configurable coordinates
  - world_lane : the lane corners expressed in the world reference system ((0,0,0) in the bottom left corner, X towards the pins, Y away from the lane, Z towards the sky)
- visualization : enables visualization for the entire pipeline, if False visualization works only if the specific pipe has visualization: True


## pipeline
```yml
pipeline:
  - name: module_name
    type: execute
    params: {my_param: foo}
```
The pipeline represents the ordered list of operations that are going to be executed on our stereo videos provided in the global configuration.

Each element of the pipeline is called a Pipe and provides different functionalities.
Each pipe has a `name` referring to its name and a `type` that can be either `execute` or `load` depending on the wanted configuration.

### Pipes
  >Important Note : Some modules have dependencies on previous ones. In that case the dependencies must be executed/loaded first.

  #### Intrinsic Calibration
  ```yml
  - name: intrinsic
    type: execute | load
    params: { # optional for load
      images_path : "resources/images/calibration", 
      visualization : False, # optional (default:False)
      checkerboard_sizes : [[9, 6], [9, 6]] # optional (default:[[9, 6], [9, 6]])
    }
  ```

  Performs intrinsic camera calibration on the environment cameras given the checkerboard calibration images in `images_path`. 
  - images_path : folder containing a folder for each camera name. Each camera folder will need at least 10 calibration images with a checkerboard
  - visualization : True makes the checkerboard detection visible (default:False)
  - checkerboard_sizers : The size of the images checkerboard for each camera, expressed as a list of arrays (default:[[9, 6], [9, 6]])

  #### Video Synchronization
  ```yml
  - name: video_synchronization
    type: execute | load
    params: {
      save_path : "resources/videos/sync",
      visualize: False # optinal (default:False)
    }
  ```

  Synchronizes the videos loaded in the environment based on audio correlation and saves the synchronized videos in the `save_path`
  - visualize : shows the parallel video processing

  #### Video Stabilization
  ```yml
  - name: video_stabilization
    type: execute | load
    params: {
      save_path : "resources/videos/stabilized",
      visualization : False # optional (default:False)
    }
  ```

  Stabilizes the videos loaded in the environment using optical flow and salient point detection algorithms. Saves the stabilized videos in `save_path`

  #### Video Undistortion
  > dependencies : [intrinsic](#intrinsic-calibration)
  ```yml
  - name: video_undistortion
    type: execute
    params: {
      save_path : "resources/videos/undistorted",
      visualization : False # optional (default:False)
    }
  ```

  Undistorts the videos loaded in the environment using the distortion parameters of the loaded camera. Saves the undistorted videos in `save_path`.

  #### Lane Detection
  ```yml
  - name: lane_detection
    type: execute | load
    params: { # optional
      scale: [0.5, 0.5], # optional (default:[0.7, 0.7])
      visualization: False # optional (default:False)
    }
  ```

  (Manually) Detects the bowling lanes in the stereo videos.
  - scale : image scale factor during manual point selection
  - visualization : visualizes the previously selected points

  #### Extrinsic Calibration
  > dependencies : [intrinsic](#intrinsic-calibration), [lane_detection](#lane-detection)
  ```yml
  - name: extrinsic
    type: execute | load
  ```

  Performs intrinsic calibration on the cameras in the environment given intrinsic parameters from [intrinsic](#intrinsic-calibration), the detected lane corners by [lane_detection](#lane-detection) and the known lane corner coordinates specified in the [global configuration](#global)

  #### Localization Visualizer
  > dependencies : [extrinsic](#extrinsic-calibration)
  ```yml
  - name: localization_viz
    type: execute
  ```
  Visualizes in a 3D plot how the cameras are located with respect to the bowling lane

  #### Object Tracker
  ```yml
  - name: object_tracker
    type: execute | load
  ```

  Tracks a moving object (the bowling ball)

  #### 3D localization
  > dependencies : [extrinsic](#extrinsic-calibration), [object_tracker](#object-tracker)
  ```yml
  - name: ball_3d_localization
    type: execute | load
  ```

  Localizes the ball in 3D

  #### 3D trajectory visualizer
  > dependencies : [3d_localization](#3d-localization)
  ```yml
  - name: trajectory_viz
    type: execute | load
  ```


  # Sample configuration
  ```yml
  savename: default
  global:
    video_names: ["opt_7.MP4", "opt_7.MP4"]
    camera_names: ["nothing", "lumix"]
    paths:
      originals: "resources/videos/originals"
    coords:
      world_lane: [[0, 0, 0], [12, 0, 0], [12, -1.07, 0], [0, -1.07, 0]]
  pipeline:
    - name: intrinsic
      type: execute
      params: {
        images_path : "resources/images/calibration",
        show_detection : False,
        checkerboard_sizes : [[9, 6], [9, 6]]
      }
    - name: video_synchronization
      type: execute
      params: {
        save_path : "resources/videos/sync"
      }
    - name: video_stabilization
      type: execute
      params: {
        save_path : "resources/videos/stabilized"
      }
    - name: video_undistortion
      type: execute
      params: {
        save_path : "resources/videos/undistorted"
      }
    - name: lane_detection
      type: execute
      params: {
        scale: [0.5, 0.5]
      }
    - name: extrinsic
      type: execute
    - name: localization_viz
      type: execute
    - name: object_tracker
      type: execute
    - name: 3d_localization
      type: execute
  ```



