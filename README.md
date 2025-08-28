# Ball Trajectory And Motion Analysis  

<div align="center">
    <img src="https://img.shields.io/badge/Version-1.0-4CAF50?style=for-the-badge" alt="Project version">
    <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python"> 
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white&style=for-the-badge" alt="OpenCV">
    <img src="https://img.shields.io/badge/Ultralytics-FF6F00?logo=ultralytics&logoColor=white&style=for-the-badge" alt="Ultralytics">
</div>

This project focuses on the detection and analysis of a bowling ball in motion on a bowling lane. 
Its primary goals are to accurately track the ball's trajectory and estimate its spin and the orientation of its rotation axis over time.
The workflow consists of several key stages:
1. _Ball detection and localization_: the bowling ball is initially detected using YOLOv8-large, an advanced object detection neural network. 
This allows for real-time and robust identification of the ball even under varying lighting conditions and complex lane backgrounds.

2. _3D position reconstruction_: once detected, the ball’s position is reconstructed in 3D space using geometric transformations based on camera calibration parameters and lane measurements. This step ensures precise trajectory estimation along the lane.

3. _Spin and rotation analysis_: to study the ball’s rotational behavior, optical flow techniques are applied to consecutive frames. This enables the calculation of spin rate and rotation axis orientation

By combining real-time detection with accurate geometric and motion analysis, this project provides a comprehensive understanding of bowling ball dynamics, useful for training, coaching, and sports analytics.

## Installation

After downloading the project, install all Python dependencies from the requirements file:

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

# Windows
# Download from https://github.com/cisco/openh264/releases/tag/v2.6.0
```

    
## Usage

To start the bowling ball tracking pipeline, run the main Python script with the path to your configuration file as an
argument:

```bash
python3 main.py ./config/dev.yml
```

To start the bowling ball tracking dashboard, run the dashapp Python script with the path to your configuration file as
an
argument:

```bash
python3 dashapp.py ./config/dev.yml
```

## Authors

- [Christian Rossi](https://github.com/Chri060)
- [Filippo Riva](https://github.com/FilippoRiva)

