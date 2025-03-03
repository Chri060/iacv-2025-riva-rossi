import cv2 as cv
import os
import numpy as np
import scipy.signal
import librosa
import subprocess

import src.utils.video_handler as video_prop

def sync_audio(video1_path, video2_path):
    # Extract the audio from the videos
    audio1, sr1 = librosa.load(video1_path, sr=None, mono=True)
    audio2, sr2 = librosa.load(video2_path, sr=None, mono=True)

    # Check if the audio sampling rates matches
    if sr1 != sr2:
        raise ValueError("Sampling rates do not match")

    # Compute the cross-correlation between the two audios
    corr = scipy.signal.correlate(audio1, audio2, mode='full')

    # Compute the lag between the two signals
    lag = np.argmax(corr) - (len(audio2) - 1)

    # Convert the lag in time difference
    time_shift = lag / sr1
    return max(time_shift, 0)

def process_video(input_path, output_path, target_fps, target_duration, start_time):
    # Open an output video handler with the correct parameters
    cap = cv.VideoCapture(input_path)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    width, height = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(output_path, fourcc, target_fps, (width, height))

    # Modify the video to have a fixed length and fps
    frame_time = 1 / target_fps
    current_time = start_time
    frame_count = int(target_fps * target_duration)
    count = 0

    while cap.isOpened() and count < frame_count:
        cap.set(cv.CAP_PROP_POS_MSEC, current_time * 1000)  #
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1
        current_time += frame_time

    cap.release()
    out.release()

def fix_fps_metadata(video_path, output_path, fps):
    subprocess.run(["ffmpeg", "-i", video_path, "-r", str(fps), "-y", output_path], check=True)

def synchronize_videos(video1_path, video2_path, output1_path, output2_path):
    # Get the properties of the two videos (fps, duration, and resolution)
    fps1, duration1, res1 = video_prop.get_video_properties(video1_path)
    fps2, duration2, res2 = video_prop.get_video_properties(video2_path)

    # Select the target fps and duration as the lowest value between the two video
    target_fps = min(fps1, fps2)
    target_duration = min(duration1, duration2)

    # Sync the audio of the videos
    start_time = sync_audio(video1_path, video2_path)

    # Create a temporary output
    temp_output1 = output1_path.replace(".MP4", "_temp.MP4")
    temp_output2 = output2_path.replace(".MP4", "_temp.MP4")

    # Process the videos to make them synchronized
    process_video(video1_path, temp_output1, target_fps, target_duration, start_time)
    process_video(video2_path, temp_output2, target_fps, target_duration, 0)

    # Fix the metadata of the output videos (the new fps value)
    fix_fps_metadata(temp_output1, output1_path, target_fps)
    fix_fps_metadata(temp_output2, output2_path, target_fps)

    # Remove the temporary output videos
    os.remove(temp_output1)
    os.remove(temp_output2)

if __name__ == "__main__":
    camera = ['lumix', 'nothing_2a']

    # Input folders for the un-synced videos
    video1 = f'resources/video/{camera[0]}/opt_7.MP4'
    video2 = f'resources/video/{camera[1]}/opt_7.MP4'

    # Output folders for the synced videos
    output1 = f'resources/video_sync/{camera[0]}/opt_7.MP4'
    output2 = f'resources/video_sync/{camera[1]}/opt_7.MP4'

    # Synchronize the videos
    synchronize_videos(video1, video2, output1, output2)
