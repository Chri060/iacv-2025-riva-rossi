import os

folder_path = "resources/calibration/nothing_2a_checkerboards_2"
name = "n2a_cal2"

def rename_images():
    # Get a list of image files in the folder (common image extensions)
    image_extensions = ".jpg"
    images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

    # Rename images sequentially
    for index, filename in enumerate(images, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{name}_{index}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    rename_images()