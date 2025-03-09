import os


# renames all the files in a folder having a certain extension
def bulk_rename(folder_path, file_name, extension=".jpg"):
    # Get a list of files in the folder matching the given extension
    images = [
        f
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in extension
    ]

    # Rename images sequentially according to given file_name
    for index, filename in enumerate(images, start=1):
        new_name = f"{file_name}_{index}{extension}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")


if __name__ == "__main__":
    folder_path = ""
    file_name = ""
    bulk_rename(folder_path, file_name)
