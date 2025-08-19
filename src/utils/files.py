import os

def bulk_rename(folder_path, file_name, extension=".jpg"):
    """
    Renames all files in the specified folder that have the given extension.

    Parameters:
        folder_path (str): Path to the folder containing files.
        file_name (str): Base name to use for renamed files.
        extension (str): File extension to filter by (default is ".jpg").
    """

    # Get a list of all files in the folder that match the given extension
    images = [
        f
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in extension
    ]

    # Rename each file sequentially
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