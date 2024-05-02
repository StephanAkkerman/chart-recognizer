import os
import shutil
from pathlib import Path

import difPy
import pandas as pd
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def format_data_directory(base_path: str = "data"):
    """
    Renames images in 'charts' and 'non-charts' subdirectories to numerical sequences.

    Args:
    - base_path (str): Path to the base directory containing 'charts' and 'non-charts'.
    """
    # Define the subdirectories to process
    subdirs = ["charts", "non-charts"]

    for subdir in subdirs:
        # Construct the full path to the subdirectory
        full_path = Path(base_path) / subdir

        # Ensure the subdirectory exists
        if not full_path.exists() or not full_path.is_dir():
            print(f"Directory {full_path} does not exist or is not a directory.")
            continue

        # Fetch all image files in the directory
        files = sorted(full_path.glob("*"))
        for idx, file in enumerate(files, start=1):
            # Construct the new file name with a numerical sequence
            new_file_name = f"{idx}{file.suffix}"
            new_file_path = full_path / new_file_name

            # Rename the file
            file.rename(new_file_path)
            print(f"Renamed {file} to {new_file_path}")


def organize(data_dir="data", test_size: float = 0.3, file_ext: str = ".png"):
    """
    Organizes the data in the data directory into train and val directories.
    Note: be sure to make a backup of your data directory before running this function.
    This function can be undone by running the unorganize function.

    Parameters
    ----------
    data_dir : str, optional
        The directory where the images are saved, by default "data"
    """
    # Define your source directories
    src_dirs = {"charts": f"{data_dir}/charts", "non-charts": f"{data_dir}/non-charts"}

    # Define your destination directories
    dest_dirs = {"train": f"{data_dir}/train", "val": f"{data_dir}/val"}

    # Ensure destination directories exist
    for cat in ["charts", "non-charts"]:
        os.makedirs(os.path.join(dest_dirs["train"], cat), exist_ok=True)
        os.makedirs(os.path.join(dest_dirs["val"], cat), exist_ok=True)

    for category, src_dir in tqdm(src_dirs.items()):
        # List all files in the source directory
        files = [
            f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))
        ]

        # Split files into train and val sets (70% train, 30% val)
        train_files, val_files = train_test_split(
            files, test_size=test_size, random_state=42
        )

        # Function to move and rename files
        def move_files(files, dest_subdir):
            for idx, file in enumerate(files):
                # Define new file name
                new_file_name = f"{category}_{idx}{file_ext}"
                src_file_path = os.path.join(src_dir, file)
                dest_file_path = os.path.join(
                    dest_dirs[dest_subdir], category, new_file_name
                )

                # Move and rename the file
                shutil.move(src_file_path, dest_file_path)

        # Move and rename train and val files
        move_files(train_files, "train")
        move_files(val_files, "val")


def unorganize(data_dir="data - Copy"):
    # Define your destination directories (now acting as source)
    src_dirs = {"train": f"{data_dir}/train", "val": f"{data_dir}/val"}

    # Define your original source directories (now acting as destination)
    dest_dirs = {"charts": f"{data_dir}/charts", "non-charts": f"{data_dir}/non-charts"}

    # Ensure original source directories exist
    for dest_dir in dest_dirs.values():
        os.makedirs(dest_dir, exist_ok=True)

    for src_subdir, src_dir_path in src_dirs.items():
        # Process each file in the train and val directories
        for category in dest_dirs.keys():
            full_src_dir = Path(src_dir_path) / category
            if not full_src_dir.exists():
                continue  # Skip if the category directory does not exist

            files = [
                f for f in os.listdir(full_src_dir) if os.path.isfile(full_src_dir / f)
            ]
            for file in tqdm(files, desc=f"Merging {category} from {src_subdir}"):
                src_file_path = full_src_dir / file
                dest_file_path = Path(dest_dirs[category]) / file

                # Rename the file if a file with the same name exists in the destination
                counter = 1
                while dest_file_path.exists():
                    # Append a counter to the file's name to make it unique
                    name, ext = os.path.splitext(file)
                    new_name = f"{name}_{counter}{ext}"
                    dest_file_path = Path(dest_dirs[category]) / new_name
                    counter += 1

                # Move the file to the destination directory with a unique name
                shutil.move(src_file_path, dest_file_path)


def to_RGB(main_subdir: str = "crypto-charts", data_dir: str = "data"):
    """
    Converts all images in specified subdirectories to RGB format.
    """
    # Define the main directory path
    main_dir = Path(data_dir) / main_subdir

    # Subdirectories to process
    subdirs = ["charts", "non-charts"]

    # Loop over each subdirectory
    for subdir in subdirs:
        full_path = main_dir / subdir

        # Ensure the subdirectory exists
        if not full_path.exists() or not full_path.is_dir():
            print(f"Directory {full_path} does not exist or is not a directory.")
            continue  # Skip to the next subdirectory if this one is invalid

        # Fetch all image files in the directory
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
        ]  # Add or remove extensions as needed
        files = sorted(
            f
            for f in full_path.glob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        )

        for file in tqdm(files, desc=f"Processing {subdir}"):
            # Open the image file
            with Image.open(file) as img:
                # Convert the image to RGB format if not already RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    # Save the image back to the file
                    img.save(file)


def upload(data_dir: str = "data", subdir: str = "crypto-charts") -> None:
    # https://huggingface.co/blog/image-search-datasets
    # Be sure to move folders out of the subdir that do not need to be uploaded

    # Load the image dataset
    dataset = load_dataset(f"{data_dir}/{subdir}")

    # Add id column
    dataset = dataset.map(
        lambda x: {"id": x["image"].filename.split("\\")[-1]}, num_proc=8
    )

    # Upload the dataset to the Hub
    dataset.push_to_hub(f"StephanAkkerman/{subdir}", commit_message="new data upload")


def create_metadata(data_dir: str = "data", subdir: str = "crypto-charts"):
    dir_path = os.path.join(data_dir, subdir)
    records = []

    # Traverse the subdirectories within the main directory
    for label_dir in ["charts", "non-charts"]:
        full_path = os.path.join(dir_path, label_dir)
        for filename in os.listdir(full_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                records.append(
                    {
                        "file_name": f"{label_dir}/{filename}",
                        "id": filename,  # ID is the same as filename
                        "label": label_dir,
                    }
                )

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(records)

    # Save DataFrame to CSV
    metadata_csv_path = os.path.join(dir_path, "metadata.csv")
    df.to_csv(metadata_csv_path, index=False)

    print(f"Metadata DataFrame written to {metadata_csv_path}")


def upload_all_datasets():
    for subdir in ["crypto-charts", "stock-charts", "fintwit-images"]:
        upload(subdir=subdir)


def remove_duplicates():
    for main_dir in ["crypto-charts", "stock-charts", "fintwit-images"]:
        for sub_dir in ["charts", "non-charts"]:
            dif = difPy.build(f"data/{main_dir}/{sub_dir}")
            search = difPy.search(dif)
            search.move_to(destination_path=f"data/duplicates/{main_dir}/{sub_dir}/")


if __name__ == "__main__":
    upload_all_datasets()
