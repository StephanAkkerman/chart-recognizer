import io
import os

import difPy
from datasets import Dataset, concatenate_datasets, load_dataset
from concurrent.futures import ThreadPoolExecutor
from fastai.vision.all import *
from PIL import Image
from tqdm import tqdm


def get_hf_dataset(
    dataset_name: str, split: str = "train", cache_dir: str = "downloads"
) -> Dataset:
    return load_dataset(
        f"StephanAkkerman/{dataset_name}", split=split, cache_dir=cache_dir
    )


def save_image(image_data, file_path):
    """Save an image to the specified file path, maintaining the original format."""
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, Image.Image):
        image = image_data
    else:
        image = Image.open(image_data)

    if image.format:
        format = image.format
    else:
        format = "PNG"  # Default format if no format is detected

    # Adjust file_path extension based on the format
    file_path = f"{os.path.splitext(file_path)[0]}.{format.lower()}"
    image.save(file_path)


def save_all_images(dataset, output_dir):
    """Save all images from the dataset to the specified directory, organized by label."""
    labels = dataset.features["label"].names
    label_dirs = {label: os.path.join(output_dir, label) for label in labels}
    for label_dir in label_dirs.values():
        os.makedirs(label_dir, exist_ok=True)

    def process_item(index_item):
        index, item = index_item  # Unpack the tuple
        label = labels[item["label"]]
        label_dir = label_dirs[label]
        file_name = item["id"]  # Temporarily omit the extension
        file_path = os.path.join(label_dir, file_name)
        save_image(item["image"], file_path)
        return file_path

    items_with_index = enumerate(dataset)  # Create an iterable of index, item pairs
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_item, items_with_index), total=len(dataset)))


def save_dataset_images():
    datasets = ["crypto-charts", "stock-charts", "fintwit-images"]

    for dataset_name in datasets:
        dataset = get_hf_dataset(dataset_name)
        output_dir = os.path.join(
            "downloaded-data", dataset_name
        )  # Directory named after the dataset
        save_all_images(dataset, output_dir)


def get_dls_from_images(batch_size: int = 32, img_size: int = 300):
    # Only do this if there is new data
    # TODO: skip images that are already downloaded
    # save_dataset_images()

    path = Path("downloaded-data")

    # DataBlock definition
    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # Input is an image, output is a category
        get_items=get_image_files,  # Fetch image paths dynamically
        splitter=RandomSplitter(valid_pct=0.3, seed=42),  # Creates the validation set
        get_y=parent_label,  # Uses folder names as labels
        item_tfms=Resize(img_size),
        batch_tfms=[
            *aug_transforms(size=img_size, min_scale=0.75),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    # Load the data into DataLoaders
    dls = datablock.dataloaders(path, bs=batch_size, num_workers=0)
    # dls.show_batch()
    # matplotlib.pyplot.show()
    return dls


def get_dls_from_dataset(batch_size: int = 32, img_size: int = 300):
    # Load dataset from Hugging Face
    dataset = concatenate_datasets(
        [
            get_hf_dataset("crypto-charts"),
            get_hf_dataset("stock-charts"),
            get_hf_dataset("fintwit-images"),
        ]
    )

    # DataBlock definition
    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # Input is an image, output is a category
        get_items=lambda x: x,  # Dummy function, dataset provides the items
        splitter=RandomSplitter(valid_pct=0.3, seed=42),  # Creates the validation set
        get_x=lambda x: x["image"],
        get_y=lambda x: x["label"],
        item_tfms=Resize(img_size),
        batch_tfms=[
            *aug_transforms(size=img_size, min_scale=0.75),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    # Load the data into DataLoaders
    # num_workers=0 is used to avoid a deadlock issue with DataLoader on Windows
    return datablock.dataloaders(dataset, bs=batch_size, num_workers=0)


def remove_duplicates():
    # dif = difPy.build(['data/', 'C:/Path/to/Folder_B/', 'C:/Path/to/Folder_C/', ... ])
    print("Searching for duplicates...")
    dif = difPy.build("data/crypto-charts/charts/")
    search = difPy.search(dif)


if __name__ == "__main__":
    remove_duplicates()
