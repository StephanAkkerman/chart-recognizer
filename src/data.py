import io
import os

from datasets import Dataset, concatenate_datasets, load_dataset
from concurrent.futures import ThreadPoolExecutor
from fastai.vision.all import (
    Path,
    DataBlock,
    ImageBlock,
    CategoryBlock,
    Resize,
    aug_transforms,
    Normalize,
    get_image_files,
    parent_label,
    imagenet_stats,
)
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from fastcore.foundation import L, range_of


def get_hf_dataset(
    dataset_name: str, split: str = "train", cache_dir: str = "downloads"
) -> Dataset:
    return load_dataset(
        f"StephanAkkerman/{dataset_name}", split=split, cache_dir=cache_dir
    )


def save_image(image_data, file_path, default_format="PNG"):
    """Save an image to the specified file path, maintaining the original format."""
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, Image.Image):
        image = image_data
    else:
        image = Image.open(image_data)

    format = image.format or default_format  # Use the image's format or default to PNG
    file_path = f"{os.path.splitext(file_path)[0]}.{format.lower()}"
    image.save(file_path, format=format)


def save_all_images(dataset, output_dir):
    """Save all images from the dataset to the specified directory, organized by label."""
    labels = dataset.features["label"].names
    label_dirs = {label: os.path.join(output_dir, label) for label in labels}

    # Create label directories and track existing files by base name
    existing_files = {}
    for label, label_dir in label_dirs.items():
        os.makedirs(label_dir, exist_ok=True)
        existing_files[label] = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}

    def process_item(item):
        label = labels[item["label"]]
        base_name = item["id"]  # Assume 'id' is the base name of the file
        file_path = os.path.join(label_dirs[label], base_name)

        # Check if the file already exists; if not, save it
        if base_name not in existing_files[label]:
            save_image(image_data=item["image"], file_path=file_path)
        else:
            existing_files[label].remove(
                base_name
            )  # Mark file as still existing in dataset

    # Process all items in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_item, dataset), total=len(dataset)))

    # Remove files that no longer exist in the dataset
    for label, files in existing_files.items():
        for base_name in files:
            for ext in [
                ".jpg",
                ".jpeg",
                ".png",
            ]:  # Consider common image file extensions
                file_path = os.path.join(label_dirs[label], f"{base_name}{ext}")
                if os.path.exists(file_path):
                    os.remove(file_path)


def save_dataset_images(
    datasets: list = ["crypto-charts", "stock-charts", "fintwit-images"],
    image_dir: str = "downloaded-images",
):
    for dataset_name in datasets:
        dataset = get_hf_dataset(dataset_name)
        output_dir = os.path.join(
            image_dir, dataset_name
        )  # Directory named after the dataset
        save_all_images(dataset, output_dir)


def TrainValTestSplitter(
    test_size=0.1, valid_size=0.2, random_state=None, stratify=None, shuffle=True
):
    "Create function that splits `items` into random train, validation, and test subsets."

    def _inner(o):
        # First, split into training + validation and test
        train_val, test = train_test_split(
            range_of(o),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify if stratify is not None else None,
            shuffle=shuffle,
        )

        # Split train_val into actual training and validation sets
        if stratify is not None:
            stratify_train_val = stratify[train_val]
        else:
            stratify_train_val = None

        train, valid = train_test_split(
            train_val,
            test_size=valid_size,
            random_state=random_state,
            stratify=stratify_train_val,
            shuffle=shuffle,
        )
        return L(train), L(valid), L(test)

    return _inner


def get_dls_from_images(config: dict):
    # Only do this if there is new data
    # TODO: skip images that are already downloaded
    save_dataset_images(
        datasets=config["data"]["datasets"], image_dir=config["data"]["image_dir"]
    )

    path = Path(config["data"]["image_dir"])
    img_size = config["data"]["transformations"]["img_size"]

    # Create the custom splitter
    splitter = TrainValTestSplitter(
        test_size=config["data"]["test_split"],
        valid_size=config["data"]["val_split"],
        random_state=42,
    )

    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # Input is an image, output is a category
        get_items=get_image_files,  # Fetch image paths dynamically
        splitter=splitter,  # Creates the validation set
        get_y=parent_label,  # Uses folder names as labels
        item_tfms=Resize(img_size),
        batch_tfms=[
            *aug_transforms(
                size=img_size, min_scale=config["data"]["transformations"]["min_scale"]
            ),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    # Load the data into DataLoaders
    dls = datablock.dataloaders(path, bs=config["model"]["batch_size"], num_workers=0)
    # dls.show_batch()
    # matplotlib.pyplot.show()
    return dls.loaders[0], dls.loaders[1], dls.loaders[2]


def get_dls_from_dataset(config: dict):
    # Load dataset from Hugging Face
    dataset = concatenate_datasets(
        [
            get_hf_dataset("crypto-charts"),
            get_hf_dataset("stock-charts"),
            get_hf_dataset("fintwit-images"),
        ]
    )

    # Create the custom splitter
    splitter = TrainValTestSplitter(
        test_size=config["data"]["test_split"],
        valid_size=config["data"]["val_split"],
        random_state=42,
    )
    img_size = config["data"]["transformations"]["img_size"]

    # DataBlock definition
    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # Input is an image, output is a category
        get_items=lambda x: x,  # Dummy function, dataset provides the items
        splitter=splitter,  # Creates the validation set
        get_x=lambda x: x["image"],
        get_y=lambda x: x["label"],
        item_tfms=Resize(img_size),
        batch_tfms=[
            *aug_transforms(
                size=img_size, min_scale=config["data"]["transformations"]["min_scale"]
            ),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    # num_workers=0 is used to avoid a deadlock issue with DataLoader on Windows
    dls = datablock.dataloaders(
        dataset, bs=config["model"]["batch_size"], num_workers=0
    )
    return dls.loaders[0], dls.loaders[1], dls.loaders[2]
