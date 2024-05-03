import io
import os

from datasets import Dataset, concatenate_datasets, load_dataset
from concurrent.futures import ThreadPoolExecutor
from fastai.vision.all import *
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

    def process_item(item):
        label = labels[item["label"]]
        label_dir = label_dirs[label]
        file_name = item["id"]  # Temporarily omit the extension
        file_path = os.path.join(label_dir, file_name)
        save_image(item["image"], file_path)
        return file_path

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_item, dataset), total=len(dataset)))


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
    # save_dataset_images(datasets=config["data"]["datasets", image_dir=config["data"]["image_dir"])

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
