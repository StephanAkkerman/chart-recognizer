import timm
import torch
from fastai.vision.all import *
from datasets import load_dataset, concatenate_datasets


def get_hf_dataset(
    dataset_name: str, split: str = "train", cache_dir: str = "downloads"
):
    return load_dataset(
        f"StephanAkkerman/{dataset_name}", split=split, cache_dir=cache_dir
    )


def get_data(batch_size: int = 32):
    # Load dataset from Hugging Face
    crypto_charts = get_hf_dataset("crypto-charts")
    stock_charts = get_hf_dataset("stock-charts")

    dataset = concatenate_datasets([crypto_charts, stock_charts])

    # DataBlock definition
    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=lambda x: x,  # Dummy function, dataset provides the items
        splitter=RandomSplitter(valid_pct=0.3, seed=42),
        get_x=lambda x: x["image"],
        get_y=lambda x: x["label"],
        item_tfms=Resize(300),
        batch_tfms=[
            *aug_transforms(size=300, min_scale=0.75),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    # Load the data into DataLoaders
    # num_workers=0 is used to avoid a deadlock issue with DataLoader on Windows
    return datablock.dataloaders(dataset, bs=batch_size, num_workers=0)


# https://timm.fast.ai/
def main(
    model_name: str = "efficientnet_b0",
    num_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    monitor: str = "f1_score",
):
    # Models must be in timm.list_models(pretrained=True)
    # https://huggingface.co/docs/timm/en/results for model results
    print("Loading data...")
    dls = get_data(batch_size=batch_size)
    print("Data loaded.")

    model = timm.create_model(model_name, pretrained=True, num_classes=dls.c)
    learn = Learner(dls, model, metrics=[accuracy, F1Score(), Precision(), Recall()])

    # Find an appropriate learning rate
    learn.lr_find()
    # suggested_lr = learn.lr_find(suggest_funcs=(valley, slide))[0]

    # Train the model
    learn.fine_tune(
        num_epochs,
        base_lr=lr,
        cbs=[
            SaveModelCallback(monitor=monitor),
            EarlyStoppingCallback(
                monitor=monitor, min_delta=0.1, patience=3
            ),  # maybe change to f1_score
        ],
    )

    # Is this the same as learn.save()?
    torch.save(model.state_dict(), f"{model_name}.pth")
    upload(model)


def load_saved(model_name: str):
    # Load the model
    model = timm.create_model(
        model_name=model_name, checkpoint_path=f"{model_name}.pth", num_classes=2
    )
    model.eval()
    return model


def upload(model):
    # Push it to the 🤗 Hub
    timm.models.push_to_hf_hub(
        model, "chart-recognizer", model_config={"label_names": ["chart", "non-chart"]}
    )


if __name__ == "__main__":
    main()
