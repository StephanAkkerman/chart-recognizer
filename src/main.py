import timm
import torch
from fastai.vision.all import *
from datasets import load_dataset


def get_data(batch_size: int = 32):
    # Load dataset from Hugging Face
    dataset = load_dataset(
        "StephanAkkerman/fintwit-charts", split="train", cache_dir="downloads"
    )

    # DataBlock definition
    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=lambda x: x,  # Dummy function, dataset provides the items
        splitter=RandomSplitter(valid_pct=0.3, seed=42),
        get_x=lambda x: PILImage.create(x["image"]).convert(
            "RGB"
        ),  # some images are RGBA
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

    save(model, learn, model_name)


def save(model, learn, model_name: str):
    # Is this the same as learn.save()?
    torch.save(model.state_dict(), f"{model_name}.pth")

    # Save the model
    # learn.save("model_name")
    # learn.export("model_name.pkl")


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
