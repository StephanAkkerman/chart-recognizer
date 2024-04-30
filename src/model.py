import datetime
import logging

import matplotlib
import timm
import torch
from fastai.vision.all import *

# Local imports
from data import get_dls_from_images, get_dls_from_dataset


# https://timm.fast.ai/
def train(
    model_name: str = "efficientnet_b0",
    num_epochs: int = 5,
    batch_size: int = 32,  # still needs to be optimized
    lr: float = 1e-3,
    monitor: str = "f1_score",
    load_dataset_in_mem: bool = False,
    save_dir: str = "output",
):
    # Models must be in timm.list_models(pretrained=True)
    # https://huggingface.co/docs/timm/en/results for model results
    logging.info("Getting dataloader...")

    # This should only be done if you have a large amount of RAM (64GB+)
    # 48GB is enough for 10k images
    if load_dataset_in_mem:
        dls = get_dls_from_dataset(batch_size=batch_size)
    else:
        dls = get_dls_from_images(batch_size=batch_size)
    # Save dls?
    # torch.save(dls, "dls.pkl")
    # dls = toch.load("dls.pkl")
    logging.info("Dataloader loaded.")

    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    learn = Learner(
        dls,
        model,
        metrics=[accuracy, F1Score(), Precision(), Recall()],
        loss_func=CrossEntropyLossFlat(),
    )

    # Find an appropriate learning rate
    # learn.lr_find()
    suggested_lr = learn.lr_find(suggest_funcs=(valley, slide))[0]

    # Train the model
    logging.info("Starting training...")
    learn.fine_tune(
        num_epochs,
        base_lr=suggested_lr,
        cbs=[
            SaveModelCallback(monitor=monitor),
            # EarlyStoppingCallback(
            #    monitor=monitor, min_delta=0.1, patience=3
            # ),
        ],
    )

    logging.info("Training completed.")

    # TODO: add logic to only upload better models
    upload(model)

    # TODO: improve model_name to include timestamp
    # Is this the same as learn.save()?
    torch.save(model.state_dict(), f"{save_dir}/{model_name}.pth")

    # DISPLAY CLASSIFICATION RESULT
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    matplotlib.pyplot.show()

    # cleaner = ImageClassifierCleaner(learn)


def load_saved(model_name: str, save_dir: str = "output"):
    """
    Loads a saved model from disk.

    Parameters
    ----------
    model_name : str
        The name of the model to load, excluding `.pth`.

    Returns
    -------
    Timm model
        The loaded model.
    """
    model = timm.create_model(
        model_name=model_name,
        checkpoint_path=f"{save_dir}/{model_name}.pth",
        num_classes=2,
    )
    model.eval()
    return model


def upload(model, model_name: str = "chart-recognizer") -> None:
    """
    Pushes the timm model to the 🤗 Hub.

    Parameters
    ----------
    model : timm model
        The finetuned model to upload.
    """
    timm.models.push_to_hf_hub(
        model, model_name, model_config={"label_names": ["chart", "non-chart"]}
    )
