import datetime
import logging
import json
import matplotlib
import timm
import torch
from fastai.vision.all import *

# Local imports
from data import get_dls_from_images, get_dls_from_dataset


# https://timm.fast.ai/
class ChartRecognizer:
    def __init__(self):
        # Read model args from config.json
        with open("config.json", "r") as config_file:
            self.config = json.load(config_file)

        model_name = self.config["model"]["timm_model_name"]

        # Change format of timm_models
        timm_models = [
            model.split(".")[0] for model in timm.list_models(pretrained=True)
        ]
        if model_name not in timm_models:
            raise ValueError(
                f"Model {model_name} not found in timm.list_models(pretrained=True)"
            )

        self.model = timm.create_model(model_name, pretrained=True, num_classes=2)

        # For converting config.json to function
        self.metrics_dict = {
            "f1": F1Score(),
            "precision": Precision(),
            "recall": Recall(),
            "accuracy": accuracy,
        }

    def train(self):
        # Models must be in timm.list_models(pretrained=True)
        # https://huggingface.co/docs/timm/en/results for model results
        logging.info("Getting dataloader...")

        # This should only be done if you have a large amount of RAM (64GB+)
        # 48GB is enough for 10k images
        if self.config["data"]["load_datasets_in_memory"]:
            train_dl, val_dl, test_dl = get_dls_from_dataset(config=self.config)
        else:
            train_dl, val_dl, test_dl = get_dls_from_images(config=self.config)
        logging.info("Dataloader loaded.")

        dls = DataLoaders(train_dl, val_dl)

        metrics = [
            self.metrics_dict[metric] for metric in self.config["model"]["metrics"]
        ]

        learn = Learner(
            dls,
            self.model,
            metrics=metrics,
            # Default loss is CrossEntropyLossFlat
        )

        # Find an appropriate learning rate
        # learn.lr_find()
        suggested_lr = learn.lr_find(suggest_funcs=(valley, slide))[0]

        # Train the model
        logging.info("Starting training...")
        learn.fine_tune(
            self.config["model"]["epochs"],
            base_lr=suggested_lr,
            cbs=[
                SaveModelCallback(
                    monitor=self.metrics_dict[self.config["model"]["monitor"]]
                ),
                # EarlyStoppingCallback(
                #    monitor=monitor, min_delta=0.1, patience=3
                # ),
            ],
        )

        logging.info("Training completed.")

        # Evaluate on test set
        logging.info("Evaluating on test set...")
        test_results = learn.validate(dl=test_dl)
        logging.info(
            f"Test Results - Loss: {test_results[0]}, Metrics: {test_results[1:]}"
        )

        return

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
