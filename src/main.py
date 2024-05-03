import torch

import logging
from model import ChartRecognizer

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Make directories that do not yet exist

    # Display CUDA info
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for i, device in enumerate(devices):
        logging.info(f"CUDA Device {i}: {device}")

    # Train the model
    logging.info("Training the model")
    model = ChartRecognizer()
    model.train()
    logging.info("Model trained and saved to output/FinTwitBERT")
