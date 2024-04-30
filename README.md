# A Specialized Image Model For Financial Tweets

![chart-recognizer banner](img/banner.png)

---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Supported versions">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

chart-recognizer is an image model specifically trained to recognize financial charts from social media sources. It's designed to recognize if an image posted on social media such as Twitter, is a financial chart or something else.

## Introduction

Social media users post a lot of useful financial information, including their predictions of financial assets. However, it is often hard to distinguish if the images that they post also contain useful information. This model was developed to fill this gap, to recognize if an image is a financial chart. 

I use this model in combination with my two other projects [FinTwit-bot](https://github.com/StephanAkkerman/fintwit-bot) and [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment) to track market sentiment accross Twitter.

## Table of Contents
- [Datasets](#datasets)
- [Model Details](#model-details)
- [Model Results](#model-results)
- [Installation](#installation)
- [Usage](#usage)
- [Finetuning](#finetuning-datasets)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Datasets
chart-recognizer has been trained on my dataset. So far I have not been able to find another image dataset about financial charts. The dataset can be found on [Huggingface](https://huggingface.co/datasets/StephanAkkerman/fintwit-charts). It consists of about 2,000 images of financial tweets, with 80% including charts and 20% something else.

## Model Details
The model is finetuned from [Timm's efficientnet](https://huggingface.co/docs/timm/en/models/efficientnet) and has an accuracy of 97% on the 30% validation set.

## Installation
```bash
# Clone this repository
git clone https://github.com/StephanAkkerman/chart-recognizer
# Install required packages
pip install -r requirements.txt
```

## Usage
The model can be found on [Huggingface](https://huggingface.co/StephanAkkerman/chart-recognizer). It can be used together with the transformers library.

```python
from transformers import pipeline

# Create a sentiment analysis pipeline
pipe = pipeline(
    "image-classification",
    model="StephanAkkerman/chart-recognizer",
)

# Get the predicted sentiment
print(pipe(image))
```

## Citation
If you use chart-recognizer in your research, please cite as follows:

```bibtex
@misc{chart-recognizer,
  author = {Stephan Akkerman},
  title = {chart-recognizer: A Specialized Image Model for Financial Charts},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/chart-recognizer}}
}
```

## Contributing
Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
