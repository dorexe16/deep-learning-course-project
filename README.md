# Fine-tune Vision Transformer (ViT) Model

This project focuses on fine-tuning a Vision Transformer (ViT) model on a skin cancer dataset to classify different types of skin lesions. The dataset contains images of various skin cancer types, including malignant and benign oncological diseases.

## Dataset

The dataset used for this project is available on Kaggle: [Skin Cancer 9 Classes ISIC](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).


## Create Virtual Envioment with Conda
To set up the environment and install the required packages, follow these steps:
```bash
conda create --name vit_env python=3.8
conda activate vit_env
pip install -r requirements.txt
```

## How to Train

To train the model, run the `train.py` file. This script handles data preprocessing, model training, and evaluation.

```bash
python train.py
```
