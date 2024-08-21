# Fine-tune Vision Transformer (ViT) Model

This project focuses on fine-tuning a Vision Transformer (ViT) model on a skin cancer dataset to classify different types of skin lesions. The dataset contains images of various skin cancer types, including malignant and benign oncological diseases.

## Dataset

The dataset used for this project 
1. Subset from ISIC on Kaggle: [Skin Cancer 9 Classes ISIC - Kaggle Subset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).
2. ISIC 2019 Challange dataset: [Skin Cancer 9 Classes ISIC - 2019 Dataset](https://challenge.isic-archive.com/data/#2019).


## Create Virtual Envioment with Conda
To set up the environment and install the required packages, follow these steps:
```bash
conda create --name vit_env python=3.8
conda activate vit_env
pip install -r requirements.txt
```

## How to Train

We conduct a few different experiments 
- class reduction
- Transfer Learning on DTD
- ViT Hybrid Model
- ViT direct finetune
- Ensemble model with EfficientNet
All the different methods are in the `code` folder. 
To train the model, run one of the `train_x.py` files. This script handles data preprocessing, model training, and evaluation.

```bash
python train_x.py
```

## GIT 

`git clone https://github.com/talyabs/deep-learning-course-project.git`

- new branch
`git checkout -b develop`

1. make a change to file
2. Add it with + 
3. commit message - upper left, buttom commit
4. `git push` /  `git push --set-upstream origin develop`
5. PR + merge
6. pull changes `git pull` 