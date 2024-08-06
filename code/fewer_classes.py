import os
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers.data.data_collator import torch_default_data_collator
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.__version__)
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of ViT
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=15),   # Randomly rotate images by ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
    transforms.RandomGrayscale(p=0.1),       # Convert images to grayscale with a probability
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),  # Random Gaussian blur
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.1),  # Random affine transformations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define custom elastic deformation transformation
class ElasticTransform:
    def __init__(self, alpha=36, sigma=6):
        self.alpha = alpha
        self.sigma = sigma
        self.augmenter = iaa.ElasticTransformation(alpha=self.alpha, sigma=self.sigma)

    def __call__(self, img):
        img = np.array(img)
        img = self.augmenter(image=img)
        return Image.fromarray(img)

# Define transformations for the Skin Cancer Dataset
skin_cancer_transform = transforms.Compose([
    ElasticTransform(alpha=36, sigma=6),  # Apply elastic deformation
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset class with label grouping
class SkinCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {
            'non-melanoma skin cancer': 0,
            'benign lesion': 1,
            'pre-malignant/malignant': 2
        }
        class_mapping = {
            'actinic keratosis': 'pre-malignant/malignant',
            'basal cell carcinoma': 'non-melanoma skin cancer',
            'dermatofibroma': 'benign lesion',
            'melanoma': 'pre-malignant/malignant',
            'nevus': 'benign lesion',
            'pigmented benign keratosis': 'benign lesion',
            'seborrheic keratosis': 'benign lesion',
            'squamous cell carcinoma': 'non-melanoma skin cancer'
        }

        for original_label in os.listdir(root_dir):
            if original_label == '.DS_Store':
                continue
            new_label = class_mapping[original_label]
            for img_name in os.listdir(os.path.join(root_dir, original_label)):
                self.images.append(os.path.join(root_dir, original_label, img_name))
                self.labels.append(self.label_map[new_label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": label}

# Create datasets and dataloaders
train_dataset = SkinCancerDataset(root_dir=train_path, transform=skin_cancer_transform)
test_dataset = SkinCancerDataset(root_dir=test_path, transform=skin_cancer_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Calculate class weights
class_counts = Counter(train_dataset.labels)
class_weights = {class_id: 1.0 / count for class_id, count in class_counts.items()}
weights = [class_weights[label] for label in train_dataset.labels]
sample_weights = torch.tensor(weights, dtype=torch.float)
weighted_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

# Custom trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(weight=torch.tensor(list(class_weights.values())).to(logits.device))(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Load the saved model
fine_tuned_model = ViTForImageClassification.from_pretrained("./overfitted_vit_model", num_labels=3, ignore_mismatched_sizes=True)
fine_tuned_model.to(device)

# Adjust training arguments for fine-tuning
fine_tuning_args = TrainingArguments(
    output_dir='./fine_tuned_results',
    num_train_epochs=10,  # Fewer epochs for fine-tuning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./fine_tuned_logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    fp16=True,
)

# Instantiate the Trainer for fine-tuning
fine_tuner = CustomTrainer(
    model=fine_tuned_model,
    args=fine_tuning_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DefaultDataCollator(),
    tokenizer=image_processor,
)

def evaluate_model(trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Define the class names
    class_names = ['non-melanoma skin cancer', 'benign lesion', 'pre-malignant/malignant']

    report = classification_report(
        labels,
        preds,
        target_names=class_names,  # Specify class names
        zero_division=0  # Handle classes with no predicted samples
    )
    print(report)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    correct_preds = np.where(preds == labels)[0]
    incorrect_preds = np.where(preds != labels)[0]

    def save_examples(dataset, indices, labels, preds, prefix):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        for i in indices[:10]:  # Save first 10 examples
            sample = dataset[i]
            img = sample['pixel_values']
            img = img.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
            img = std * img + mean  # Unnormalize
            img = np.clip(img, 0, 1)  # Ensure values are in range [0, 1]
            true_label = labels[i]
            pred_label = preds[i]
            plt.imshow(img)
            plt.title(f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}")
            plt.axis('off')
            plt.savefig(f"{prefix}_example_{i}.png")
            plt.close()
            
# Fine-tune the model
fine_tuner.train()
evaluate_model(fine_tuner, test_dataset)
