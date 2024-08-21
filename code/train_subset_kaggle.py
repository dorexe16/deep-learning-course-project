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
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.__version__)

# # Define paths
train_path = "/data/talya/deep-learning-course-project/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
test_path = "/data/talya/deep-learning-course-project/Skin cancer ISIC The International Skin Imaging Collaboration/Test"

add_samples_to_test = False


def calculate_class_weights(dataset, label_map):
    label_counts = Counter()
    for i in range(len(dataset)):
        label = dataset[i]["labels"]
        label_counts[label] += 1

    total_samples = len(dataset)
    num_classes = len(label_map)
    class_weights = [0] * num_classes

    for label, count in label_counts.items():
        class_weights[label] = total_samples / (num_classes * count)

    return torch.tensor(class_weights).to(device)


def ensure_min_test_samples(train_dataset, test_dataset, label_map, min_samples=30):
    print("Ensuring minimum test samples...")
    class_counts = {label: 0 for label in label_map.values()}

    # Count samples in the test dataset
    for idx in range(len(test_dataset)):
        label = test_dataset[idx]["labels"]
        class_counts[label] += 1

    # Move samples from train to test dataset if needed
    additional_samples = []
    for label, count in class_counts.items():
        print(f"Class {label} has {count} samples.")
        if count < min_samples:
            needed = min_samples - count
            # Find indices of this label in the train dataset
            train_indices = [
                i for i, sample in enumerate(train_dataset) if sample["labels"] == label
            ]
            selected_indices = train_indices[:needed]

            # Add these samples to the test dataset and remove them from the train dataset
            additional_samples.extend([train_dataset[i] for i in selected_indices])
            train_dataset = Subset(
                train_dataset,
                [i for i in range(len(train_dataset)) if i not in selected_indices],
            )

    # Combine additional samples with the original test dataset
    combined_dataset = list(test_dataset) + additional_samples
    test_dataset = Subset(combined_dataset, list(range(len(combined_dataset))))

    return train_dataset, test_dataset


# Custom evaluation function to compute accuracy
def compute_metrics(eval_pred):
    # Unpack predictions and labels
    logits, labels = eval_pred
    # Convert logits to tensor
    logits = torch.tensor(logits)
    # Convert labels to tensor
    labels = torch.tensor(labels)
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)
    # Calculate accuracy
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}


# Custom data collator
def collate_fn(batch):
    # Unpack the batch and split into images and labels
    images, labels = zip(*batch)
    # Stack images and convert them into a tensor
    images = torch.stack(images)
    # Convert labels into a tensor
    labels = torch.tensor(labels)
    return {"pixel_values": images, "labels": labels}


def evaluate_model(trainer, test_dataset, label_map):
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Reverse the label map to get class names from indices
    class_names = {v: k for k, v in label_map.items()}

    # Classification report
    all_labels = range(len(class_names))  # Ensure this matches the number of classes
    report = classification_report(
        labels,
        preds,
        labels=all_labels,  # Specify all possible labels
        target_names=[class_names[i] for i in all_labels],  # Use real class names
        zero_division=0,  # Handle classes with no predicted samples
    )
    print(report)

    # Save classification report to file
    with open("classification_report.txt", "w") as f:
        f.write(report)


class ElasticTransform:
    def __init__(self, alpha=36, sigma=6):
        self.alpha = alpha
        self.sigma = sigma
        self.augmenter = iaa.ElasticTransformation(alpha=self.alpha, sigma=self.sigma)

    def __call__(self, img):
        img = np.array(img)
        img = self.augmenter(image=img)
        return Image.fromarray(img)


# Custom Dataset class
class SkinCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {
            label: idx
            for idx, label in enumerate(sorted(os.listdir(root_dir)))
            if label != ".DS_Store"
        }

        for label in self.label_map:
            for img_name in os.listdir(os.path.join(root_dir, label)):
                self.images.append(os.path.join(root_dir, label, img_name))
                self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": label}


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to the input size of ViT
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
        transforms.RandomRotation(degrees=15),  # Randomly rotate images by ±15 degrees
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # Random color adjustments
        transforms.RandomGrayscale(p=0.1),
        ElasticTransform(
            alpha=36, sigma=6
        ),  # Convert images to grayscale with a probability
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Define transformations for validation (no augmentation)
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to the input size of ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=0.00001,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=76,
    weight_decay=0.001,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


original_train_dataset = SkinCancerDataset(
    root_dir=train_path, transform=train_transform
)
original_test_dataset = SkinCancerDataset(root_dir=test_path, transform=val_transform)

# Store the label map from the original dataset
label_map = original_test_dataset.label_map
train_dataset = Subset(original_train_dataset, range(len(original_train_dataset)))
test_dataset = Subset(original_test_dataset, range(len(original_test_dataset)))
# Print size of items in train and in test
print("Original Train Size: ", len(original_train_dataset))
print("Original Test Size: ", len(original_test_dataset))
class_weights = calculate_class_weights(train_dataset, label_map)


if add_samples_to_test:
    train_dataset, test_dataset = ensure_min_test_samples(
        train_dataset, test_dataset, label_map, min_samples=40
    )
    print("Train Size: ", len(train_dataset))
    print("Test Size: ", len(test_dataset))
    # save new train and test datasets
    torch.save(train_dataset, "train_dataset.pth")
    torch.save(test_dataset, "test_dataset.pth")
else:
    # load the train and test datasets
    train_dataset = torch.load("train_dataset.pth")
    test_dataset = torch.load("test_dataset.pth")


# Preprocess function
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# Load the saved model
fine_tuned_model = ViTForImageClassification.from_pretrained(
    "./overfitted_vit_model", num_labels=9, ignore_mismatched_sizes=True
)
fine_tuned_model.to(device)

# Adjust training arguments for fine-tuning
fine_tuning_args = TrainingArguments(
    output_dir="./fine_tuned_results",
    num_train_epochs=25,  # Fewer epochs for fine-tuning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./fine_tuned_logs",
    logging_steps=10,
    eval_strategy="epoch",
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
    class_weights=class_weights,  # Pass the class weights here
)


# Fine-tune the model
fine_tuner.train()
evaluate_model(fine_tuner, test_dataset, label_map)
fine_tuned_model.save_pretrained("./fine_tuned_vit_model25")
