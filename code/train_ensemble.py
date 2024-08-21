import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Load pretrained EfficientNet model

add_samples_to_test = False
num_folds = 5
num_labels = 9

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.__version__)

efficientnet_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
efficientnet_model.classifier[1] = nn.Linear(
    efficientnet_model.classifier[1].in_features, num_labels
)  # Adjust the output layer to match the number of classes
efficientnet_model.to(device)

train_path = "/data/talya/deep-learning-course-project/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
test_path = "/data/talya/deep-learning-course-project/Skin cancer ISIC The International Skin Imaging Collaboration/Test"


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


# Custom data collator
def collate_fn(batch):
    # Unpack the batch and split into images and labels
    images, labels = zip(*[(item["pixel_values"], item["labels"]) for item in batch])
    # Stack images and convert them into a tensor
    images = torch.stack(images)
    # Convert labels into a tensor
    labels = torch.tensor(labels)
    return {"pixel_values": images, "labels": labels}


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


# Custom Trainer to handle ensemble
class EnsembleTrainer(Trainer):
    def __init__(
        self, *args, vit_model=None, effnet_model=None, class_weights=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.vit_model = vit_model
        self.effnet_model = effnet_model
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        vit_outputs = self.vit_model(**inputs)
        effnet_outputs = self.effnet_model(
            inputs["pixel_values"].to(device)
        )  # Ensure tensors are sent to the device

        # Combine logits from both models
        combined_logits = (vit_outputs.logits + effnet_outputs) / 2

        # Compute loss with combined logits
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(combined_logits, labels)
        return (loss, vit_outputs) if return_outputs else loss

    def predict(self, test_dataset):
        vit_predictions = super().predict(test_dataset)
        effnet_predictions = []

        for batch in DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=collate_fn,
        ):
            with torch.no_grad():
                effnet_outputs = self.effnet_model(batch["pixel_values"].to(device))
                effnet_predictions.append(effnet_outputs.cpu().numpy())

        effnet_predictions = np.concatenate(effnet_predictions, axis=0)
        combined_logits = (vit_predictions.predictions + effnet_predictions) / 2
        combined_predictions = np.argmax(combined_logits, axis=1)

        return combined_predictions, vit_predictions.label_ids


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

original_train_dataset = SkinCancerDataset(
    root_dir=train_path, transform=train_transform
)
original_test_dataset = SkinCancerDataset(root_dir=test_path, transform=val_transform)

# Store the label map from the original dataset
label_map = original_test_dataset.label_map
class_names = {v: k for k, v in label_map.items()}

train_dataset = Subset(original_train_dataset, range(len(original_train_dataset)))
test_dataset = Subset(original_test_dataset, range(len(original_test_dataset)))

# Print size of items in train and in test
print("Original Train Size: ", len(original_train_dataset))
print("Original Test Size: ", len(original_test_dataset))


if add_samples_to_test:
    train_dataset, test_dataset = ensure_min_test_samples(
        train_dataset, test_dataset, label_map, min_samples=40
    )
    print("Train Size: ", len(train_dataset))
    print("Test Size: ", len(test_dataset))
    # Preprocess function
    # save new train and test datasets
    torch.save(train_dataset, "train_dataset.pth")
    torch.save(test_dataset, "test_dataset.pth")
else:
    # load the train and test datasets
    train_dataset = torch.load("train_dataset.pth")
    test_dataset = torch.load("test_dataset.pth")

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


# Load the saved model
fine_tuned_model = ViTForImageClassification.from_pretrained(
    "./overfitted_vit_model", num_labels=9, ignore_mismatched_sizes=True
)
fine_tuned_model.to(device)

# Create the KFold object


def train():
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Initialize lists to store results for each fold
    fold_accuracies = []
    fold_classification_reports = []
    class_weights = calculate_class_weights(train_dataset, label_map)

    for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):
        print(f"Training fold {fold + 1}/{num_folds}...")

        # Create train and validation subsets for the current fold
        train_subset = Subset(train_dataset, train_index)
        val_subset = Subset(train_dataset, val_index)

        # Reinitialize the models to reset weights
        vit_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=9,
            ignore_mismatched_sizes=True,
        ).to(device)

        effnet_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        effnet_model.classifier[1] = nn.Linear(
            effnet_model.classifier[1].in_features, num_labels
        )
        effnet_model.to(device)

        # Define new training arguments for this fold
        fold_training_args = TrainingArguments(
            output_dir=f"./fine_tuned_results_fold_{fold + 1}",
            num_train_epochs=25,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"./fine_tuned_logs_fold_{fold + 1}",
            logging_steps=10,
            eval_strategy="epoch",
            fp16=True,
        )

        # Instantiate the Ensemble Trainer for this fold
        fold_trainer = EnsembleTrainer(
            model=vit_model,
            args=fold_training_args,
            vit_model=vit_model,
            effnet_model=effnet_model,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            data_collator=DefaultDataCollator(),
            tokenizer=image_processor,
            class_weights=class_weights,
        )

        # Fine-tune the ensemble model for this fold
        fold_trainer.train()

        # Evaluate the model on the validation set for this fold
        ensemble_predictions, labels = fold_trainer.predict(val_subset)

        # Calculate accuracy and generate a classification report for this fold
        accuracy = np.mean(ensemble_predictions == labels)
        fold_accuracies.append(accuracy)

        report = classification_report(
            labels,
            ensemble_predictions,
            labels=range(len(class_names)),
            target_names=[class_names[i] for i in range(len(class_names))],
            zero_division=0,
        )
        fold_classification_reports.append(report)

        print(f"Fold {fold + 1} accuracy: {accuracy}")
        print(report)

    # After all folds are completed, summarize the results
    print(f"Mean accuracy over {num_folds} folds: {np.mean(fold_accuracies)}")

    # Save the results to file
    with open("kfold_classification_reports.txt", "w") as f:
        for fold, report in enumerate(fold_classification_reports):
            f.write(f"Fold {fold + 1} Report:\n")
            f.write(report)
            f.write("\n")

    # Save the final ensemble models (optional, can choose to save one of the models)
    vit_model.save_pretrained("./final_fine_tuned_vit_model")
    torch.save(effnet_model.state_dict(), "./final_fine_tuned_efficientnet_model.pth")


def predict_test():
    # Define the path to the saved models from the 2nd fold
    vit_model_path = "./fine_tuned_results_fold_2"
    effnet_model_path = "./fine_tuned_results_fold_2/efficientnet_model.pth"

    # Load the ViT model
    vit_model = ViTForImageClassification.from_pretrained(
        "./final_fine_tuned_vit_model"
    )
    vit_model.to(device)

    # Load the fine-tuned EfficientNet model
    effnet_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    effnet_model.classifier[1] = nn.Linear(
        effnet_model.classifier[1].in_features, num_labels
    )
    effnet_model.load_state_dict(
        torch.load("./final_fine_tuned_efficientnet_model.pth")
    )
    effnet_model.to(device)

    # Set both models to evaluation mode
    vit_model.eval()
    effnet_model.eval()

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to the input size of ViT
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the test dataset
    # test_dataset = SkinCancerDataset(root_dir=test_path, transform=val_transform)
    test_dataset = torch.load("test_dataset.pth")

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Get predictions from both models
            vit_outputs = vit_model(inputs)
            effnet_outputs = effnet_model(inputs)

            # Combine logits from both models
            combined_logits = (vit_outputs.logits + effnet_outputs) / 2

            # Get the final predictions
            predictions = torch.argmax(combined_logits, dim=1)

            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate a classification report
    from sklearn.metrics import classification_report

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=[class_names[i] for i in range(len(class_names))],
        zero_division=0,
    )

    # Print the classification report
    print(report)

    # Optionally, save the classification report to a file
    with open("final_test_classification_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    # train()
    predict_test()
