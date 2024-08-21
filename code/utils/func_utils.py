from collections import Counter
from PIL import Image, ImageFile
import os
import torch
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import Subset


def calculate_class_weights(dataset, label_map, device):
    label_counts = Counter()
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is None:  # Skip None samples
            continue
        label = sample["labels"]
        label_counts[label] += 1

    total_samples = sum(label_counts.values())
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
    # Filter out None values
    batch = [item for item in batch if item is not None]
    images, labels = zip(*[(item["pixel_values"], item["labels"]) for item in batch])
    images = torch.stack(images)
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


def filter_invalid_images(file_list, root_dir):
    valid_files = []
    for img_name in file_list:
        img_path = os.path.join(root_dir, img_name + ".jpg")
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify that the file is an image
            valid_files.append(img_name)
        except (IOError, SyntaxError) as e:
            print(f"Removing invalid image: {img_path}, error: {e}")
    return valid_files
