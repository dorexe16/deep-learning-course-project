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
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of ViT
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=15),   # Randomly rotate images by ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
    transforms.RandomGrayscale(p=0.1),       # Convert images to grayscale with a probability
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transformations for validation (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the DTD dataset
dtd_data_train = datasets.DTD(root="./data", split="train", download=True, transform=train_transform)
dtd_data_val = datasets.DTD(root="./data", split="train", download=True, transform=val_transform)

# To overfit, select a small subset of the dataset
num_samples = 500
train_indices = random.sample(range(len(dtd_data_train)), num_samples)
train_data = Subset(dtd_data_train, train_indices)

# Use a different subset for evaluation
val_indices = random.sample(range(len(dtd_data_val)), num_samples)
val_data = Subset(dtd_data_val, val_indices)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=400, shuffle=True)
val_loader = DataLoader(val_data, batch_size=50, shuffle=False)

# Custom data collator
def collate_fn(batch):
    # Unpack the batch and split into images and labels
    images, labels = zip(*batch)
    # Stack images and convert them into a tensor
    images = torch.stack(images)
    # Convert labels into a tensor
    labels = torch.tensor(labels)
    return {"pixel_values": images, "labels": labels}

# Load pretrained ViT model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=47
).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=0.000001,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=75,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

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

# Define Trainer with custom data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,  # Use custom data collator
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the validation set
results = trainer.evaluate()
print("Validation Accuracy:", results["eval_accuracy"])
model.save_pretrained("./overfitted_vit_model")
#

#------------------------------------------------------------------------fine tune

# # Define paths
# train_path = "C:/Users/dorex/Desktop/deed_learning_research/skin_cencer_dataset/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
# test_path = "C:/Users/dorex/Desktop/deed_learning_research/skin_cencer_dataset/Skin cancer ISIC The International Skin Imaging Collaboration/Test"
train_path = "/data/talya/deep-learning-course-project/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
test_path = "/data/talya/deep-learning-course-project/Skin cancer ISIC The International Skin Imaging Collaboration/Test"


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


# Define transformations
transform = transforms.Compose([
    ElasticTransform(alpha=36, sigma=6),  # Apply elastic deformation
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Custom Dataset class
class SkinCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(root_dir))) if label != '.DS_Store'}

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


train_dataset = SkinCancerDataset(root_dir=train_path, transform=transform)
test_dataset = SkinCancerDataset(root_dir=test_path, transform=transform)

# Load pre-trained ViT model
# model = ViTForImageClassification.from_pretrained(
#     'google/vit-base-patch16-224-in21k',
#     num_labels=len(train_dataset.label_map)  # Number of classes in your dataset
# )
# model.to(device)  # Move model to GPU

# Preprocess function
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')



# Custom trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss



# Load the saved model
fine_tuned_model = ViTForImageClassification.from_pretrained("./overfitted_vit_model",num_labels=9,ignore_mismatched_sizes=True)
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
)


def evaluate_model(trainer, test_dataset):
    # Get predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Classification report
    all_labels = range(9)  # Ensure this matches the number of classes
    report = classification_report(
        labels,
        preds,
        labels=all_labels,  # Specify all possible labels
        target_names=[f"Class_{i}" for i in all_labels],
        zero_division=0  # Handle classes with no predicted samples
    )
    print(report)

    # Save classification report to file
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # Save some example predictions
    correct_preds = np.where(preds == labels)[0]
    incorrect_preds = np.where(preds != labels)[0]

    def save_examples(dataset, indices, labels, preds, prefix):
        for i in indices[:10]:  # Save first 10 examples
            sample = dataset[i]  # Access the sample
            img = sample['pixel_values']  # Get the image tensor
            img = img.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
            true_label = labels[i]
            pred_label = preds[i]
            plt.imshow(img)
            plt.title(f"True: {true_label}, Pred: {pred_label}")
            plt.axis('off')  # Turn off axis for better image display
            plt.savefig(f"{prefix}_example_{i}.png")
            plt.close()

# Fine-tune the model
fine_tuner.train()
evaluate_model(fine_tuner, test_dataset)
#
# data_collator = DefaultDataCollator()
#
#
#
# # Instantiate the Trainer
# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     data_collator=data_collator,
#     tokenizer=image_processor,
# )
#
# # Train the model
# trainer.train()
#
# # Evaluate the model
# results = trainer.evaluate()
# print(results)
#
#
# # Generate detailed evaluation report
#
#
# evaluate_model(trainer, test_dataset)
