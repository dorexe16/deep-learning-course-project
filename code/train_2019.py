import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from collections import Counter


from sklearn.metrics import classification_report
from torch.utils.data import Dataset
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
import imgaug.augmenters as iaa
import random
from utils.func_utils import calculate_class_weights, filter_invalid_images
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    average_precision_score,
)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

desired_train_size = False
ImageFile.LOAD_TRUNCATED_IMAGES = True  # This will allow loading truncated images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.__version__)

# Define paths
train_path = (
    "/data/talya/deep-learning-course-project/code/2019/ISIC_2019_Training_Input_2"
)
test_path = "/data/talya/deep-learning-course-project/code/2019/ISIC_2019_Test_Input"
labels_path = "/data/talya/deep-learning-course-project/code/2019/ISIC_2019_Training_GroundTruth.csv"


def evaluate_model(trainer, dataset, label_map, output_dir="./results"):
    print("Evaluating model...")
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Convert numerical labels to class names
    inverse_label_map = {v: k for k, v in label_map.items()}
    full_name_map = {
        "MEL": "melanoma",
        "NV": "melanocytic nevus",
        "BCC": "basal cell carcinoma",
        "AK": "actinic keratosis",
        "BKL": "benign keratosis",
        "DF": "dermatofibroma",
        "VASC": "vascular lesion",
        "SCC": "squamous cell carcinoma",
    }

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
    print("Classification Report:")
    print(report)

    # Sensitivity (Recall), Specificity, PPV, NPV, Accuracy, F1 Score
    sensitivity = recall_score(labels, preds, average=None)
    specificity = []
    ppv = precision_score(labels, preds, average=None)
    npv = []
    f1 = f1_score(labels, preds, average=None)
    accuracy = accuracy_score(labels, preds)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    num_classes = cm.shape[0]

    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
        npv.append(tn / (tn + fn) if (tn + fn) != 0 else 0)

    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    for i, class_name in enumerate(full_name_map.values()):
        print(f"{class_name} (Sensitivity/Recall): {sensitivity[i]:.4f}")
        print(f"{class_name} (Specificity): {specificity[i]:.4f}")
        print(f"{class_name} (PPV): {ppv[i]:.4f}")
        print(f"{class_name} (NPV): {npv[i]:.4f}")
        print(f"{class_name} (F1 Score): {f1[i]:.4f}")

    # AUC, AUC80, and Mean Average Precision
    auc_scores = {}
    auc80_scores = {}
    mean_ap = {}

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels == i, predictions.predictions[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[inverse_label_map[i]] = roc_auc

        # Calculate AUC80
        if np.sum(tpr >= 0.8) >= 2:  # Ensure at least 2 points are available
            auc80_scores[inverse_label_map[i]] = auc(fpr[tpr >= 0.8], tpr[tpr >= 0.8])
        else:
            auc80_scores[
                inverse_label_map[i]
            ] = None  # or set to 0.0, depending on your preference

        # Mean Average Precision
        mean_ap[inverse_label_map[i]] = average_precision_score(
            labels == i, predictions.predictions[:, i]
        )

        # Plot ROC curve
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f"{full_name_map[inverse_label_map[i]]} (AUC = {roc_auc:.2f})",
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    # Save the ROC curve plot
    roc_curve_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()

    print(f"\nROC curves saved to: {roc_curve_path}")

    print("\nAUC Scores:")
    for class_name, auc_score in auc_scores.items():
        print(f"{full_name_map[class_name]} (AUC): {auc_score:.4f}")

    print("\nAUC80 Scores:")
    for class_name, auc80_score in auc80_scores.items():
        if auc80_score is not None:
            print(f"{full_name_map[class_name]} (AUC80): {auc80_score:.4f}")
        else:
            print(f"{full_name_map[class_name]} (AUC80): Not enough data to calculate")

    print("\nMean Average Precision:")
    for class_name, map_score in mean_ap.items():
        print(f"{full_name_map[class_name]} (Mean Average Precision): {map_score:.4f}")
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "auc_scores": auc_scores,
        "auc80_scores": auc80_scores,
        "mean_ap": mean_ap,
        "classification_report": report,
    }


# Load the labels
labels_df = pd.read_csv(labels_path)
labels_df.set_index("image", inplace=True)

# Map class names to indices
class_names = labels_df.columns.tolist()
label_map = {class_name: idx for idx, class_name in enumerate(class_names)}


def get_label(img_name):
    labels = labels_df.loc[img_name].values
    return np.argmax(labels)  # Assuming one-hot encoding in the CSV file


class ElasticTransform:
    def __init__(self, alpha=36, sigma=6):
        self.alpha = alpha
        self.sigma = sigma
        self.augmenter = iaa.ElasticTransformation(alpha=self.alpha, sigma=self.sigma)

    def __call__(self, img):
        img = np.array(img)
        img = self.augmenter(image=img)
        return Image.fromarray(img)


class SkinCancerDataset(Dataset):
    def __init__(self, root_dir, labels_df, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels_df = labels_df
        self.images = list(
            self.labels_df.index
        )  # Use the image names from the labels file
        self.failed_loads = 0  # Counter for failed image loads

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(
            self.root_dir, img_name + ".jpg"
        )  # Assuming images have .jpg extension

        try:
            image = Image.open(img_path).convert("RGB")
        except OSError as e:
            print(f"Error loading image {img_path}: {e}")
            self.failed_loads += 1  # Increment the counter when a load fails
            return self.__getitem__(
                (idx + 1) % len(self)
            )  # Return next valid item (recursive call)

        label = get_label(img_name)

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


def get_label(img_name):
    labels = labels_df.loc[img_name].values
    return np.argmax(labels)  # Assuming one-hot encoding in the CSV file


def create_balanced_test_set(
    filenames, labels, percentage=0.05, min_samples_per_class=20
):
    test_filenames = []
    test_labels = []

    # Counter for each class
    class_counts = Counter(labels)

    for class_label in set(labels):
        # Get indices for this class
        class_indices = [i for i, label in enumerate(labels) if label == class_label]

        # Calculate number of samples for test set
        num_samples = max(
            int(class_counts[class_label] * percentage), min_samples_per_class
        )
        num_samples = min(
            num_samples, class_counts[class_label] - min_samples_per_class
        )  # Ensure min_samples_per_class remain in training set

        # Ensure we have a valid number of samples to take
        if num_samples > 0:
            selected_indices = random.sample(class_indices, num_samples)
            test_filenames.extend([filenames[i] for i in selected_indices])
            test_labels.extend([labels[i] for i in selected_indices])

    return test_filenames, test_labels


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


label_files = set(labels_df.index)

# # Get the list of files in the train and test directories
train_files = set([os.path.splitext(f)[0] for f in os.listdir(train_path)])
test_files = set([os.path.splitext(f)[0] for f in os.listdir(test_path)])

# # Find the intersection between the label files and the files in the directories
train_files_in_labels = train_files.intersection(label_files)
test_files_in_labels = test_files.intersection(label_files)

# # Print the number of matching files
print(f"Number of training images with labels: {len(train_files_in_labels)}")
print(f"Number of testing images with labels: {len(test_files_in_labels)}")


train_filenames = filter_invalid_images(list(labels_df.index), train_path)
train_labels = [get_label(fname) for fname in train_filenames]

initial_class_counts = Counter(train_labels)
print(f"Initial class distribution: {initial_class_counts}")
test_filenames_split, test_labels_split = create_balanced_test_set(
    train_filenames, train_labels, min_samples_per_class=50
)

# The remaining files will be your training set
train_filenames_split = [
    fname for fname in train_filenames if fname not in test_filenames_split
]
train_labels_split = [get_label(fname) for fname in train_filenames_split]

# Verify the class distribution in the test set
test_class_counts = Counter(test_labels_split)
print(f"Test set class distribution: {test_class_counts}")

# Verify the class distribution in the training set
train_class_counts = Counter(train_labels_split)
print(f"Training set class distribution: {train_class_counts}")

if desired_train_size:

    sampled_indices = random.sample(
        range(len(train_filenames_split)), desired_train_size
    )
    train_filenames_sampled = [train_filenames_split[i] for i in sampled_indices]
    train_labels_sampled = [train_labels_split[i] for i in sampled_indices]

    # Verify the class distribution in the sampled training set
    train_class_counts_sampled = Counter(train_labels_sampled)
    print(f"Sampled Training set class distribution: {train_class_counts_sampled}")

    train_dataset = SkinCancerDataset(
        root_dir=train_path,
        labels_df=labels_df.loc[train_filenames_sampled],
        transform=train_transform,
    )
else:
    train_dataset = SkinCancerDataset(
        root_dir=train_path,
        labels_df=labels_df.loc[train_filenames_split],
        transform=train_transform,
    )
test_dataset = SkinCancerDataset(
    root_dir=train_path,
    labels_df=labels_df.loc[test_filenames_split],
    transform=val_transform,
)

# Print sizes
print("New Train Size: ", len(train_dataset))
print("New Test Size: ", len(test_dataset))

class_weights = calculate_class_weights(train_dataset, label_map, device)

# Preprocess function
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# Load the saved model
fine_tuned_model = ViTForImageClassification.from_pretrained(
    "./overfitted_vit_model", num_labels=len(class_names), ignore_mismatched_sizes=True
)
fine_tuned_model.to(device)


# with LR scheduler
fine_tuning_args = TrainingArguments(
    output_dir="./fine_tuned_results",
    num_train_epochs=20,  # Fewer epochs for fine-tuning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./fine_tuned_logs",
    logging_steps=10,
    eval_strategy="epoch",
    fp16=True,
    lr_scheduler_type="cosine",  # You can choose from 'linear', 'cosine', 'cosine_with_restarts', etc.
)

# Instantiate the Trainer for fine-tuning
fine_tuner = CustomTrainer(
    model=fine_tuned_model,
    args=fine_tuning_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Ensure only the test set is used for evaluation
    data_collator=DefaultDataCollator(),
    tokenizer=image_processor,
    class_weights=class_weights,  # Pass the class weights here
)

# Fine-tune the model
fine_tuner.train()

# Evaluate on the test set
evaluate_model(fine_tuner, test_dataset, label_map)
fine_tuned_model.save_pretrained("./fine_tuned_vit_model25")


# melanoma (MEL), melanocytic nevus (NV), basal cell carcinoma (BCC), actinic keratosis (AK), benign keratosis (BKL), dermatofibroma (DF), vascular lesion (VASC), and squamous cell carcinoma (SCC).
