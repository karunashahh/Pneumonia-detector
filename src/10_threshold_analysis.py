import tensorflow as tf
import numpy as np
import os

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# Get the project folder
project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

data_path = os.path.join(
    project_path,
    "Data",
    "chest_xray"
)

train_path = os.path.join(data_path, "train")

model_path = os.path.join(
    project_path,
    "vgg16_pneumonia_best.keras"
)


img_size = (224, 224)
batch_size = 32
seed = 42


# Load validation data
val_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary",
    shuffle=True
)


print("Classes:", val_data.class_names)


# Load trained model
model = tf.keras.models.load_model(model_path)


# Get labels and predictions from the SAME batches
y_true = []
y_probability = []


for images, labels in val_data:

    predictions = model.predict(
        images,
        verbose=0
    ).flatten()

    y_true.extend(labels.numpy().flatten())
    y_probability.extend(predictions)


y_true = np.array(y_true)
y_probability = np.array(y_probability)


# Test different thresholds
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


print("\nThreshold Analysis")
print("-" * 80)


for threshold in thresholds:

    y_pred = (y_probability >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = tn / (tn + fp)

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy:.3f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"Specificity: {specificity:.3f} | "
        f"F1: {f1:.3f}"
    )