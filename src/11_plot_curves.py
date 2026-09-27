import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
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


# Get labels and predictions from the same batches
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


# ROC curve
fpr, tpr, thresholds = roc_curve(
    y_true,
    y_probability
)

roc_auc = roc_auc_score(
    y_true,
    y_probability
)


plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.show()


# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(
    y_true,
    y_probability
)

average_precision = average_precision_score(
    y_true,
    y_probability
)


plt.figure(figsize=(7, 6))

plt.plot(
    recall,
    precision,
    label=f"Average Precision = {average_precision:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()

plt.tight_layout()

plt.show()


print("\nCurve Results:")
print("ROC-AUC:", roc_auc)
print("Average Precision:", average_precision)