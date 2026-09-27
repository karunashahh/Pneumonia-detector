import tensorflow as tf
import numpy as np
import os

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
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

test_path = os.path.join(data_path, "test")

model_path = os.path.join(
    project_path,
    "vgg16_pneumonia_best.keras"
)


img_size = (224, 224)
batch_size = 32


# Load the test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary",
    shuffle=False
)


print("Classes:", test_data.class_names)


# Load the trained model
model = tf.keras.models.load_model(model_path)


# Get the true labels
y_true = np.concatenate([
    labels.numpy().flatten()
    for images, labels in test_data
])


# Get model predictions
y_probability = model.predict(test_data)

y_probability = y_probability.flatten()

y_pred = (y_probability >= 0.5).astype(int)


# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()


# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0

recall = tp / (tp + fn) if (tp + fn) > 0 else 0

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

f1 = f1_score(y_true, y_pred)

auc = roc_auc_score(y_true, y_probability)


print("\nConfusion Matrix:")
print(cm)

print("\nTest Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall / Sensitivity:", recall)
print("Specificity:", specificity)
print("F1 Score:", f1)
print("ROC-AUC:", auc)


print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=test_data.class_names
    )
)