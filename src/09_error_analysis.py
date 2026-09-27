import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


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


# Load test data
test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary",
    shuffle=False
)


class_names = test_data.class_names

print("Classes:", class_names)


# Load trained model
model = tf.keras.models.load_model(model_path)


# Get images and labels
images = []
labels = []

for batch_images, batch_labels in test_data:
    images.extend(batch_images.numpy())
    labels.extend(batch_labels.numpy().flatten())


images = np.array(images)
labels = np.array(labels)


# Get predictions
probabilities = model.predict(images, batch_size=batch_size).flatten()

predictions = (probabilities >= 0.5).astype(int)


# Find incorrect predictions
wrong_indices = np.where(predictions != labels)[0]

print("\nTotal incorrect predictions:", len(wrong_indices))


# Show some incorrect predictions
plt.figure(figsize=(12, 10))

number_to_show = min(12, len(wrong_indices))

for i in range(number_to_show):

    index = wrong_indices[i]

    plt.subplot(3, 4, i + 1)

    plt.imshow(images[index].astype("uint8"), cmap="gray")

    true_label = class_names[int(labels[index])]
    predicted_label = class_names[int(predictions[index])]

    confidence = probabilities[index]

    plt.title(
        f"True: {true_label}\n"
        f"Pred: {predicted_label}\n"
        f"Prob: {confidence:.2f}"
    )

    plt.axis("off")


plt.tight_layout()
plt.show()