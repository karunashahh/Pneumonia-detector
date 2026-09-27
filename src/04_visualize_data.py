import tensorflow as tf
import matplotlib.pyplot as plt
import os


project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

data_path = os.path.join(
    project_path,
    "Data",
    "chest_xray"
)

train_path = os.path.join(data_path, "train")


img_size = (224, 224)
batch_size = 32
seed = 42


train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)


class_names = train_data.class_names

print("Classes:", class_names)


# Get one batch of images
images, labels = next(iter(train_data))


plt.figure(figsize=(10, 8))

for i in range(9):
    plt.subplot(3, 3, i + 1)

    plt.imshow(images[i].numpy().astype("uint8"))

    label = int(labels[i].numpy()[0])
    plt.title(class_names[label])

    plt.axis("off")

plt.tight_layout()
plt.show()