import tensorflow as tf
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
test_path = os.path.join(data_path, "test")

img_size = (224, 224)
batch_size = 32
seed = 42


# Use part of the training data for validation
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)

val_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)

# Keep the test set completely separate
test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary",
    shuffle=False
)


print("\nClass names:")
print(train_data.class_names)

print("\nDataset sizes:")
print("Training batches:", tf.data.experimental.cardinality(train_data).numpy())
print("Validation batches:", tf.data.experimental.cardinality(val_data).numpy())
print("Test batches:", tf.data.experimental.cardinality(test_data).numpy())