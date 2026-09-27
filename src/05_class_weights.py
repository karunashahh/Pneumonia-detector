import os

# Get the project folder
project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

data_path = os.path.join(
    project_path,
    "Data",
    "chest_xray",
    "train"
)


normal_count = len(os.listdir(os.path.join(data_path, "NORMAL")))
pneumonia_count = len(os.listdir(os.path.join(data_path, "PNEUMONIA")))

total = normal_count + pneumonia_count

normal_weight = total / (2 * normal_count)
pneumonia_weight = total / (2 * pneumonia_count)


print("NORMAL images:", normal_count)
print("PNEUMONIA images:", pneumonia_count)

print("\nClass weights:")
print("NORMAL:", normal_weight)
print("PNEUMONIA:", pneumonia_weight)
