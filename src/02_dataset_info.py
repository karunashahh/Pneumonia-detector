import os

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

data_path = os.path.join(
    project_path,
    "Data",
    "chest_xray"
)


def count_images(folder):
    counts = {}

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)

        if not os.path.isdir(class_path):
            continue

        count = 0

        for file in os.listdir(class_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1

        counts[class_name] = count

    return counts


for split in ["train", "val", "test"]:

    folder = os.path.join(data_path, split)
    counts = count_images(folder)

    print(f"\n{split.upper()}")

    for class_name, count in counts.items():
        print(f"{class_name}: {count}")

    print("Total:", sum(counts.values()))