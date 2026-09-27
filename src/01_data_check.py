import os

# Get the project folder
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(project_path, "Data", "chest_xray")

train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "val")
test_path = os.path.join(data_path, "test")


print("Checking dataset folders...\n")

print("Train:", os.path.exists(train_path))
print("Validation:", os.path.exists(val_path))
print("Test:", os.path.exists(test_path))


print("\nClasses in each folder:")

for name, path in [("Train", train_path),
                   ("Validation", val_path),
                   ("Test", test_path)]:

    if os.path.exists(path):
        print(name, ":", os.listdir(path))
    else:
        print(name, ": folder not found")