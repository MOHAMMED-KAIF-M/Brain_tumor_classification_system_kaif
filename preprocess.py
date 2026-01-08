import os
import imagehash
from PIL import Image

# ✅ Correct Dataset Path (Use your real path here)
dataset_root = r"C:\Users\Mohammed kaif M\OneDrive\Desktop\Brain tumor\datasets"

# Folders containing class subfolders
main_folders = ["training", "testing"]

# Class names (same inside both folders)
class_folders = [
    "pituitary_tumor",
    "glioma_tumor",
    "no_tumor",
    "meningioma_tumor"
]

hashes = {}   # stores unique image hashes

for main in main_folders:                     # training + testing
    for cls in class_folders:                 # 4 classes
        folder_path = os.path.join(dataset_root, main, cls)

        print(f"\nChecking folder: {folder_path}")

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            try:
                img = Image.open(file_path)
                img_hash = str(imagehash.phash(img))

                if img_hash in hashes:
                    print(f"Duplicate found → Deleting: {file_path}")
                    os.remove(file_path)
                else:
                    hashes[img_hash] = file_path

            except Exception as e:
                print("Error:", e)
