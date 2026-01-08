import cv2
import os

input_root = r"C:\\Users\\Mohammed kaif M\\OneDrive\\Desktop\\Brain tumor\\datasets"
output_root = r"C:\\Users\\Mohammed kaif M\\OneDrive\\Desktop\\Brain tumor\\resized_datasets"

img_size = (224, 224)

classes = [
    "glioma_tumor",
    "no_tumor",
    "meningioma_tumor",
    "pituitary_tumor"
]

splits = ["training", "testing"]

# Create output root folder
os.makedirs(output_root, exist_ok=True)

total_count = 0  # total resized images

for split in splits:
    print(f"\n=== Processing {split.upper()} data ===")
    split_count = 0  # per split
    
    split_input_path = os.path.join(input_root, split)
    split_output_path = os.path.join(output_root, split)
    os.makedirs(split_output_path, exist_ok=True)

    for cls in classes:
        class_count = 0  # per class

        class_input_path = os.path.join(split_input_path, cls)
        class_output_path = os.path.join(split_output_path, cls)

        os.makedirs(class_output_path, exist_ok=True)

        for filename in os.listdir(class_input_path):
            img_path = os.path.join(class_input_path, filename)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping unreadable file: {img_path}")
                continue

            resized = cv2.resize(img, img_size)
            cv2.imwrite(os.path.join(class_output_path, filename), resized)

            class_count += 1
            split_count += 1
            total_count += 1

        print(f"  {cls}: {class_count} images resized.")

    print(f"➡ Total in {split}: {split_count} images resized.")

print(f"\n✔ DONE! TOTAL IMAGES RESIZED: {total_count}")
