import cv2
import os
import imgaug as ia
from imgaug import augmenters as iaa

input_folder = "#"
output_folder = "#"
augmented_images_per_input = 5

# Define the desired size
import imgaug.augmenters as iaa

desired_size = (416, 416)

augmentation = iaa.Sequential([
    iaa.Resize({"height": desired_size[0], "width": desired_size[1]}),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-90, 90)),
    iaa.Affine(rotate=180),
    iaa.Sharpen(alpha=(0, 1.0))  # Sharpen the image
])


def augment_and_save_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        original_image = cv2.imread(image_path)

        if original_image is None:
            print(f"Unable to read {image_file}. Skipping.")
            continue

        for i in range(augmented_images_per_input):
            augmented_images = augmentation(images=[original_image])

            for j, augmented_image in enumerate(augmented_images):
                output_path = os.path.join(output_folder, f"aug_{i}_{j}_{image_file}")
                cv2.imwrite(output_path, augmented_image)

                print(f"Augmented {image_file} and saved as {output_path}")

if __name__ == "__main__":
    augment_and_save_images(input_folder, output_folder)
