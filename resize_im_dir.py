import os
import cv2


def resize_images(folder_path, width, height):
    for img_filename in os.listdir(folder_path):
        if img_filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, img_filename)
            img = cv2.imread(img_path)

            # Resize the image
            resized_img = cv2.resize(img, (width, height))

            # Save the resized image, overwriting the original file
            cv2.imwrite(img_path, resized_img)
            print(f"Resized and saved: {img_path}")


if __name__ == "__main__":
    folder_path = "im"
    width = 800
    height = 370

    resize_images(folder_path, width, height)
