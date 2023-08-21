import os
import cv2
import argparse
import numpy as np
from models import build_stixel_net
from albumentations import (
    Compose,
    Resize,
    Normalize,
)
import tensorflow.keras.backend as K

directory = "out"

# Check if the directory exists
if not os.path.exists(directory):
    # If it doesn't exist, create it
    os.makedirs(directory)


def test_single_image(model, img, label_size=(100, 50)):
    assert img is not None

    h, w, c = img.shape
    val_aug = Compose([Resize(370, 800), Normalize(p=1.0)])
    aug_img = val_aug(image=img)["image"]
    aug_img = aug_img[np.newaxis, :]
    predict = model.predict(aug_img, batch_size=1)
    predict = K.reshape(predict, label_size)
    predict = K.eval(K.argmax(predict, axis=-1))

    for x, py in enumerate(predict):
        x0 = int(x * w / 100)
        x1 = int((x + 1) * w / 100)
        y = int((py + 0.5) * h / 50)
        cv2.rectangle(img, (x0, 0), (x1, y), (0, 0, 255), 1)

    return img


def main(args):
    assert os.path.isfile(args.model_path)
    from config import Config

    dt_config = Config()
    model = build_stixel_net()
    model.load_weights(args.model_path)

    # Directory containing the images
    img_folder = "im"

    # Iterate through all .jpg files in the folder
    for i, img_filename in enumerate(os.listdir(img_folder)):
        if img_filename.endswith(".jpg"):
            img_path = os.path.join(img_folder, img_filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            # img = img.astype(np.float32) / 255.0  # Normalize the pixel values to [0, 1]

            result = test_single_image(model, img)
            cv2.imwrite(f"out/result{i}.png", result.astype(np.uint8))  # Save the result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saved_models/model-033.h5")
    parsed_args = parser.parse_args()
    main(parsed_args)
