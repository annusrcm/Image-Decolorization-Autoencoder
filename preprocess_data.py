import cv2
import os
from datetime import datetime

from logger import Logger
from config import Config

config = Config()


def preprocess_training_data(path, color_path, gray_path):
    count = 1
    for root, sub_dir, files in os.walk(path):
        for filename in files:
            try:
                print("Processing : {}".format(filename))
                if str(filename).endswith(".jpg"):
                    image = cv2.imread(os.path.join(root, filename))
                    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    image = cv2.resize(image, (128, 128))
                    gray_img = cv2.resize(gray_img, (128, 128))

                    gray_img_name = "gray_" + str(count) + ".jpg"
                    color_img_name = "color_" + str(count) + ".jpg"

                    cv2.imwrite(os.path.join(gray_path, gray_img_name), gray_img)
                    cv2.imwrite(os.path.join(color_path, color_img_name), image)

                    count += 1
            except Exception as e:
                Logger.log(e)
                Logger.log("Could not preprocess : {}".format(os.path.join(root,filename)))
    Logger.log("{} training images preprocessed".format(count))


def preprocess_testing_data(test_data_path, test_gray_path):
    count = 1
    for file in os.listdir(test_data_path):
        try:
            print("Processing : {}".format(file))
            img = cv2.imread(os.path.join(test_data_path, file), 0)
            img = cv2.resize(img, (128, 128))
            gray_name = "gray_" + str(count) + ".jpg"
            cv2.imwrite(os.path.join(test_gray_path, gray_name), img)
            count += 1
        except Exception as e:
            Logger.log(e)
            Logger.log("Could not preprocess : {}".format(os.path.join(test_data_path, file)))
    Logger.log("{} testing images preprocessed".format(count))


if __name__ == "__main__":
    start = datetime.now()
    preprocess_training_data(config.path, config.color_path, config.gray_path)
    preprocess_testing_data(config.test_data_path, config.test_gray_path)
    Logger.log("Prepossessing took : {} seconds".format(datetime.now() - start))
