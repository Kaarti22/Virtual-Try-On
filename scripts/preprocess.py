import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
train_human_images_path = '../data/train/humans/'
train_clothes_path = '../data/train/clothes/'
test_human_images_path = '../data/test/humans/'
test_clothes_path = '../data/test/clothes/'

# Define output paths
output_train_human_images_path = '../data/train/preprocessed_humans/'
output_train_clothes_path = '../data/train/preprocessed_clothes/'
output_test_human_images_path = '../data/test/preprocessed_humans/'
output_test_clothes_path = '../data/test/preprocessed_clothes/'

# Create output directories if they don't exist
os.makedirs(output_train_human_images_path, exist_ok=True)
os.makedirs(output_train_clothes_path, exist_ok=True)
os.makedirs(output_test_human_images_path, exist_ok=True)
os.makedirs(output_test_clothes_path, exist_ok=True)

def preprocess_images(input_dir, output_dir, img_size=(256, 192)):
    for img_file in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size) # resize the image
        img = img.astype('float32') / 255.0 # normalize image to [0, 1]
        cv2.imwrite(os.path.join(output_dir, img_file), img * 255) # save image in uint8 format
        
# preprocess training images
preprocess_images(train_human_images_path, output_train_human_images_path)
preprocess_images(train_clothes_path, output_train_clothes_path)

# preprocess test images
preprocess_images(test_human_images_path, output_test_human_images_path)
preprocess_images(test_clothes_path, output_test_clothes_path)

print("Preprocessing completed.")