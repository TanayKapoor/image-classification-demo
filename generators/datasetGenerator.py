import os
import shutil
import random

source_data_dir = 'raw_data'

data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

classes = os.listdir(source_data_dir)

train_ratio = 0.8

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_images(class_name, image_list, train_dir, val_dir):
    num_images = len(image_list)
    num_train = int(num_images * train_ratio)
    
    random.shuffle(image_list)
    
    train_images = image_list[:num_train]
    val_images = image_list[num_train:]
    
    for image_name in train_images:
        src_path = os.path.join(source_data_dir, class_name, image_name)
        dest_path = os.path.join(train_dir, class_name, image_name)
        shutil.copy2(src_path, dest_path)
    
    for image_name in val_images:
        src_path = os.path.join(source_data_dir, class_name, image_name)
        dest_path = os.path.join(val_dir, class_name, image_name)
        shutil.copy2(src_path, dest_path)

for class_name in classes:
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)
    create_dir(class_train_dir)
    create_dir(class_val_dir)
    
    image_list = os.listdir(os.path.join(source_data_dir, class_name))
    
    copy_images(class_name, image_list, train_dir, val_dir)

print("Data organization completed.")
