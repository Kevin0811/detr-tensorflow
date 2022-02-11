import tensorflow as tf
from random import shuffle
import pandas as pd
import numpy as np
import imageio
import os
from os.path import expanduser
import glob # for reading files

from .import processing
from .transformation import detr_transform
from .. import bbox

""" Load the hardhat dataset ↓ """

def load_data_from_index(index, class_names, filenames, anns, config, augmentation, img_dir):
    # Open the image
    image = imageio.imread(os.path.join(expanduser("~"), "Desktop", "detr-tensorflow","detr_tf", "data","hardhat", img_dir, filenames[index]))
    # Select all the annotatiom (bbox and class) on this image
    image_anns = anns[anns["filename"] == filenames[index]]    
    
    # Convert all string class to number (the target class)
    t_class = image_anns["class"].map(lambda x: class_names.index(x)).to_numpy()
    # Select the width&height of each image (should be the same since all the ann belongs to the same image)
    width = image_anns["width"].to_numpy()
    height = image_anns["height"].to_numpy()
    # Select the xmin, ymin, xmax and ymax of each bbox, Then, normalized the bbox to be between and 0 and 1
    # Finally, convert the bbox from xmin,ymin,xmax,ymax to x_center,y_center,width,height
    bbox_list = image_anns[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    bbox_list = bbox_list / [width[0], height[0], width[0], height[0]] 
    t_bbox = bbox.xy_min_xy_max_to_xcycwh(bbox_list)
    
    # Transform and augment image with bbox and class if needed
    image, t_bbox, t_class = detr_transform(image, t_bbox, t_class, config, augmentation=augmentation)

    # Normalized image
    image = processing.normalized_images(image, config)

    return image.astype(np.float32), t_bbox.astype(np.float32), np.expand_dims(t_class, axis=-1).astype(np.int64)

def load_tfcsv_dataset(config, batch_size, augmentation=False, exclude=[], ann_dir=None, ann_file=None, img_dir=None):
    
    ann_dir = config.data.ann_dir if ann_dir is None else ann_dir
    ann_file = config.data.ann_file if ann_file is None else ann_file
    img_dir = config.data.img_dir if img_dir is None else img_dir

    anns = pd.read_csv(os.path.join(expanduser("~"), "Desktop", "detr-tensorflow","detr_tf", "data","hardhat", img_dir, ann_file))
    for name  in exclude:
        anns = anns[anns["class"] != name]

    unique_class = anns["class"].unique()
    unique_class.sort()
    

    # Set the background class to 0
    config.background_class = 0
    class_names = ["background"] + unique_class.tolist()


    filenames = anns["filename"].unique().tolist()
    indexes = list(range(0, len(filenames)))
    shuffle(indexes)

    dataset = tf.data.Dataset.from_tensor_slices(indexes)
    dataset = dataset.map(lambda idx: processing.numpy_fc(
        idx, load_data_from_index, 
        class_names=class_names, filenames=filenames, anns=anns, config=config, augmentation=augmentation, img_dir=img_dir)
    ,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    

    # Filter labels to be sure to keep only sample with at least one bbox
    dataset = dataset.filter(lambda imgs, tbbox, tclass: tf.shape(tbbox)[0] > 0)
    # Pad bbox and labels
    dataset = dataset.map(processing.pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset, class_names

""" Load the hardhat dataset ↑ """

""" Load the k4b dataset ↓ """

def read_image(filename, label):
    # 讀取並解碼圖片
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    # 轉換型別
    #image_std = tf.image.per_image_standardization(image_decoded)
    image_converted = tf.cast(image_decoded, tf.float16)
    #print("image shape\n" + str(image_decoded))

    return image_converted, label

def load_text_file(file_name):
    # current_path = os.getcwd()
    path = 'D:\\Dataset\\Body\\K2HPD'
    file_path = os.path.join(path, file_name)
    f = open(file_path, 'r') # 開啟並讀取檔案
    lines = f.readlines() # 讀取檔案內容的每一行文字為陣列
    image_list = list()
    skeleton_list = list()

    for line in lines:
        line = line.split()
        image_name = line[0]
        image_path = os.path.join(path, 'image', image_name)
        image_list.append(image_path)
        skeleton_xy = line[1]
        skeleton_xy = skeleton_xy.split(',')
        skeletons = list()
        for coor in range(0, 29, 2):
            skeletons.append([float(skeleton_xy[coor]), float(skeleton_xy[coor+1])])
        #print(skeletons)
        skeleton_list.append(skeletons)
    f.close() # 關閉檔案
    #skeleton_list = tf.cast(skeleton_list, tf.float16)
    return image_list, skeleton_list

def load_k4b_dataset(batch_size, file_name):
    image_list, skeleton_list = load_text_file(file_name)
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(image_list),tf.constant(skeleton_list)))
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=False)
    # Batch images
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

""" Load the k4b dataset ↑ """

""" Load the coco dataset ↓ """

def read_coco_image(filename, label):
    # 讀取並解碼圖片
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # 轉換型別
    image_std = tf.image.per_image_standardization(image_decoded)
    image_converted = tf.cast(image_std, tf.float16)
    #print("image shape\n" + str(image_decoded))

    return image_converted, label

def load_coco_text(file_name):
    # current_path = os.getcwd()
    path = 'D:\\Dataset\\Hand\\COCO'
    file_path = os.path.join(path, file_name)
    f = open(file_path, 'r') # 開啟並讀取檔案
    lines = f.readlines() # 讀取檔案內容的每一行文字為陣列
    image_list = list()
    skeleton_list = list()

    for line in lines:
        line = line.split()
        image_name = line[0]
        image_path = os.path.join(path, 'image', image_name)
        image_list.append(image_path)
        skeleton_xy = line[1]
        skeleton_xy = skeleton_xy.split(',')
        skeletons = list()
        for coor in range(0, 41, 2): # skeleton: 42
            skeletons.append([float(skeleton_xy[coor]),float(skeleton_xy[coor+1])])
        skeleton_list.append(skeletons)
    f.close() # 關閉檔案
    return image_list, skeleton_list

""" Load the coco dataset ↑ """


""" Load the freihand dataset ↓ """

def read_freihand_image(file_path, label, mask_path):
    # 讀取並解碼圖片
    image_string = tf.io.read_file(file_path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_converted = tf.cast(image_decoded, tf.float32)
    # 轉換型別
    image_std = tf.image.per_image_standardization(image_converted)
    
    #print("image shape\n" + str(image_decoded))

    mask_string = tf.io.read_file(mask_path)
    mask_decoded = tf.image.decode_jpeg(mask_string, channels=3)
    mask_gray = tf.image.rgb_to_grayscale(mask_decoded)
    mask_converted = tf.cast(mask_gray, tf.float32)
    #mask_std = tf.image.per_image_standardization(mask_converted)
    
    return image_std, label, mask_converted

def load_freihand_text(file_name):
    # current_path = os.getcwd()
    base_path = os.path.join('D:\\', 'Dataset', 'Hand', 'FreiHAND')
    text_path = os.path.join(base_path, file_name)

    f = open(text_path, 'r') # 開啟並讀取檔案
    lines = f.readlines() # 讀取檔案內容的每一行文字為陣列
    image_list = list()
    mask_list = list()
    skeleton_list = list()

    for line in lines:
        line = line.split(' ')
        image_name = line[0]
        image_path = os.path.join(base_path, 'rgb', image_name)
        image_list.append(image_path)

        mask_path = os.path.join(base_path, 'mask', image_name)
        mask_list.append(mask_path)

        skeleton_xy = line[1]
        skeleton_xy = skeleton_xy.split(',')
        skeletons = list()
        try:
            for coord in range(0, 41, 2): # skeleton: 42
                skeletons.append([float(skeleton_xy[coord]),float(skeleton_xy[coord+1])])
            skeleton_list.append(skeletons)
        except:
            print(image_name)
    f.close() # 關閉檔案
    return image_list, skeleton_list, mask_list

def load_freihand_dataset(batch_size, file_name):
    image_list, skeleton_list, mask_list = load_freihand_text(file_name)
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(image_list),tf.constant(skeleton_list), tf.constant(mask_list)))
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=False)
    dataset = dataset.map(read_freihand_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

""" Load the freihand dataset ↑ """

""" Load the vTouch dataset ↓ """

def read_vtouch_image(file_path, skeleton_label, mask_path, gesture_label):

    """ 讀取彩色圖片 """
    # 讀取並解碼圖片
    image_string = tf.io.read_file(file_path) # 讀取檔案
    image_decoded = tf.image.decode_jpeg(image_string, channels=3) # 解碼圖片
    image_converted = tf.cast(image_decoded, tf.float32) # int 轉 float
    image_std = tf.image.per_image_standardization(image_converted) # 標準化
    
    #print("image shape\n" + str(image_decoded))

    """ 讀取黑白圖片 """
    mask_string = tf.io.read_file(mask_path)
    mask_decoded = tf.image.decode_jpeg(mask_string, channels=1) # 注意維度
    #mask_gray = tf.image.rgb_to_grayscale(mask_decoded)
    mask_converted = tf.cast(mask_decoded, tf.float32) # int 轉成 float
    #mask_std = tf.image.per_image_standardization(mask_converted)
    
    return image_std, skeleton_label, mask_converted, gesture_label

def load_vtouch_text():

    data_folder = "D:\\vTouch Gesture\\data\\"

    image_list = list()
    mask_list = list()
    skeleton_list = list()
    gesture_list = list()

    gesture_num = 0

    data_nums = 0

    for gesture in os.listdir(data_folder):
        hand_data_path = data_folder + gesture + '\\*[0-9].jpg'

        hand_data_path = glob.iglob(hand_data_path.encode('unicode_escape'))

        for path in hand_data_path:
            item_path = path.decode()

            image_list.append(item_path)

            mask_list.append(item_path[:-4] + '_dpt.jpg')

            hand_kp = np.load(item_path[:-4] + '.npy')
            skeletons = list()
            for i in range(21): # skeleton: 42
                skeletons.append([float(hand_kp[i][0]),float(hand_kp[i][1])])
            skeleton_list.append(skeletons)

            gesture_list.append(gesture_num)

            data_nums += 1

        gesture_num += 1

    print("Total Data Nums:", data_nums)

    return image_list, skeleton_list, mask_list, gesture_list

def load_vtouch_dataset(batch_size):
    image_list, skeleton_list, mask_list, gesture_list = load_vtouch_text()
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(image_list),tf.constant(skeleton_list), tf.constant(mask_list), tf.constant(gesture_list)))
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=False)
    dataset = dataset.map(read_vtouch_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

""" Load the freihand dataset ↑ """