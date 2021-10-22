import tensorflow as tf
import os
import cv2
import numpy as np

from detr_tf.networks.detr import get_detr_model
from detr_tf.training_config import TrainingConfig

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = TrainingConfig()

detr = get_detr_model(config, 
                      include_top=False, 
                      nb_class=None, 
                      num_decoder_layers=6, 
                      num_encoder_layers=6, 
                      weights=None)

detr.load_weights("weights/COCO/detr-model_50.ckpt")

def read_k4b_image(filepath):
    # 讀取並解碼圖片
    image_string = tf.io.read_file(filepath)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # 轉換型別
    #image_std = tf.image.per_image_standardization(image_decoded)
    image_converted = tf.cast(image_decoded, tf.float32)
    #print("image shape\n" + str(image_decoded))

    return image_converted

#path = 'D:\\Dataset\\Body\\K2HPD'
path = 'D:\\Dataset\\Hand\\COCO'
image_name = '000012.jpg'
image_path = os.path.join(path, 'image', image_name)

image_save_path = os.path.join(path, 'output', image_name)


image = read_k4b_image(image_path)

image = tf.reshape(image, [1, 128, 128, 3])

model_outputs = detr(image, training=True)

def show_result(eval_image, lable):
    r, g, b = cv2.split(eval_image)
    eval_image = cv2.merge([b, g, r])
    #print(lable.shape)
    #lable = np.reshape(lable,(15,2))
    #print(lable)
    for coords in lable:
        print(coords)
        eval_image = cv2.circle(eval_image, (int(coords[0]),int(coords[1])), 5, (255, 0, 0), -1)
    cv2.imwrite(image_save_path, eval_image)
    cv2.imshow('Image', eval_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_u8 = tf.cast(image[0], tf.uint8)
np_image = np.array(image_u8)

skeleton_lable = tf.cast(model_outputs['pred_pos'][0], tf.int32)
skeleton_lable = np.array(skeleton_lable)

show_result(np_image, skeleton_lable)