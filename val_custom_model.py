import tensorflow as tf
import os
import cv2
import numpy as np

image_size = [224, 224]
keypoints = 21

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

custom_model = tf.keras.models.load_model('custom_model_v10_frei.h5')

def read_frei_image(filepath):
    # 讀取並解碼圖片
    image_string = tf.io.read_file(filepath)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_converted = tf.cast(image_decoded, tf.float32)
    # 轉換型別
    image_std = tf.image.per_image_standardization(image_converted)
    
    #print("image shape\n" + str(image_decoded))

    return image_std

#path = 'D:\\Dataset\\Body\\K2HPD'
base_path = os.path.join('D:\\', 'Dataset', 'Hand', 'FreiHAND')
image_name = '00000120.jpg'
image_path = os.path.join(base_path, 'rgb', image_name)

image_save_path = os.path.join(os.getcwd(), 'output_image', image_name)

image = read_frei_image(image_path)

image = tf.reshape(image, [1, image_size[0], image_size[1], 3])

#image_tf = tf.convert_to_tensor(image, dtype=tf.float32)

model_outputs = custom_model(image)


def show_result(eval_image, lable):
    r, g, b = cv2.split(eval_image)
    eval_image = cv2.merge([b, g, r])

    lable = np.reshape(lable, (keypoints,2))

    for coords in lable:
        eval_image = cv2.circle(eval_image, (int(coords[0]),int(coords[1])), 2, (255, 0, 0), -1)

    cv2.imwrite(image_save_path, eval_image)
    cv2.imshow('Image', eval_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#image_u8 = tf.cast(image[0], tf.uint8)
#np_image = np.array(image_u8)

input_image = cv2.imread(image_path)

pred_pos = np.array(model_outputs['pred_pos']).reshape(keypoints, 2)

#skeleton_lable = tf.cast(model_outputs, tf.int32)
pred_skeleton_lable = np.multiply(pred_pos, image_size)

print(pred_skeleton_lable)

show_result(input_image, pred_skeleton_lable)