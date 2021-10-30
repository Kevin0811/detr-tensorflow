import tensorflow as tf
import os
import cv2
import numpy as np

image_size = [224, 224]
keypoints = 21

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 載入模型和權重
custom_model = tf.keras.models.load_model('custom_model_v25_frei.h5')

def read_frei_image(filepath):
    # 讀取並解碼圖片
    image_string = tf.io.read_file(filepath)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_converted = tf.cast(image_decoded, tf.float32)
    # 轉換型別
    image_std = tf.image.per_image_standardization(image_converted)

    return image_std

def show_result(eval_image, model_outputs):

    if "pred_mask" in model_outputs: # 若模型輸出存在遮罩
        pred_mask = model_outputs["pred_mask"] # 取得預測的遮罩
        pred_mask = tf.image.resize(pred_mask, image_size) # 調整為原圖大小
        pred_mask = tf.math.multiply(pred_mask, 255) # 調整數值介於 0~225
        pred_mask = tf.cast(pred_mask, tf.uint8) # 轉換數值類型為 uint8
        pred_mask = pred_mask.numpy() # Tensor to numpy
        #print(pred_mask[0].shape)
        pred_mask_img = cv2.cvtColor(pred_mask[0], cv2.COLOR_GRAY2BGR) # Gray to BGR 用於 opencv imshow 顯示
        cv2.imshow('M', pred_mask_img) # 顯示遮罩圖片
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    r, g, b = cv2.split(eval_image)
    eval_image = cv2.merge([b, g, r])

    # 讀取模型預測位置 (數值介於0~1)
    pred_pos = np.array(model_outputs['pred_pos'])
    # 轉換為像素位置
    pred_skeleton_lable = np.multiply(pred_pos, 224)
    lable = np.reshape(pred_skeleton_lable, (keypoints,2))

    # 將關鍵點放上圖片
    for coords in lable:
        eval_image = cv2.circle(eval_image, (round(coords[0]),round(coords[1])), 2, (255, 0, 0), -1)
        print(round(coords[0]),round(coords[1]))

    # 儲存帶有關鍵點的圖片
    cv2.imwrite(image_save_path, eval_image)
    cv2.imshow('Image', eval_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#path = 'D:\\Dataset\\Body\\K2HPD'
base_path = os.path.join('D:\\', 'Dataset', 'Hand', 'FreiHAND')
image_name = '00050609.jpg'
image_path = os.path.join(base_path, 'rgb', image_name)

image_save_path = os.path.join(os.getcwd(), 'output_image', image_name)

image = read_frei_image(image_path)

image = tf.reshape(image, [1, image_size[0], image_size[1], 3])

#image_tf = tf.convert_to_tensor(image, dtype=tf.float32)

model_outputs = custom_model(image)

#image_u8 = tf.cast(image[0], tf.uint8)
#np_image = np.array(image_u8)

input_image = cv2.imread(image_path)

#print(model_outputs['pred_pos'])
#print(pred_pos)

#skeleton_lable = tf.cast(model_outputs, tf.int32)

#print(pred_skeleton_lable)

show_result(input_image, model_outputs)