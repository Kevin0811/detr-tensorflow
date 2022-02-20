import tensorflow as tf
import os
import cv2
import numpy as np
import sys

from tfswin.ape import AbsoluteEmbedding
from tfswin.basic import BasicLayer
from tfswin.embed import PatchEmbedding
from tfswin.merge import PatchMerging
from tfswin.norm import LayerNorm


image_size = [224, 224]
keypoints = 21
# 要載入的模型
model_name = 'weights\custom_model_v3.6_vtouch.h5'

actions = np.array(['open', 'fist', 'one', 'two', 'three', 'four', 'six','eight', 'nine', 'ok', 'check', 'like', 'middel', 'yo'])

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 載入模型和權重
custom_model = tf.keras.models.load_model(model_name, custom_objects={"TFSwin>PatchEmbedding": PatchEmbedding}, compile=False)

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
    
    # RGB 轉 BGR 用於顯示
    r, g, b = cv2.split(eval_image)
    eval_image = cv2.merge([b, g, r])

    # 顯示手勢辨識結果
    if "pred_gesture" in model_outputs:
        
        # 從模型輸出中提取手勢辨識結果
        pred_gesture = model_outputs["pred_gesture"].numpy()[0].tolist()
        
        # 取最大的值
        max_val = max(pred_gesture)

        
        gesture_index = pred_gesture.index(max_val)
        #print(gesture_index)

        gesture_text = actions[gesture_index]
        # 顯示當前手勢的種類
        cv2.putText(eval_image, gesture_text, (10, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

    # 讀取模型預測位置 (數值介於0~1)
    pred_pos = np.array(model_outputs['pred_pos'])
    # 轉換為像素位置
    pred_skeleton_lable = np.multiply(pred_pos, 224)
    lable = np.reshape(pred_skeleton_lable, (keypoints,2))

    # 將關鍵點放上圖片
    for coords in lable:
        eval_image = cv2.circle(eval_image, (round(coords[1]),round(coords[0])), 2, (255, 0, 0), -1)
        #print(round(coords[0]),round(coords[1]))


    # 儲存帶有關鍵點的圖片
    #cv2.imwrite(image_save_path, eval_image)
    cv2.imshow('Image', eval_image)


# 取得相機
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# 讀取畫面
while cap.isOpened(): # 不要使用 while true

    ret, frame = cap.read()
    # ret 为 True 或 False，代表有没有读到图片
    # frame 為 numpy array，代表讀取到的圖片
    
    if not ret:
        # 沒有畫面
        print("got nothing from capture")
        break

    #print(frame.shape)

    if frame is not None:

        frame = cv2.flip(frame, 1)

        hand_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_img = hand_img[128:352, 208:432]

        #print(hand_img.shape)

        

        tf_hand_img = tf.convert_to_tensor(hand_img, dtype=tf.float32)

        std_hand_img = tf.image.per_image_standardization(tf_hand_img)

        input_img = tf.reshape(std_hand_img, [1, 224, 224, 3])

        model_outputs = custom_model(input_img)

        show_result(hand_img, model_outputs)

        # cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
        cv2.rectangle(frame, (208, 128), (432, 352), (255, 0, 0), 3)
            
        # 顯示圖片
        cv2.imshow('Camera view', frame)
            
        # 等待，若按下 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("frame is None")

# 關閉視窗
cv2.destroyAllWindows()
# 釋放相機
cap.release()