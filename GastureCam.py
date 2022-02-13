import tensorflow as tf
import os
import cv2
import numpy as np

image_size = [224, 224]
keypoints = 21
# 要載入的模型
model_name = 'custom_model_v25_frei.h5'

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 載入模型和權重
custom_model = tf.keras.models.load_model(model_name)

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
    #cv2.imwrite(image_save_path, eval_image)
    cv2.imshow('Image', eval_image)


# 取得相機
cap = cv2.VideoCapture(0)

cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)


# 讀取畫面
while cap.isOpened(): # 不要使用 while true

    ret, frame = cap.read()
    # ret 为 True 或 False，代表有没有读到图片
    # frame 為 numpy array，代表讀取到的圖片
    
    if not ret:
        # 沒有畫面
        continue

    hand_img = frame[128:352, 208:432]

    # cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
    cv2.rectangle(frame, (128, 208), (352, 432), (255, 0, 0), 3)

    # 標準化
    mean_hand_img = np.mean(hand_img, axis=(1,2), keepdims=True)
    std_hand_img = np.sqrt(((hand_img - mean_hand_img)**2).mean((1,2), keepdims=True))

    tf_hand_img = tf.convert_to_tensor(std_hand_img, dtype=tf.float32)

    model_outputs = custom_model(tf_hand_img)

    show_result(hand_img, model_outputs)
        
    # 顯示圖片
    cv2.imshow('Camera view', frame)
        
    # 等待，若按下 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 關閉視窗
cv2.destroyAllWindows()
# 釋放相機
cap.release()