import tensorflow as tf
import time
import numpy as np
import datetime

import tensorboard

from detr_tf.networks.resnet_backbone import ResNet50Backbone
from detr_tf.loss.loss import new_get_losses
from detr_tf.data.tfcsv import load_freihand_dataset

# 相關變數
image_size = [224, 224]
keypoints = 21
batch_size = 32
dataset = 'frei'

learning_rate = 0.00025

training_epoch = 50
print_step = 250

# 調整 GPU 設定
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 輸入圖片宣告
image_input = tf.keras.Input((image_size[0], image_size[1], 3))

# 骨幹
backbone = tf.keras.applications.MobileNetV3Large(input_shape=(image_size[0], image_size[1], 3), 
                                                  alpha=1.0, 
                                                  minimalistic=False, 
                                                  include_top=False,
                                                  weights=None, 
                                                  input_tensor=image_input, 
                                                  pooling=None)

# 前饋神經網路
pos_layer = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(keypoints*2, activation="sigmoid"), # change
            ], name="Position_layer")

# 資料處理流程
featuremap = backbone.output

#print(featuremap)

pos_preds = pos_layer(featuremap)

outputs = {'pred_pos': pos_preds}

custom_model = tf.keras.Model(image_input, outputs, name="custom_model")

# 印出架構
custom_model.summary()

# 優化器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 載入資料集
train_dt = load_freihand_dataset(batch_size, "freihand_train_annos.txt")
valid_dt = load_freihand_dataset(batch_size, "freihand_test_annos.txt")

# 紀錄
#writer = SummaryWriter('logs/k4b-1')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer('logs/mobilenet-frei-' + current_time)

total_train_step = 0
total_val_step = 0
avg_loss = 0
avg_loss_array = []

# 執行驗證
def validation(val_model, val_data, val_step):
    print('\nStart Validation\n')

    val_avg_loss = 0
    val_step = 0

    for step , (images, skeleton_lable, mask) in enumerate(val_data):
        val_step += 1
        # 執行估計
        val_output = val_model(images)
        # 計算損失值(MSE誤差)
        val_loss_value, val_crds_loss ,val_aux_loss, val_gesture_loss, val_shared_loss, val_gesture_acc= new_get_losses(val_output, skeleton_lable, None,  batch_size, keypoints, image_size, mask)

        val_avg_loss += val_loss_value

    # 紀錄損失值 
    with train_summary_writer.as_default():
        tf.summary.scalar('val_loss', val_avg_loss/val_step, total_train_step)

    print('Validation Compelet\n')
    return val_step

# 動態調整學習率
def change_learning_rate(array, lr):
    array = array[-1000:]
    lr_changed = False
    if len(array)==1000 and lr > 1e-6:
        if np.mean(array[:100])-np.mean(array[-100:]) < 0.8:
            lr = lr * 0.5
            lr_changed = True
    return lr, lr_changed

# 驗證
#total_val_step = validation(custom_model, valid_dt, total_val_step)

tf.keras.backend.set_learning_phase(True)

# 進行訓練
for epoch_nb in range(training_epoch):
    print("\nStart of Epoch %d\n" % (epoch_nb,))

    # Training
    total_loss = 0
    time_counter = time.time()

    # 調整 learning_rate
    if epoch_nb%10 == 0 and epoch_nb != 0 and learning_rate > 5e-5:
        learning_rate = learning_rate*0.5
        optimizer.learning_rate.assign(learning_rate)

    for step , (images, skeleton_lable, mask) in enumerate(train_dt):
        total_train_step += 1

        with tf.GradientTape() as tape:
            # 估計
            model_output = custom_model(images)
            # 計算損失值
            loss_value, crds_loss ,aux_loss, gesture_loss, shared_loss, gesture_acc = new_get_losses(model_output, skeleton_lable, None, batch_size, keypoints, image_size, mask)

            total_loss += loss_value
            
            with train_summary_writer.as_default():
                # 紀錄損失值
                tf.summary.scalar('loss', loss_value, total_train_step)
                # 紀錄學習率
                tf.summary.scalar('learning rate', optimizer.learning_rate, total_train_step)

        # 計算梯度
        grads = tape.gradient(loss_value, custom_model.trainable_weights)
        # 更新權重
        optimizer.apply_gradients(zip(grads, custom_model.trainable_weights))
        

        if step % print_step == 0 and step != 0:
            # 計算執行時間
            elapsed = time.time() - time_counter
            # 計算 將此 step 區間的平均損失值
            avg_loss = total_loss/print_step
            # 將平均損失值紀錄到 avg_loss_array
            avg_loss_array.append(avg_loss)

            # 紀錄平均損失值
            with train_summary_writer.as_default():
                tf.summary.scalar('avg_loss', avg_loss, total_train_step)
            # 強制從暫存中寫入
            train_summary_writer.flush()

            # 印出資料
            print(f"Epoch: [{epoch_nb}], Step: [{step}], time : [{elapsed:.2f}], loss : [{avg_loss:.2f}]")
            
            time_counter = time.time()
            total_loss = 0

    # 印出學習率
    print("Learning rate: " + str(optimizer.learning_rate))
    # 驗證
    total_val_step = validation(custom_model, valid_dt, total_val_step)

    # 儲存模型和權重
    #tf.saved_model.save(custom_model, '/weights/custom_model_' + dataset + '.h5')
    custom_model.save('weights/custom_model_mobilenet_v10_' + dataset + '.h5')
    custom_model.save_weights('weights/'+ dataset +'/custom-model_mobilenet_v10_' + current_time + ".ckpt")

print('Traingin Compelet')