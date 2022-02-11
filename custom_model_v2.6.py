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
batch_size = 6
dataset = 'frei'

training_epoch = 100
print_step = 500

# 設定 GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 輸入圖片宣告
image_input = tf.keras.Input((image_size[0], image_size[1], 3))

# 骨幹
#backbone = ResNet50Backbone(name='Backbone_ResNet50')
resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, 
                                                 weights=None, 
                                                 input_tensor=image_input,
                                                 input_shape=(image_size[0], image_size[1], 3), 
                                                 pooling=None)
backbone = tf.keras.models.Sequential([
            resnet
            ], name="Backbone_layer")

# 前饋神經網路 + dropout
pos_layer = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(keypoints*2, activation="sigmoid"), # change
            ], name="Position_layer")

# [new] 調整過的上採樣層，直接輸出原圖大小(224*224)
mask_layer = tf.keras.models.Sequential([
             tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, activation="relu"),
             tf.keras.layers.UpSampling2D(size=(2, 2)),
             tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, activation="relu"),
             tf.keras.layers.UpSampling2D(size=(3, 3)),
             tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu"),
             tf.keras.layers.UpSampling2D(size=(3, 3)),
             tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, activation="sigmoid"),
             ], name="Mask_layer")
# 資料處理流程
featuremap = backbone(image_input)

pos_preds = pos_layer(featuremap)

mask_preds = mask_layer(featuremap)

outputs = {
    'pred_pos': pos_preds,
    'pred_mask': mask_preds
}

custom_model = tf.keras.Model(image_input, outputs, name="custom_model")

# 印出架構
custom_model.summary()

backbone_learning_rate = 0.00025
mask_learning_rate = 0.0001
pos_learning_rate = 0.00025

# 優化器
backbone_optimizer = tf.keras.optimizers.Adam(learning_rate=backbone_learning_rate)
mask_optimizer = tf.keras.optimizers.Adam(learning_rate=mask_learning_rate)
pos_optimizer = tf.keras.optimizers.Adam(learning_rate=pos_learning_rate)

# 載入資料集
train_dt = load_freihand_dataset(batch_size, "freihand_train_annos.txt")
valid_dt = load_freihand_dataset(batch_size, "freihand_test_annos.txt")

# 紀錄
#writer = SummaryWriter('logs/k4b-1')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer('logs/resnet-mask-frei-' + current_time)

total_train_step = 0
total_val_step = 0
avg_loss = 0
avg_loss_array = []

# 執行驗證
def validation(val_model, val_data, val_step):
    print('\nStart Validation\n')

    val_avg_loss = 0

    for step , (images, skeleton_lable, mask) in enumerate(val_data):
        val_step += 1
        # 執行估計
        val_output = val_model(images)
        # 計算損失值(MSE誤差)
        val_loss_value, val_coords_loss ,val_aux_loss = new_get_losses(val_output, skeleton_lable, batch_size, keypoints, image_size, mask)

        val_avg_loss += val_loss_value

    # 紀錄損失值 
    with train_summary_writer.as_default():
        tf.summary.scalar('val_loss', val_avg_loss/val_step, total_train_step)
            
    print('Validation Compelet\n')
    return val_step

# 調整學習率(未使用)
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

# 進行訓練
for epoch_nb in range(training_epoch):
    print("\nStart of Epoch %d\n" % (epoch_nb,))

    # Training
    total_loss = 0
    time_counter = time.time()

    # Assing learning_rate
    if epoch_nb%15 == 0 and epoch_nb != 0:

        backbone_learning_rate = backbone_optimizer.learning_rate*0.5
        mask_learning_rate = mask_optimizer.learning_rate*0.5
        pos_learning_rate = pos_optimizer.learning_rate*0.5

        if backbone_learning_rate > 5e-5:
            backbone_optimizer.learning_rate.assign(backbone_learning_rate)
        if mask_learning_rate > 5e-5:
            mask_optimizer.learning_rate.assign(mask_learning_rate)
        if pos_learning_rate > 5e-5:
            pos_optimizer.learning_rate.assign(pos_learning_rate)

    for step , (images, skeleton_lable, mask) in enumerate(train_dt):
        total_train_step += 1

        loss_value = 0
        aux_loss = 0
        coords_loss = 0

        with tf.GradientTape(persistent=True) as tape:
            # 估計
            model_output = custom_model(images)
            # 計算損失值
            loss_value, coords_loss ,aux_loss = new_get_losses(model_output, skeleton_lable, batch_size, keypoints, image_size, mask)

            total_loss += loss_value
            
            with train_summary_writer.as_default():
                # 紀錄損失值
                tf.summary.scalar('loss', loss_value, total_train_step)
                tf.summary.scalar('coords_loss', coords_loss, total_train_step)
                tf.summary.scalar('aux_loss', aux_loss, total_train_step)
                # 紀錄學習率
                tf.summary.scalar('learning rate', backbone_optimizer.learning_rate, total_train_step)

        # 取得權重
        backbone_weights = custom_model.get_layer("Backbone_layer").trainable_variables
        mask_weights = custom_model.get_layer("Mask_layer").trainable_variables
        pos_weights = custom_model.get_layer("Position_layer").trainable_variables

        # 計算梯度
        backbone_grads = tape.gradient(loss_value, backbone_weights)
        mask_grads = tape.gradient(aux_loss, mask_weights)
        pos_grads = tape.gradient(coords_loss, pos_weights)

        del tape

        # 更新權重
        backbone_optimizer.apply_gradients(zip(backbone_grads, backbone_weights))
        mask_optimizer.apply_gradients(zip(mask_grads, mask_weights))
        pos_optimizer.apply_gradients(zip(pos_grads, pos_weights))
        

        if step % print_step == 0 and step != 0:
            # 計算執行時間
            elapsed = time.time() - time_counter
            # 計算 將此 step 區間的平均損失值
            avg_loss = total_loss/print_step
            # 將平均損失值紀錄到 avg_loss_array
            avg_loss_array.append(avg_loss)

            # 調整 learning rate
            #learning_rate, lr_changed = change_learning_rate(avg_loss_array, learning_rate)
            # 將調整後的 learning rate 加入 優化器
            #if lr_changed:
                #optimizer.learning_rate.assign(learning_rate)
                #avg_loss_array = []

            # 紀錄平均損失值
            with train_summary_writer.as_default():
                tf.summary.scalar('avg_loss', avg_loss, total_train_step)
            # 印出資料
            print(f"Epoch: [{epoch_nb}], Step: [{step}], time : [{elapsed:.2f}], loss : [{avg_loss:.2f}]")
            
            time_counter = time.time()
            total_loss = 0

    # 打印學習率
    #print("Learning rate: " + str(optimizer.learning_rate))
    # 驗證
    total_val_step = validation(custom_model, valid_dt, total_val_step)

    # 儲存模型和權重
    #tf.saved_model.save(custom_model, '/weights/custom_model_' + dataset + '.h5')
    custom_model.save('weights/custom_model_v26_' + dataset + '.h5')
    custom_model.save_weights('weights/'+ dataset +'/custom-model_' + current_time + ".ckpt")

print('Traingin Compelet')