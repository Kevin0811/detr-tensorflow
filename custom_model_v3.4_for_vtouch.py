import tensorflow as tf
import time
import numpy as np
import datetime

import tensorboard

from detr_tf.networks.resnet_backbone import ResNet50Backbone
from detr_tf.loss.loss import new_get_losses
from detr_tf.data.tfcsv import load_vtouch_dataset

# 相關變數
image_size = [224, 224]
keypoints = 21
batch_size = 6
dataset = 'vtouch'
version = 'v3.4'
waiting4header = False
load_pretrained = False
pretrained_model_name = 'weights\custom_model_v2.5_frei.h5'

training_epoch = 100
print_step = 200
gesture_cnt = 14

# 設定 GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# 圖片輸入層
image_input = tf.keras.Input((image_size[0], image_size[1], 3))

# 骨幹層
if load_pretrained: # 讀取預訓練權重 [new in v3.4]
    print("Loading pretrained weights...")
    pretrained_model = tf.keras.models.load_model(pretrained_model_name, compile=False)
    backbone = pretrained_model.get_layer('Backbone_layer')
    
    # 鎖定權重
    backbone.trainable = False

    waiting4header = True
else:
    #backbone = ResNet50Backbone(name='Backbone_ResNet50')
    resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, 
                                                     weights=None, # 不要載入imagenet 預訓練權重(效果很差)
                                                     input_tensor=image_input,
                                                     input_shape=(image_size[0], image_size[1], 3), 
                                                     pooling=None)
    backbone = tf.keras.models.Sequential([
                resnet
                ], name="Backbone_layer")

# 回歸與分類共用層
# [new in v3.1] 將回歸網路和分類網路的上層部分共用
# [new in v3.2] 增加共用層的比例
shared_layer = tf.keras.models.Sequential([
               tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu'),
               tf.keras.layers.MaxPool2D((2,2)),
               tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(512, activation="relu"),
               tf.keras.layers.Dropout(0.2),
               tf.keras.layers.Dense(265, activation="relu"),
               ], name="Shared_layer")

# 前饋神經網路 + dropout (用於回歸手部關鍵點)
pos_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation="relu"), # [new in v3.2]
            tf.keras.layers.Dense(keypoints*2, activation="sigmoid"), # change
            ], name="Position_layer")

# 前饋神經網路 + dropout (用於分類手勢)
gesture_layer = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(gesture_cnt, activation="softmax"), # change
                ], name="Gesture_layer")

# 上採樣 + transpose (用於語意分割)
# [new in v2.6] 調整過的上採樣層，直接輸出原圖大小(224*224)
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

mask_preds = mask_layer(featuremap)

shared_outputs = shared_layer(featuremap)
pos_preds = pos_layer(shared_outputs)
gesture_preds = gesture_layer(shared_outputs)

# 輸出資料結構
outputs = {
    'pred_pos': pos_preds,
    'pred_mask': mask_preds,
    'pred_gesture': gesture_preds
}

custom_model = tf.keras.Model(image_input, outputs, name="custom_model")

# 印出架構
custom_model.summary()

backbone_initial_lr = 0.00025
mask_initial_lr = 0.0001
shared_initial_lr = 0.00002
pos_initial_lr = 0.00025
gesture_initial_lr = 0.00002

backbone_end_lr =  5e-5
mask_end_lr = 1e-5
shared_end_lr = 1e-6
pos_end_lr = 5e-5
gesture_end_lr = 1e-6

# 優化器
backbone_optimizer = tf.keras.optimizers.Adam(learning_rate=backbone_initial_lr)
mask_optimizer = tf.keras.optimizers.Adam(learning_rate=mask_initial_lr)
shared_optimizer = tf.keras.optimizers.Adam(learning_rate=shared_initial_lr)
pos_optimizer = tf.keras.optimizers.Adam(learning_rate=pos_initial_lr)
gesture_optimizer = tf.keras.optimizers.Adam(learning_rate=gesture_initial_lr)

# 載入資料集
vtouch_hand_dataset = load_vtouch_dataset(batch_size)

total_data_batch = tf.data.experimental.cardinality(vtouch_hand_dataset).numpy()

# 切分資料集
train_dt = vtouch_hand_dataset.skip(int(total_data_batch/10))
valid_dt = vtouch_hand_dataset.take(int(total_data_batch/10))

print("Train Dataset Length:", tf.data.experimental.cardinality(train_dt).numpy())
print("Valid Dataset Length:", tf.data.experimental.cardinality(valid_dt).numpy())

# 紀錄
#writer = SummaryWriter('logs/k4b-1')
current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
#train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer('logs/v3/resnet+mask+gesture+shared-' + version + '-' + dataset + '-' + current_time)

#with train_summary_writer.as_default():
#    tf.summary.graph(custom_model.get_concrete_model().graph)

total_train_step = 0
total_val_step = 0
avg_loss = 0
#avg_loss_array = []

# 執行驗證
def validation(val_model, val_data, val_step):
    print('\n>>> Start Validation', end=' ')

    val_avg_loss = 0
    val_avg_crds_loss = 0
    val_avg_aux_loss = 0
    val_avg_gesture_loss = 0
    val_avg_shared_loss = 0
    val_avg_gesture_acc = 0

    data_len = len(val_data)

    for step , (images, skeleton_lable, mask, gesture_label) in enumerate(val_data):
        val_step += 1
        # 執行估計
        val_output = val_model(images)
        # 計算損失值(MSE誤差)
        val_loss_value, val_crds_loss ,val_aux_loss, val_gesture_loss, val_shared_loss, val_gesture_acc = new_get_losses(val_output, skeleton_lable,gesture_label,  batch_size, keypoints, image_size, mask)

        val_avg_loss += val_loss_value
        val_avg_crds_loss += val_crds_loss
        val_avg_aux_loss += val_aux_loss
        val_avg_gesture_loss += val_gesture_loss
        val_avg_shared_loss += val_shared_loss
        val_avg_gesture_acc += val_gesture_acc

    val_avg_loss = val_avg_loss/data_len
    val_avg_crds_loss = val_avg_crds_loss/data_len
    val_avg_aux_loss = val_avg_aux_loss/data_len
    val_avg_gesture_loss = val_avg_gesture_loss/data_len
    val_avg_shared_loss = val_avg_shared_loss/data_len
    val_avg_gesture_acc = val_avg_gesture_acc/data_len

    # 紀錄損失值 
    with train_summary_writer.as_default():
        tf.summary.scalar('val_avg_loss', val_avg_loss, total_train_step)
        tf.summary.scalar('val_avg_crds_loss', val_avg_crds_loss, total_train_step)
        tf.summary.scalar('val_avg_aux_loss', val_avg_aux_loss, total_train_step)
        tf.summary.scalar('val_avg_gesture_loss', val_avg_gesture_loss, total_train_step)
        tf.summary.scalar('val_avg_shared_loss', val_avg_shared_loss, total_train_step)
        tf.summary.scalar('val_avg_gesture_acc', val_avg_gesture_acc, total_train_step)
    
    print('\r>>> Validation Compeleted')
    print(f"Results: average loss : [{val_avg_loss:.3f}], crd loss : [{val_avg_crds_loss:.3f}], aux loss : [{val_avg_aux_loss:.3f}], gesture loss : [{val_avg_gesture_loss:.3f}]\n")    
    return val_step

# 調整學習率(PolynomialDecay)
def decayed_learning_rate(step, initial_learning_rate, end_learning_rate, decay_steps, power=5):
    step = min(step, decay_steps)
    return ((initial_learning_rate - end_learning_rate)*(1 - step / decay_steps) ** (power)) + end_learning_rate

# 驗證
#total_val_step = validation(custom_model, valid_dt, total_val_step)

# 進行訓練
for epoch_nb in range(training_epoch):
    print("\n>>> Start of Epoch %d\n" % (epoch_nb,))

    # Training
    total_loss = 0
    loss_value = 0
    time_counter = time.time()

    # Assing learning_rate 調整學習率
    backbone_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, backbone_initial_lr, backbone_end_lr, training_epoch))
    mask_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, mask_initial_lr, mask_end_lr, training_epoch))
    pos_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, pos_initial_lr, pos_end_lr, training_epoch))
    gesture_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, gesture_initial_lr, gesture_end_lr, training_epoch))
    shared_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, shared_initial_lr, shared_end_lr, training_epoch))


    # 紀錄學習率
    tf.summary.scalar('Backbone learning rate', backbone_optimizer.learning_rate, total_train_step)
    tf.summary.scalar('Mask learning rate', mask_optimizer.learning_rate, total_train_step)
    tf.summary.scalar('Position learning rate', pos_optimizer.learning_rate, total_train_step)
    tf.summary.scalar('Gesture learning rate', gesture_optimizer.learning_rate, total_train_step)
    tf.summary.scalar('Shared learning rate', shared_optimizer.learning_rate, total_train_step)


    # 1 step = <batch size> images
    for step , (images, skeleton_lable, mask, gesture_label) in enumerate(train_dt):
        total_train_step += 1

        #loss_value = 0
        #aux_loss = 0
        #crds_loss = 0

        # [new in v3.3] 分段訓練
        # 依據 Tensorflow 官方指引，若骨幹網路使用預訓練權重
        # 則在訓練前半段先鎖定其權重，待至其他網路收斂後再一起加入訓練
        if waiting4header and step > 1 and loss_value < 0.5:
            backbone.trainable = True
            waiting4header = False
            print("Start training backbone")

        with tf.GradientTape(persistent=True) as tape:
            # 估計
            model_output = custom_model(images)
            # 計算損失值
            loss_value, crds_loss ,aux_loss, gesture_loss, shared_loss, gesture_acc = new_get_losses(model_output, skeleton_lable, gesture_label, batch_size, keypoints, image_size, mask)

            total_loss += loss_value
            
            with train_summary_writer.as_default():
                # 紀錄損失值
                tf.summary.scalar('total_loss', loss_value, total_train_step)
                tf.summary.scalar('crds_loss', crds_loss, total_train_step)
                tf.summary.scalar('aux_loss', aux_loss, total_train_step)
                tf.summary.scalar('gesture_loss', gesture_loss, total_train_step)
                tf.summary.scalar('gesture_acc', gesture_acc, total_train_step)
                tf.summary.scalar('shared_loss', shared_loss, total_train_step)

        # 取得權重
        backbone_weights = custom_model.get_layer("Backbone_layer").trainable_variables
        mask_weights = custom_model.get_layer("Mask_layer").trainable_variables
        pos_weights = custom_model.get_layer("Position_layer").trainable_variables
        gesture_weights = custom_model.get_layer("Gesture_layer").trainable_variables
        shared_weights = custom_model.get_layer("Shared_layer").trainable_variables

        # 計算梯度
        backbone_grads = tape.gradient(loss_value, backbone_weights)
        mask_grads = tape.gradient(aux_loss, mask_weights)
        pos_grads = tape.gradient(crds_loss, pos_weights)
        gesture_grads = tape.gradient(gesture_loss, gesture_weights)
        shared_grads = tape.gradient(shared_loss, shared_weights)

        del tape

        # 更新權重
        backbone_optimizer.apply_gradients(zip(backbone_grads, backbone_weights))
        mask_optimizer.apply_gradients(zip(mask_grads, mask_weights))
        pos_optimizer.apply_gradients(zip(pos_grads, pos_weights))
        gesture_optimizer.apply_gradients(zip(gesture_grads, gesture_weights))
        shared_optimizer.apply_gradients(zip(shared_grads, shared_weights))
        

        if step % print_step == 0 and step != 0:
            # 計算執行時間
            elapsed = time.time() - time_counter
            # 計算 將此 step 區間的平均損失值
            avg_loss = total_loss/print_step

            # 紀錄平均損失值
            with train_summary_writer.as_default():
                tf.summary.scalar('avg_loss', avg_loss, total_train_step)
            # 印出資料
            print(f"Epoch: [{epoch_nb}], Step: [{step}], time : [{elapsed:.2f}], average loss : [{avg_loss:.5f}]")
            
            # [new in v3.3] 分段訓練
            # 依據 Tensorflow 官方指引，若骨幹網路使用預訓練權重
            # 則在訓練前半段先鎖定其權重，待至其他網路收斂後再一起加入訓練
            if waiting4header and avg_loss < 0.5:
                backbone.trainable = True
                waiting4header = False
                print("Start training backbone")

            time_counter = time.time()
            total_loss = 0

            #custom_model.save('weights/custom_model_' + version + '_' + dataset + '.h5')

    # 打印學習率
    #print("Learning rate: " + str(optimizer.learning_rate))

    # 驗證
    total_val_step = validation(custom_model, valid_dt, total_val_step)

    # 儲存模型和權重
    #tf.saved_model.save(custom_model, '/weights/custom_model_' + dataset + '.h5')
    custom_model.save('weights/custom_model_' + version + '_' + dataset + '.h5')
    custom_model.save_weights('weights/'+ dataset +'/custom-model_' + current_time + ".ckpt")

print('Training Completed !')