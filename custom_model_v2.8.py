import tensorflow as tf
import time
import numpy as np
import datetime

from detr_tf.loss.loss import new_get_losses
from detr_tf.data.tfcsv import load_new_frei_dataset

from tfswin import SwinTransformerTiny224

# 讀取參數
import argparse
parser = argparse.ArgumentParser(description='Custom model')
# 批次大小
parser.add_argument('-s','--batch_size', default=8, type=int, dest='batch_size', help='Batch size')
# 訓練次數
parser.add_argument('-e','--epoch', default=100, type=int, dest='epoch', help='the number of passes of the entire training dataset')
# 選擇骨幹層網路
parser.add_argument('-b','--backbone', default='ResNet', type=str, dest='backbone_type', help='ResNet or SwinTransformer or MobileNet')

# 預訓練權重檔案位置
parser.add_argument('-p','--path', default=None, type=str, dest='path', help='path of the pretrained weight')
# 預訓練權重檔案位置
parser.add_argument('-w','--wait', default=False, type=bool, dest='waiting4header', help='Wait until the loss value is < 0.5, restart training pretrained layer')

# 解析參數(轉換格式)
args = parser.parse_args()
args = vars(args)

# 將參數帶入變數
batch_size = args['batch_size']
training_epoch = args['epoch']
backbone_type = args['backbone_type']
pretrained_model_path = args['path']
waiting4header = args['waiting4header']

# 相關變數
image_size = [224, 224]
keypoints = 21
print_step = int(1600/batch_size)

dataset = 'Frei_vTouch'
version = 'v2.8'

print('\n>>> Training Detial\n')
print('{0:<20}'.format('Batch size:'), batch_size)
print('{0:<20}'.format('Epoch:'), training_epoch)
print('{0:<20}'.format('Backbone:'), backbone_type)
print('{0:<20}'.format('Pretrain weight:'), pretrained_model_path)
print('{0:<20}'.format('Waiting for header:'), waiting4header, '\n')


# 設定 GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# 圖片輸入層
image_input = tf.keras.Input((image_size[0], image_size[1], 3))

# 骨幹層
# 讀取預訓練權重 [new in v3.4]
if  pretrained_model_path is not None:
    pretrained_model = tf.keras.models.load_model(pretrained_model_path, compile=False)
    backbone = pretrained_model.get_layer('Backbone_layer')
    # 先鎖定權重
    backbone.trainable = False
    waiting4header = True
# Swin Transformer
elif backbone_type=='SwinTransformer':
    
    backbone = tf.keras.models.Sequential([
                SwinTransformerTiny224(include_top=False)
                ], name="Backbone_layer")
    # 回歸與分類共用層
    shared_layer = tf.keras.models.Sequential([
               # 加入 tf.keras.layersGlobalAveragePooling2D
               tf.keras.layers.GlobalAveragePooling2D(),
               tf.keras.layers.Dense(512, activation="relu"),
               tf.keras.layers.Dropout(0.2),
               tf.keras.layers.Dense(512, activation="relu"),
               ], name="Shared_layer")
# ResNet
elif backbone_type=='ResNet':
    
    resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, 
                                                 weights=None, 
                                                 input_tensor=image_input,
                                                 input_shape=(image_size[0], image_size[1], 3), 
                                                 pooling=None)
    backbone = tf.keras.models.Sequential([
                resnet
                ], name="Backbone_layer")
    # 回歸與分類共用層
    shared_layer = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="relu"),
            ], name="Shared_layer")
# MobileNet
elif backbone_type=='MobileNet':
    
    mobilenet = tf.keras.applications.MobileNetV3Large(input_shape=(image_size[0], image_size[1], 3), 
                                                  alpha=1.0, 
                                                  minimalistic=False, 
                                                  include_top=False,
                                                  weights=None, 
                                                  input_tensor=image_input, 
                                                  pooling=None)
    backbone = tf.keras.models.Sequential([
                mobilenet
                ], name="Backbone_layer")
    # 回歸與分類共用層
    shared_layer = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="relu"),
            ], name="Shared_layer")
else:
    print('Error while building Backbone Layer')


# 前饋神經網路 + dropout (用於回歸手部關鍵點)
pos_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation="relu"), # [new in v3.2]
            tf.keras.layers.Dense(keypoints*2, activation="sigmoid"), # change
            ], name="Position_layer")

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


# 輸出資料結構
outputs = {
    'pred_pos': pos_preds,
    'pred_mask': mask_preds,
}

custom_model = tf.keras.Model(image_input, outputs, name="custom_model")

# 印出架構
custom_model.summary()

backbone_initial_lr = 2.5e-5
mask_initial_lr = 1e-4
shared_initial_lr = 1e-5
pos_initial_lr = 2.5e-4

backbone_end_lr =  2.5e-6
mask_end_lr = 1e-5
shared_end_lr = 1e-6
pos_end_lr = 2.5e-5

# 優化器
backbone_optimizer = tf.keras.optimizers.Adam(learning_rate=backbone_initial_lr)
mask_optimizer = tf.keras.optimizers.Adam(learning_rate=mask_initial_lr)
shared_optimizer = tf.keras.optimizers.Adam(learning_rate=shared_initial_lr)
pos_optimizer = tf.keras.optimizers.Adam(learning_rate=pos_initial_lr)

# 載入資料集
new_frei_hand_dataset = load_new_frei_dataset(batch_size)

total_data_batch = tf.data.experimental.cardinality(new_frei_hand_dataset).numpy()

# 切分資料集
train_dt = new_frei_hand_dataset.skip(int(total_data_batch/10))
valid_dt = new_frei_hand_dataset.take(int(total_data_batch/10))


print("Train Dataset Length:", tf.data.experimental.cardinality(train_dt).numpy())
print("Valid Dataset Length:", tf.data.experimental.cardinality(valid_dt).numpy())

# 紀錄
current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
train_summary_writer = tf.summary.create_file_writer('logs/'+ backbone_type + version + '-' + dataset + '-' + current_time)

#with train_summary_writer.as_default():
#    tf.summary.graph(custom_model.get_concrete_model().graph)

total_train_step = 0
total_val_step = 0

# 執行驗證
def validation(val_model, val_data, val_step):
    print('\n>>> Start Validation', end=' ')

    val_avg_loss = 0
    val_avg_crds_loss = 0
    val_avg_aux_loss = 0
    val_avg_shared_loss = 0

    data_len = len(val_data)

    for step , (images, skeleton_lable, mask) in enumerate(val_data):
        val_step += 1
        # 執行估計
        val_output = val_model(images)
        # 計算損失值(MSE誤差)
        val_loss, val_crds_loss ,val_aux_loss, val_gesture_loss, val_shared_loss, val_gesture_acc = new_get_losses(val_output, skeleton_lable, None,  batch_size, keypoints, image_size, mask)

        val_avg_loss += val_loss
        val_avg_crds_loss += val_crds_loss
        val_avg_aux_loss += val_aux_loss
        val_avg_shared_loss += val_shared_loss
    
    val_avg_loss = val_avg_loss/data_len
    val_avg_crds_loss = val_avg_crds_loss/data_len
    val_avg_aux_loss = val_avg_aux_loss/data_len
    val_avg_shared_loss = val_avg_shared_loss/data_len

    # 紀錄損失值 
    with train_summary_writer.as_default():
        tf.summary.scalar('val_avg_loss', val_avg_loss, total_train_step)
        tf.summary.scalar('val_avg_crds_loss', val_avg_crds_loss, total_train_step)
        tf.summary.scalar('val_avg_aux_loss', val_avg_aux_loss, total_train_step)
        tf.summary.scalar('val_avg_shared_loss', val_avg_shared_loss, total_train_step)
    
    print('\r>>> Validation Compeleted\n')
    print(f"Results: average loss : [{val_avg_loss/len(val_data):.5f}], crd loss : [{val_crds_loss:.5f}], aux loss : [{val_aux_loss:.5f}]\n")
    return val_step

# 調整學習率(PolynomialDecay)
def decayed_learning_rate(step, initial_learning_rate, end_learning_rate, decay_steps, power=5):
    step = min(step, decay_steps)
    return ((initial_learning_rate - end_learning_rate)*(1 - step / decay_steps) ** (power)) + end_learning_rate

if backbone_type=='MobileNet':
    tf.keras.backend.set_learning_phase(True)

# Training
total_loss = 0
total_crds_loss = 0
total_aux_loss = 0
total_shared_loss = 0
time_counter = time.time()

# 進行訓練
for epoch_nb in range(training_epoch):
    print("\n>>> Start of Epoch %d\n" % (epoch_nb,))


    # Assing learning_rate 調整學習率
    backbone_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, backbone_initial_lr, backbone_end_lr, training_epoch))
    mask_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, mask_initial_lr, mask_end_lr, training_epoch))
    pos_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, pos_initial_lr, pos_end_lr, training_epoch))
    shared_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, shared_initial_lr, shared_end_lr, training_epoch))


    # 1 step = <batch size> images
    for step , (images, skeleton_lable, mask) in enumerate(train_dt):
        total_train_step += 1

        with tf.GradientTape(persistent=True) as tape:
            # 估計
            model_output = custom_model(images)
            # 計算損失值
            loss_value, crds_loss ,aux_loss, gesture_loss, shared_loss, gesture_acc = new_get_losses(model_output, skeleton_lable, None, batch_size, keypoints, image_size, mask)

            total_loss += loss_value

        # 取得權重
        backbone_weights = custom_model.get_layer("Backbone_layer").trainable_variables
        mask_weights = custom_model.get_layer("Mask_layer").trainable_variables
        pos_weights = custom_model.get_layer("Position_layer").trainable_variables
        shared_weights = custom_model.get_layer("Shared_layer").trainable_variables

        # 計算梯度
        backbone_grads = tape.gradient(loss_value, backbone_weights)
        mask_grads = tape.gradient(aux_loss, mask_weights)
        pos_grads = tape.gradient(crds_loss, pos_weights)
        shared_grads = tape.gradient(shared_loss, shared_weights)

        del tape

        # 更新權重
        backbone_optimizer.apply_gradients(zip(backbone_grads, backbone_weights))
        mask_optimizer.apply_gradients(zip(mask_grads, mask_weights))
        pos_optimizer.apply_gradients(zip(pos_grads, pos_weights))
        shared_optimizer.apply_gradients(zip(shared_grads, shared_weights))
        

        if step % print_step == 0 and step != 0:
            # 計算執行時間
            elapsed = time.time() - time_counter
            # 計算 將此 step 區間的平均損失值
            avg_loss = total_loss/print_step
            avg_crds_loss = total_crds_loss/print_step
            avg_aux_loss = total_aux_loss/print_step
            avg_shared_loss = total_shared_loss/print_step

            total_loss = 0
            total_crds_loss = 0
            total_aux_loss = 0
            total_shared_loss = 0
            time_counter = time.time()

            # 紀錄平均損失值
            with train_summary_writer.as_default():
                tf.summary.scalar('avg_loss', avg_loss, total_train_step)
                tf.summary.scalar('avg_crds_loss', avg_crds_loss, total_train_step)
                tf.summary.scalar('avg_aux_loss', avg_aux_loss, total_train_step)
                tf.summary.scalar('avg_shared_loss', avg_shared_loss, total_train_step)

            # 印出資料
            print(f"Epoch: [{epoch_nb}], Step: [{step}], time : [{elapsed:.2f}], average loss : [{avg_loss:.5f}]")
            
            # [new in v3.3] 分段訓練
            # 依據 Tensorflow 官方指引，若骨幹網路使用預訓練權重
            # 則在訓練前半段先鎖定其權重，待至其他網路收斂後再一起加入訓練
            if waiting4header and step > 1 and loss_value < 3:
                backbone.trainable = True
                waiting4header = False
                print("Start training locked layers")

    # 驗證
    total_val_step = validation(custom_model, valid_dt, total_val_step)

    # 儲存模型和權重
    custom_model.save('weights/custom_model_' + version + '_' + dataset + '_' + backbone_type + '.h5')
    custom_model.save_weights('weights/'+ dataset +'/custom-model_' + version + '_' + current_time + ".ckpt")

print('Training Completed !')