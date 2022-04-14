import tensorflow as tf
import time
import numpy as np
import datetime

from detr_tf.loss.loss import new_get_losses
from detr_tf.data.tfcsv import load_vtouch_dataset

# 命名
# crds = position = keypoints = pos
# aux = mask
# gesture

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
gesture_cnt = 14
print_step = int(1600/batch_size)

dataset = 'vTouch'
version = 'v3.7_gesture_only'

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
if pretrained_model_path is not None: # 讀取預訓練權重 [new in v3.4]
    pretrained_model = tf.keras.models.load_model(pretrained_model_path, compile=False)

    layer_name = [layer.name for layer in pretrained_model.layers]

    backbone = pretrained_model.get_layer('Backbone_layer')
    shared_layer = pretrained_model.get_layer('Shared_layer')
    
    backbone.trainable = False
    shared_layer.trainable = False

# Swin Transformer
elif backbone_type=='SwinTransformer':

    from tfswin import SwinTransformerTiny224
    
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

# 前饋神經網路 + dropout (用於分類手勢)
gesture_layer = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, activation="relu"), # [new in v3.6] 64 → 128
                tf.keras.layers.Dense(gesture_cnt, activation="softmax"), # change
                ], name="Gesture_layer")

# 資料處理流程
featuremap = backbone(image_input)
shared_outputs = shared_layer(featuremap)
gesture_preds = gesture_layer(shared_outputs)

# 輸出資料結構
outputs = {
    'pred_gesture': gesture_preds
}

custom_model = tf.keras.Model(image_input, outputs, name="custom_model")

# 印出架構
custom_model.summary()

backbone_initial_lr = 2.5e-5
shared_initial_lr = 1e-5
gesture_initial_lr = 1e-5

backbone_end_lr =  2.5e-6
shared_end_lr = 1e-6
gesture_end_lr = 1e-6

# 優化器
backbone_optimizer = tf.keras.optimizers.Adam(learning_rate=backbone_initial_lr)
shared_optimizer = tf.keras.optimizers.Adam(learning_rate=shared_initial_lr)
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
train_summary_writer = tf.summary.create_file_writer('logs/'+ backbone_type + version + '-' + dataset + '-' + current_time)

total_train_step = 0
total_val_step = 0

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
        val_loss, val_crds_loss ,val_aux_loss, val_gesture_loss, val_shared_loss, val_gesture_acc = new_get_losses(val_output, skeleton_lable,gesture_label,  batch_size, keypoints, image_size, mask)

        val_avg_loss += val_loss
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

    # 紀錄損驗證資料集的平均失值值
    with train_summary_writer.as_default():
        tf.summary.scalar('val_avg_loss', val_avg_loss, total_train_step)
        tf.summary.scalar('val_avg_crds_loss', val_avg_crds_loss, total_train_step)
        tf.summary.scalar('val_avg_aux_loss', val_avg_aux_loss, total_train_step)
        tf.summary.scalar('val_avg_gesture_loss', val_avg_gesture_loss, total_train_step)
        tf.summary.scalar('val_avg_shared_loss', val_avg_shared_loss, total_train_step)
        tf.summary.scalar('val_avg_gesture_acc', val_avg_gesture_acc, total_train_step)
    
    print('\r>>> Validation Compeleted\n')
    print(f"Results: val average loss : [{val_avg_loss:.5f}], val gesture acc : [{val_avg_gesture_acc:.5f}]\n")
    return val_step

# 調整學習率(PolynomialDecay)
def decayed_learning_rate(step, initial_learning_rate, end_learning_rate, decay_steps, power=5):
    step = min(step, decay_steps)
    return ((initial_learning_rate - end_learning_rate)*(1 - step / decay_steps) ** (power)) + end_learning_rate

if backbone_type=='MobileNet':
    tf.keras.backend.set_learning_phase(True)

# 進行訓練
for epoch_nb in range(training_epoch):
    print("\n>>> Start of Epoch %d\n" % (epoch_nb,))

    total_loss = 0
    total_crds_loss = 0
    total_aux_loss = 0
    total_gesture_loss = 0
    total_shared_loss = 0
    total_gesture_acc = 0
    time_counter = time.time()

    # Assing learning_rate 調整學習率
    backbone_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, backbone_initial_lr, backbone_end_lr, training_epoch))
    gesture_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, gesture_initial_lr, gesture_end_lr, training_epoch))
    shared_optimizer.learning_rate.assign(decayed_learning_rate(epoch_nb, shared_initial_lr, shared_end_lr, training_epoch))

    # 紀錄學習率
    #with train_summary_writer.as_default():
    #    tf.summary.scalar('Backbone learning rate', backbone_optimizer.learning_rate, total_train_step)
    #    tf.summary.scalar('Gesture learning rate', gesture_optimizer.learning_rate, total_train_step)
    #    tf.summary.scalar('Shared learning rate', shared_optimizer.learning_rate, total_train_step)

    # 1 step = <batch size> images
    for step , (images, skeleton_lable, mask, gesture_label) in enumerate(train_dt):
        total_train_step += 1

        with tf.GradientTape(persistent=True) as tape:
            # 估計
            model_output = custom_model(images)
            # 計算損失值
            loss_value, crds_loss ,aux_loss, gesture_loss, shared_loss, gesture_acc = new_get_losses(model_output, skeleton_lable, gesture_label, batch_size, keypoints, image_size, mask)

            total_loss += loss_value
            total_crds_loss += crds_loss
            total_aux_loss += aux_loss
            total_gesture_loss += gesture_loss
            total_shared_loss += shared_loss
            total_gesture_acc += gesture_acc

        # 取得權重
        backbone_weights = custom_model.get_layer("Backbone_layer").trainable_variables
        gesture_weights = custom_model.get_layer("Gesture_layer").trainable_variables
        shared_weights = custom_model.get_layer("Shared_layer").trainable_variables

        # 計算梯度
        backbone_grads = tape.gradient(gesture_loss, backbone_weights)
        gesture_grads = tape.gradient(gesture_loss, gesture_weights)
        shared_grads = tape.gradient(gesture_loss, shared_weights)

        del tape

        # 更新權重
        backbone_optimizer.apply_gradients(zip(backbone_grads, backbone_weights))
        gesture_optimizer.apply_gradients(zip(gesture_grads, gesture_weights))
        shared_optimizer.apply_gradients(zip(shared_grads, shared_weights))
        

        if step % print_step == 0 and step != 0:
            # 計算執行時間
            elapsed = time.time() - time_counter
            # 計算 將此 step 區間的平均損失值
            avg_loss = total_loss/print_step
            avg_crds_loss = total_crds_loss/print_step
            avg_aux_loss = total_aux_loss/print_step
            avg_gesture_loss = total_gesture_loss/print_step
            avg_gesture_acc = total_gesture_acc/print_step
            avg_shared_loss = total_shared_loss/print_step

            total_loss = 0
            total_crds_loss = 0
            total_aux_loss = 0
            total_gesture_loss = 0
            total_shared_loss = 0
            total_gesture_acc = 0
            time_counter = time.time()

            # 紀錄平均損失值
            with train_summary_writer.as_default():
                tf.summary.scalar('avg_loss', avg_loss, total_train_step)
                tf.summary.scalar('avg_crds_loss', avg_crds_loss, total_train_step)
                tf.summary.scalar('avg_aux_loss', avg_aux_loss, total_train_step)
                tf.summary.scalar('avg_gesture_loss', avg_gesture_loss, total_train_step)
                tf.summary.scalar('avg_gesture_acc', avg_gesture_acc, total_train_step)
                tf.summary.scalar('avg_shared_loss', avg_shared_loss, total_train_step)

            # 印出資料
            print(f"Epoch: [{epoch_nb}], Step: [{step}], time : [{elapsed:.2f}], average loss : [{avg_loss:.5f}], gesture accuracy : [{avg_gesture_acc:.5f}]")

            # [new in v3.3] 分段訓練
            # 依據 Tensorflow 官方指引，若骨幹網路使用預訓練權重
            # 則在訓練前半段先鎖定其權重，待至其他網路收斂後再一起加入訓練
            if waiting4header and avg_loss < 0.5:
                backbone.trainable = False
                shared_layer.trainable = False
                waiting4header = False
                print("Start training locked layers")

    # 驗證
    total_val_step = validation(custom_model, valid_dt, total_val_step)

    # 儲存模型和權重
    #tf.saved_model.save(custom_model, '/weights/custom_model_' + dataset + '.h5')
    custom_model.save('weights/custom_model_' + version + '_' + backbone_type + '.h5')
    custom_model.save_weights('weights/'+ dataset +'/custom-model_' + version + '_' + current_time + ".ckpt")


print('Training Completed !')