""" Example on how to finetune on the HardHat dataset
using custom layers. This script assume the dataset is already download 
on your computer in raw and Tensorflow Object detection csv format. 

Please, for more information, checkout the following notebooks:
    - DETR : How to setup a custom dataset
"""

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import cv2
import os

from detr_tf.data.tfcsv import load_tfcsv_dataset, load_k4b_dataset

from detr_tf.networks.detr import get_detr_model
from detr_tf.optimizers import setup_optimizers
from detr_tf.logger.training_logging import train_log, valid_log
from detr_tf.loss.loss import get_losses
from detr_tf.inference import numpy_bbox_to_image
from detr_tf.training_config import TrainingConfig, training_config_parser
from detr_tf import training

try:
    # Should be optional if --log is not set
    import wandb
except:
    wandb = None

import time


def build_model(config):
    """ 
    Build the model with the pretrained weights
    and add new layers to finetune
    """
    # Load the pretrained model with new heads at the top
    # 3 class : background head and helmet (we exclude here person from the dataset)
    detr = get_detr_model(config, include_top=False, nb_class=None, num_decoder_layers=6, num_encoder_layers=6, weights=None)
    #detr = tf.saved_model.load('/weights/k4b')
    #load_status = detr.load_weights("weights/Helmet/detr-model_10.ckpt")
    #print(str(load_status) + "\n")

    # print model
    tf.keras.utils.plot_model(detr, to_file='model.png', dpi=48)

    checkpoint = tf.train.Checkpoint(model=detr)
    #checkpoint.restore("weights/Helmet/detr-model_10.ckpt").assert_consumed()
    
    detr.summary()
    return detr

def show_result(eval_image, lable):
    r, g, b = cv2.split(eval_image)
    eval_image = cv2.merge([b, g, r])
    #print(lable.shape)
    #lable = np.reshape(lable,(15,2))
    #print(lable)
    for coor in lable:
        print(coor)
        eval_image = cv2.circle(eval_image, (int(coor[0]),int(coor[1])), 2, (255, 0, 0), -1)
 
    cv2.imshow('Image', eval_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_finetuning(config):

    # Load the model with the new layers to finetune
    detr = build_model(config)
    

    # Load the training and validation dataset and exclude the person class
    # train_dt, class_names = load_tfcsv_dataset(config, config.batch_size, augmentation=True, exclude=["person"], ann_file="_annotations.csv", img_dir="train")
    # valid_dt, _ = load_tfcsv_dataset(config, 4, augmentation=False, exclude=["person"], ann_file="_annotations.csv", img_dir="test")
    train_dt = load_k4b_dataset(config.batch_size, "train_annos.txt")
    valid_dt = load_k4b_dataset(config.batch_size, "test_annos.txt")

    # Train/finetune the transformers only
    config.train_backbone = tf.Variable(False)
    config.train_transformers = tf.Variable(False)
    config.train_nlayers = tf.Variable(True)
    # Learning rate (NOTE: The transformers and the backbone are NOT trained with)
    # a 0 learning rate. They're not trained, but we set the LR to 0 just so that it is clear
    # in the log that both are not trained at the begining
    config.backbone_lr = tf.Variable(0.0)
    config.transformers_lr = tf.Variable(0.0)
    config.nlayers_lr = tf.Variable(1e-2)

    # Setup the optimziers and the trainable variables
    optimzers = setup_optimizers(detr, config)

    eval_image, lable = training.eval(detr, valid_dt, config, evaluation_step=1)
    show_result(eval_image, lable)


    # Run the training for 180 epochs
    for epoch_nb in range(180):

        #if epoch_nb > 0:
            # After the first epoch, we finetune the transformers and the new layers
        config.train_backbone.assign(True)
        config.backbone_lr.assign(1e-3) # 0.001
        config.train_transformers.assign(True)
        config.transformers_lr.assign(1e-3)
        config.nlayers_lr.assign(0.1)
        print("Start Training")

        # Train
        training.fit(detr, train_dt, optimzers, config, epoch_nb)
        # training.fit(detr, train_dt, optimzers, config, epoch_nb, class_names)
        # 驗證
        eval_image, lable = training.eval(detr, valid_dt, config, evaluation_step=250)
        show_result(eval_image, lable)
        # print last data coor
        for coor in lable:
                print("coor : " + str(coor))
        
        if epoch_nb % 1 == 0:
            detr.save_weights("weights/Helmet/detr-model_" + str(epoch_nb) + ".ckpt")
            #tf.saved_model.save(detr, '/weights/k4b')
            
            

if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #Show more info during process
    #tf.config.experimental.set_device_policy('warn')

    config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    #if config.log:
        #wandb.init(project="detr-tensorflow", reinit=True)
        
    # Run training
    run_finetuning(config)
