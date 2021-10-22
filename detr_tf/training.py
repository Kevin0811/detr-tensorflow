import tensorflow as tf

from .optimizers import gather_gradient, aggregate_grad_and_apply
from .logger.training_logging import valid_log, train_log
from .loss.loss import get_losses,new_get_losses
import time
import numpy as np
import wandb

@tf.function
def run_train_step(model, images, skeleton_lable, optimizers, config):

    if config.target_batch is not None:
        gradient_aggregate = int(config.target_batch // config.batch_size)
    else:
        gradient_aggregate = 1

    with tf.GradientTape() as tape:
        # Compute own loss
        m_outputs = model(images, training=True)
        total_loss = new_get_losses(m_outputs, skeleton_lable, config)
        
        total_loss = total_loss / gradient_aggregate

    # Compute gradient for each part of the network
    gradient_steps = gather_gradient(model, optimizers, total_loss, tape, config)

    return m_outputs, total_loss, gradient_steps


@tf.function
def run_val_step(model, images, skeleton_lable, config):
    m_outputs = model(images, training=False)
    total_loss = new_get_losses(m_outputs, skeleton_lable, config)
    return m_outputs, total_loss


def fit(model, train_dt, optimizers, config, epoch_nb):
    """ 
    Train the model for one epoch
    """
    # Aggregate the gradient for bigger batch and better convergence
    gradient_aggregate = None
    if config.target_batch is not None:
        gradient_aggregate = int(config.target_batch // config.batch_size)

    # Timer t
    t = None

    avg_loss = 0

    for epoch_step , (images, skeleton_lable) in enumerate(train_dt):

        # Run the prediction and retrieve the gradient step for each part of the network
        m_outputs, total_loss, gradient_steps = run_train_step(model, images, skeleton_lable, optimizers, config)
        #print(total_loss)
        avg_loss+=total_loss
        
        # Load the predictions
        #if config.log:
        #    train_log(images, t_bbox, t_class, m_outputs, config, config.global_step,  class_names, prefix="train/")
        
        # Aggregate and apply the gradient (update weights)
        for name in gradient_steps:
            aggregate_grad_and_apply(name, optimizers, gradient_steps[name]["gradients"], epoch_step, config)

        # Log every 250 steps
        if epoch_step % 250 == 0 and epoch_step != 0:
            avg_loss = avg_loss/250
            t = t if t is not None else time.time()
            elapsed = time.time() - t
            print(f"Epoch: [{epoch_nb}], Step: [{epoch_step}], time : [{elapsed:.2f}], loss : [{avg_loss:.2f}]")
            #if config.log:
                #wandb.log({f"train/{k}":log[k] for k in log}, step=config.global_step)
            t = time.time()
            avg_loss = 0

        #print(config.global_step)
        
        config.global_step += 1
        
    print("Finish")

    return model


def eval(model, valid_dt, config, evaluation_step=2000):
    """
    Evaluate the model on the validation set
    """
    t = None
    avg_loss = 0
    for val_step, (images, skeleton_lable) in enumerate(valid_dt):
        # Run prediction
        m_outputs, total_loss = run_val_step(model, images, skeleton_lable, config)
        avg_loss+=total_loss
        # Log the predictions
        #if config.log:
        #    valid_log(images, skeleton_lable, m_outputs, config, val_step, config.global_step,  class_name, evaluation_step=evaluation_step, prefix="train/")
        # Log the metrics
        #if config.log and val_step == 0:
        #    wandb.log({f"val/{k}":log[k] for k in log}, step=config.global_step)
        # Log the progress
        if val_step % 100 == 0:
            t = t if t is not None else time.time()
            elapsed = time.time() - t
            print(f"Validation step: [{val_step}], time : [{elapsed:.2f}], loss : [{avg_loss/100:.2f}]")

            avg_loss = 0
            #print(m_outputs)
            #print(f"Validation step: [{val_step}], \t giou : [{log['giou_loss']:.2f}] \t l1 : [{log['l1_loss']:.2f}] \t time : [{elapsed:.2f}]")
        # 顯示最後一張圖
        if val_step+1 >= evaluation_step:
            image_u8 = tf.cast(images[0], tf.uint8)
            #eval_image = tf.io.encode_png(image_u8)
            np_image = np.array(image_u8)
            #print("Output \n" + str(m_outputs) + "\n")
            #break
            
            print("Model Output \n" + str(m_outputs['pred_pos']))

            skeleton_lable = tf.cast(m_outputs['pred_pos'][0], dtype=tf.int32) 
            skeleton_lable = np.array(skeleton_lable)

            return np_image, skeleton_lable
