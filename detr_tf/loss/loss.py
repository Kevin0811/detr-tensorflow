import tensorflow as tf
from .. import bbox
from .hungarian_matching import hungarian_matching


def get_total_losss(losses):
    """
    Get model total losss including auxiliary loss
    """
    train_loss = ["label_cost", "giou_loss", "l1_loss"]
    loss_weights = [1, 2, 5]

    total_loss = 0
    for key in losses:
        selector = [l for l, loss_name in enumerate(train_loss) if loss_name in key]
        if len(selector) == 1:
            #print("Add to the total loss", key, losses[key], loss_weights[selector[0]])
            total_loss += losses[key]*loss_weights[selector[0]]
    return total_loss


def get_losses(m_outputs, t_bbox, t_class, config):
    losses = get_detr_losses(m_outputs, t_bbox, t_class, config)

    # Get auxiliary loss for each auxiliary output
    if "aux" in m_outputs:
        for a, aux_m_outputs in enumerate(m_outputs["aux"]):
            aux_losses = get_detr_losses(aux_m_outputs, t_bbox, t_class, config, suffix="_{}".format(a))
            losses.update(aux_losses)
    
    # Compute the total loss
    total_loss = get_total_losss(losses)

    return total_loss, losses

def new_get_losses(m_outputs, skeleton_lable, batch_size, keypoints,image_size=[256,256], mask=None):

    total_loss = 0
    aux_losses = 0


    pos_preds = tf.reshape(m_outputs['pred_pos'],[batch_size, keypoints, 2])
    #pos_preds =  m_outputs['pred_pos']

    skeleton_lable = tf.math.divide(skeleton_lable, tf.cast(image_size, tf.float32))
    #skeleton_lable = tf.reshape(skeleton_lable,[batch_size, keypoints*2])

    coords_loss = get_coor_losses(pos_preds, skeleton_lable)
    #print(losses)

    total_loss = coords_loss

    # Get auxiliary loss for each auxiliary output
    if "pred_mask" in m_outputs:
        for a, pred_mask in enumerate(m_outputs["pred_mask"]):
            pred_mask = tf.image.resize(pred_mask, image_size)
            aux_losses = aux_losses + get_aux_losses(pred_mask, mask[a])
            
        total_loss = total_loss + aux_losses/batch_size

    return total_loss, coords_loss, aux_losses

def get_aux_losses(mask_pred, mask_gt):

    sub_mask = tf.math.subtract(mask_pred, mask_gt) # 相減
    sq_mask = tf.math.square(sub_mask) # 平方
    rm_mask = tf.reduce_mean(sq_mask) # 平均

    aux_loss = tf.math.sqrt(rm_mask) # 開根號

    return aux_loss


def get_coor_losses(outputs, skeleton_lable):

    #print('outputs \n' + str(outputs)+'\n')
    #print('skeleton \n' + str(skeleton_lable)+'\n')

    sub_distances = tf.math.subtract(outputs, skeleton_lable) # 相減
    #print(sub_distances)
    
    sq_distances = tf.math.square(sub_distances) # 平方

    rs_distances = tf.reduce_sum(sq_distances, axis=1, keepdims=True) # 相加
    #print('sq_distances \n' + str(sq_distances)+'\n')

    #sum_pool = 2 * tf.keras.layers.AveragePooling1D(pool_size = 2, strides = 2, padding = "valid", data_format='channels_first')(sq_distances)
    #print('sum_pool \n' + str(sum_pool)+'\n')
    # take the square root to get the distance
    #dists = tf.math.sqrt(sum_pool)
    #print('dists \n' + str(dists)+'\n')
    #reshape_dists = tf.reshape(dists,[config.batch_size, 15])
    # take the mean of the distances
    #mean_dist = tf.reduce_mean(reshape_dists, axis=1, keepdims=False)
    #print('mean_dist \n' + str(mean_dist)+'\n')

    ##rm_distances = tf.reduce_mean(sq_distances) # 平均

    sqrt_loss = tf.math.sqrt(rs_distances) # 開根號

    #mse_loss = tf.losses.mean_squared_error(outputs, skeleton_lable)

    #sq_loss = tf.math.sqrt(mse_loss) # 開根號

    coords_loss = tf.reduce_mean(sqrt_loss)

    #print(tf.print(total_loss))
    return coords_loss

def loss_labels(p_bbox, p_class, t_bbox, t_class, t_indices, p_indices, t_selector, p_selector, background_class=0):

    neg_indices = tf.squeeze(tf.where(p_selector == False), axis=-1)
    neg_p_class = tf.gather(p_class, neg_indices)
    neg_t_class = tf.zeros((tf.shape(neg_p_class)[0],), tf.int64) + background_class
    
    neg_weights = tf.zeros((tf.shape(neg_indices)[0],)) + 0.1
    pos_weights = tf.zeros((tf.shape(t_indices)[0],)) + 1.0
    weights = tf.concat([neg_weights, pos_weights], axis=0)
    
    pos_p_class = tf.gather(p_class, p_indices)
    pos_t_class = tf.gather(t_class, t_indices)

    #############
    # Metrics
    #############
    # True negative
    cls_neg_p_class = tf.argmax(neg_p_class, axis=-1)
    true_neg  = tf.reduce_mean(tf.cast(cls_neg_p_class == background_class, tf.float32))
    # True positive
    cls_pos_p_class = tf.argmax(pos_p_class, axis=-1)
    true_pos = tf.reduce_mean(tf.cast(cls_pos_p_class != background_class, tf.float32))
    # True accuracy
    cls_pos_p_class = tf.argmax(pos_p_class, axis=-1)
    pos_accuracy = tf.reduce_mean(tf.cast(cls_pos_p_class == pos_t_class, tf.float32))

    targets = tf.concat([neg_t_class, pos_t_class], axis=0)
    preds = tf.concat([neg_p_class, pos_p_class], axis=0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(targets, preds)
    loss = tf.reduce_sum(loss * weights) / tf.reduce_sum(weights)

    return loss, true_neg, true_pos, pos_accuracy


def loss_boxes(p_bbox, p_class, t_bbox, t_class, t_indices, p_indices, t_selector, p_selector):
    #print("------")
    p_bbox = tf.gather(p_bbox, p_indices)
    t_bbox = tf.gather(t_bbox, t_indices)


    p_bbox_xy = bbox.xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = bbox.xcycwh_to_xy_min_xy_max(t_bbox)

    l1_loss = tf.abs(p_bbox-t_bbox)
    l1_loss = tf.reduce_sum(l1_loss) / tf.cast(tf.shape(p_bbox)[0], tf.float32)

    iou, union = bbox.jaccard(p_bbox_xy, t_bbox_xy, return_union=True)

    _p_bbox_xy, _t_bbox_xy = bbox.merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:,:,:2], _t_bbox_xy[:,:,:2])
    bottom_right =  tf.math.maximum(_p_bbox_xy[:,:,2:], _t_bbox_xy[:,:,2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:,:,0] * size[:,:,1]
    giou = (iou - (area - union) / area)
    loss_giou = 1 - tf.linalg.diag_part(giou)

    loss_giou = tf.reduce_sum(loss_giou) / tf.cast(tf.shape(p_bbox)[0], tf.float32)

    return loss_giou, l1_loss

def get_detr_losses(m_outputs, target_bbox, target_label, config, suffix=""):

    predicted_bbox = m_outputs["pred_boxes"]
    predicted_label = m_outputs["pred_logits"]

    all_target_bbox = []
    all_target_class = []
    all_predicted_bbox = []
    all_predicted_class = []
    all_target_indices = []
    all_predcted_indices = []
    all_target_selector = []
    all_predcted_selector = []

    t_offset = 0
    p_offset = 0

    for b in range(predicted_bbox.shape[0]):

        p_bbox, p_class, t_bbox, t_class = predicted_bbox[b], predicted_label[b], target_bbox[b], target_label[b]
        t_indices, p_indices, t_selector, p_selector, t_bbox, t_class = hungarian_matching(t_bbox, t_class, p_bbox, p_class, slice_preds=True)

        t_indices = t_indices + tf.cast(t_offset, tf.int64)
        p_indices = p_indices + tf.cast(p_offset, tf.int64)

        all_target_bbox.append(t_bbox)
        all_target_class.append(t_class)
        all_predicted_bbox.append(p_bbox)
        all_predicted_class.append(p_class)
        all_target_indices.append(t_indices)
        all_predcted_indices.append(p_indices)
        all_target_selector.append(t_selector)
        all_predcted_selector.append(p_selector)

        t_offset += tf.shape(t_bbox)[0]
        p_offset += tf.shape(p_bbox)[0]

    all_target_bbox = tf.concat(all_target_bbox, axis=0)
    all_target_class = tf.concat(all_target_class, axis=0)
    all_predicted_bbox = tf.concat(all_predicted_bbox, axis=0)
    all_predicted_class = tf.concat(all_predicted_class, axis=0)
    all_target_indices = tf.concat(all_target_indices, axis=0)
    all_predcted_indices = tf.concat(all_predcted_indices, axis=0)
    all_target_selector = tf.concat(all_target_selector, axis=0)
    all_predcted_selector = tf.concat(all_predcted_selector, axis=0)


    label_cost, true_neg, true_pos, pos_accuracy = loss_labels(
        all_predicted_bbox,
        all_predicted_class,
        all_target_bbox,
        all_target_class,
        all_target_indices,
        all_predcted_indices,
        all_target_selector,
        all_predcted_selector,
        background_class=config.background_class,
    )

    giou_loss, l1_loss = loss_boxes(
        all_predicted_bbox,
        all_predicted_class,
        all_target_bbox,
        all_target_class,
        all_target_indices,
        all_predcted_indices,
        all_target_selector,
        all_predcted_selector
    )

    label_cost = label_cost
    giou_loss = giou_loss
    l1_loss = l1_loss

    return {
        "label_cost{}".format(suffix): label_cost,
        "true_neg{}".format(suffix): true_neg,
        "true_pos{}".format(suffix): true_pos,
        "pos_accuracy{}".format(suffix): pos_accuracy,
        "giou_loss{}".format(suffix): giou_loss,
        "l1_loss{}".format(suffix): l1_loss
    }
