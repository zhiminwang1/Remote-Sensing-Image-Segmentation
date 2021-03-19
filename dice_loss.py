import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.utils import to_categorical

def dice_loss(beta=1, smooth = 1e-5):
    def _dice_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        tp = K.sum(y_true_f * y_pred_f)
        fp = K.sum(y_pred_f)
        fn = K.sum(y_true_f)

        score = ((1 + beta ** 2) * tp + smooth) / (beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, focal_loss])
        return dice_loss
    return _dice_loss

# def dice_loss_with_CE(y_true, y_pred,beta=1, smooth = 1e-5):
#     y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)

#     tp = K.sum(y_true_f * y_pred_f)
#     fp = K.sum(y_pred_f)
#     fn = K.sum(y_true_f)

#     score = ((1 + beta ** 2) * tp + smooth) / (beta ** 2 * fn + fp + smooth)
#     score = tf.reduce_mean(score)
#     dice_loss = 1 - score
#     # dice_loss = tf.Print(dice_loss, [dice_loss, focal_loss])
#     return dice_loss

# output = np.array([[[0,1,1,3,2,3,0,0,0,1,0]]],dtype = np.float32)
# label = np.array([[[0,1,0,3,0,1,1,0,0,1,0]]],dtype = np.float32)
# # output = tf.nn.softmax(output)
# output = to_categorical(output)
# label = to_categorical(label)
# loss = dice_loss_with_CE(label,output)
# print(loss)