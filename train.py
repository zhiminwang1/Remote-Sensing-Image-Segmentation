from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,SGD
from model.Deeplabv3.DeeplabV3_plus import deeplabv3_plus
from model.Deeplabv3.deeplabv3plus import Deeplabv3
from model.ACFNet.ACFNet import ACFNet
from model.ACFNet.deepACFNet import DeepACFNet
from danet import danet_resnet101
from data_process.generator import get_train_val,generateData,generateValidData
from loss.focal_loss import multi_category_focal_loss1
from ACFResnet101 import acf_resnet101
from ACFResnet34 import acf_resnet34
from CAAResnet50 import caa_resnet50
from CFAMNet import cfam_resnet
from DAUNET import da_unet
from UNet import unet
import tensorflow.keras.backend as K
import tensorflow as tf
from dice_loss import dice_loss
from loss.bce_loss import bce_dice_loss
from lovasz_losses import lovasz_softmax
from tensorflow.keras.losses import categorical_crossentropy
from loss1 import acfloss,acfloss2
from WCCE import weighted_categorical_crossentropy
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_set,val_set = get_train_val(val_rate=0.25,num_rate=1)
train_numb = len(train_set)  
valid_numb = len(val_set)

size = 128
n_label = 6
EPOCHS = 30
BS = 16

# model = deeplabv3_plus(input_shape=(256,256,3),out_stride=16,num_classes=n_label)
# model = DeepACFNet(weights='cityscapes',input_shape=(size, size, 3), classes=n_label,backbone='mobilenetv2',activation='softmax')
# model = unet(height=size, width=size, channel=3, num_classes=n_label)
model = cfam_resnet(height=128, width=128, channel=3, classes=6,blocks=[3,4,6,3])
# model.compile(  optimizer=SGD(learning_rate=0.001,momentum=0.9, decay=0.0001),
#                 loss={
#                   'fine_segmentation': 'categorical_crossentropy',
#                   'coarse_segmentation': 'categorical_crossentropy',
#                   'auxiliary': 'categorical_crossentropy'},
#                 loss_weights={
#                   'fine_segmentation': 0.7,
#                   'coarse_segmentation': 0.6,
#                   'auxiliary': 0.4},
#                 metrics=['accuracy'])
# loss = multi_category_focal_loss1(alpha=[2,1,2,1,2,3], gamma=2)

model.compile(  optimizer=Adam(learning_rate=0.0005),
                loss=['categorical_crossentropy'],
                metrics=['accuracy']
             )


#poly策略
def poly_decay(epoch):
    maxEpochs = EPOCHS
    step_each_epoch = train_numb / BS
    baseLR = 0.0005
    power = 0.9
    ite = K.get_value(model.optimizer.iterations)
    # compute the new learning rate based on polynomial decay
    alpha = baseLR*((1 - (ite / float(maxEpochs*step_each_epoch)))**power)
    # return the new learning rate
    return alpha


def train(dir): 
    modelcheck = ModelCheckpoint(dir ,monitor='val_accuracy',save_best_only=True,mode='max',verbose=1) 
    # callable = [modelcheck]
     # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    es = EarlyStopping(
                        monitor='val_accuracy', 
                        min_delta=0, 
                        patience=10, 
                        verbose=1,
                        mode='max'
                    )
    
    reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss', 
                        factor=0.5, 
                        patience=3, 
                        verbose=1
                    )
    
    lrate = LearningRateScheduler(poly_decay)
    model_name = 'unet-{}'.format(int(time.time()))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(model_name))
    callable = [modelcheck,lrate,es,tensorboard] 

    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    #  获取当前时间
    start_time = datetime.datetime.now()

    H = model.fit_generator(generator=generateData(BS,train_set,size),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                            validation_data=generateValidData(BS,val_set,size),validation_steps=valid_numb//BS,callbacks=callable,max_queue_size=1)  

    #  训练总时间
    end_time = datetime.datetime.now()
    log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
    print(log_time)

    # # plot the training loss and accuracy
    # plt.style.use("ggplot")
    # N = EPOCHS

    # fig1 = plt.figure(figsize=(8,4))
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # # plt.title('Training and validation loss')
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(dir+"loss.png",dpi = 300)

    # fig2 = plt.figure(figsize=(8,4))
    # plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
    # plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
    # # plt.title('Training and validation accuracy')
    # plt.xlabel("Iterations")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.savefig(dir + "accuracy.png",dpi = 300)

    # plt.show()

if __name__=='__main__':  
    train(dir = './save_modelsl/cfamnetxx.h5')
    # mobilenet_unet(n_classes=n_label)

