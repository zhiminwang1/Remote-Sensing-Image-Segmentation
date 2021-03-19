import cv2
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import os
import tensorflow.keras.layers as K
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
from tensorflow.keras.layers import Layer
from dice_loss import dice_loss
from WCCE import weighted_categorical_crossentropy
import tensorflow as tf
from attention import PAM,CAM
import time
import datetime
# from denseCRF import CRFs

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

TEST_SET = []
img_path = './test3/images/'
for file in os.listdir(img_path):
    if(os.path.splitext(file)[1] in ['.png','.tif','.jpg']):
        TEST_SET.append(file)

image_size = 128

#[其他，水体，建筑，耕地，林地，草地]
classes = [0.,29.,76.,150.,179.,226.] 

  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 

custom_objects = {  'relu6': K.ReLU(6.),
                    'tf':tf,
                    'DepthwiseConv2D': K.DepthwiseConv2D,
                    'loss':weighted_categorical_crossentropy,
                    'PAM':PAM,
                    'CAM':CAM,
                    '_dice_loss':dice_loss
                }


def color_annotation(img):
    '''
    给class图上色
    '''
    color = np.ones([img.shape[0], img.shape[1], 3])

    #以BGR方式转换的，[B,G,R]
    color[img==0.] = [0, 0, 0]                        #其他，黑色
    color[img==29.] = [255, 0, 0]                     #水体，蓝色
    color[img==76.] = [0, 0, 255]                     #建筑, 红色
    color[img==150.] = [0, 255, 0]                    #耕地，绿色
    color[img==179.] = [255, 255, 0]                  #林地, 青色
    color[img==226.] = [0, 255, 255]                  #草地, 黄色

    return color

def predict(dir,output_path):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(dir,custom_objects=custom_objects)
    for n in tqdm(range(len(TEST_SET))):
        path = TEST_SET[n]
        name = os.path.splitext(path)[0]
        image = load_img(img_path + path)
        image = img_to_array(image)
        image = image / 255
        # image = image[0:image_size,0:image_size,:]
        h,w,_ = image.shape 
        mask_whole = np.zeros((h,w),dtype=np.uint8)
        crop = np.expand_dims(image, axis=0)
        pred = model.predict(crop,verbose=1)
        pred = np.argmax(pred,axis = 2)
        # print(pred) 
        pred = labelencoder.inverse_transform(pred[0])  
        # print (np.unique(pred))  
        pred = pred.reshape((image_size,image_size)).astype(np.uint8)
        mask_whole[0:h,0:w] = pred[:,:]
        color_image = color_annotation(mask_whole[0:h,0:w])
        # print(np.unique(color_image))
        filename = name + '_pre.png'
        # print('图片保存为{}'.format(filename))
        cv2.imwrite(output_path+filename,color_image)

if __name__ == '__main__':
    # dir = './models/cfamnet10.h5'
    # output_path='./test7/cfamnet11/'
    dir = './models/cfamnetxception.h5'
    output_path='./test3/cfamnetxception/'
    if not os.path.exists(output_path): os.mkdir(output_path)
    start_time = time.clock()
    predict(dir = dir,output_path=output_path)
    end_time = time.clock()
    log_time = "训练总时间: " + str(end_time - start_time) + "s"
    print(log_time)
    #CRFs(original_image_path='test/00212.png',predicted_image_path='predict/pre2.png',CRF_image_path='predict/pre2_crf.png',inference_times=10)
    
