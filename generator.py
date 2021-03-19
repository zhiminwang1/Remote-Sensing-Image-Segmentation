from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import os
import cv2

n_label = 6

#[其他，水体，建筑，耕地，林地，草地]
classes = [0.,29.,76.,150.,179.,226.] 
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 

filepath = './train/'

def randomcrop(img,size=128):
    height,width,_ = img.shape

    random_width = random.randint(0, width - size - 1)
    random_height = random.randint(0, height - size - 1)

    img_crop = img[random_height: random_height + size, random_width: random_width + size,:]

    return img_crop
    

def get_train_val(val_rate=0.2,num_rate=0.5):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'images'):
        if(os.path.splitext(pic)[1] in ['.png','.tif','.jpg']):
            train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    total_num = int(total_num * num_rate)
    val_num = int(val_rate * total_num)
    for i in range(total_num):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

# def get_train_val(val_rate=0.2,num_rate=0.5,block_num=1000):
#     train_url = []    
#     train_set = []
#     val_set  = []
#     for pic in os.listdir(filepath + 'images'):
#         if(os.path.splitext(pic)[1] in ['.png','.tif','.jpg']):
#             train_url.append(pic)
#     set_num = len(train_url) // block_num
#     for i in range(set_num):
#         block_url = train_url[i*block_num:(i+1)*block_num]
#         random.shuffle(block_url)
#         total_num = int(block_num * num_rate)
#         val_num = int(val_rate * total_num)
#         for j in range(total_num):
#             if j < val_num:
#                 val_set.append(train_url[i*block_num+j]) 
#             else:
#                 train_set.append(train_url[i*block_num+j])
#     return train_set,val_set
  
# data for training  
def generateData(batch_size,data=[],size=128):  
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(filepath + 'images/' + url)
            img = img_to_array(img)
            # img = randomcrop(img,size)
            img = img / 255
            train_data.append(img)  
            label = load_img(filepath + 'labels/' + url, color_mode='grayscale')
            label = img_to_array(label)
            # label = randomcrop(label,size)
            label = label.reshape((size * size,))  
            train_label.append(label)  
            if batch % batch_size==0: 
                train_data = np.array(train_data)  
                train_label = np.array(train_label).flatten()  
                train_label = labelencoder.transform(train_label) 
                train_label = to_categorical(train_label, num_classes=n_label)  
                train_label = train_label.reshape((batch_size,size*size,n_label))  
                yield (train_data,train_label) 
                train_data = []  
                train_label = []  
                batch = 0  
 
# data for validation 
def generateValidData(batch_size,data=[],size=128):  
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'images/' + url)
            img = img_to_array(img)
            # img = randomcrop(img,size)
            img = img / 255  
            valid_data.append(img)  
            label = load_img(filepath + 'labels/' + url, color_mode='grayscale')
            label = img_to_array(label)
            # label = randomcrop(label,size)
            label = label.reshape((size * size,))    
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label).flatten()  
                valid_label = labelencoder.transform(valid_label)  
                valid_label = to_categorical(valid_label, num_classes=n_label)  
                valid_label = valid_label.reshape((batch_size,size*size,n_label))  
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0 
