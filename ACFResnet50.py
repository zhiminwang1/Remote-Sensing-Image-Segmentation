from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D, Dropout,Lambda,Concatenate
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D,DepthwiseConv2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf


def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x


def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def bottleneck_Block(input, out_filters, strides=(1, 1), dilation=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same',
               dilation_rate=dilation, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x

def acf_module(coarse_input,feature_map):
    input_shape = coarse_input.get_shape().as_list()
    _,H,W,N = input_shape

    coarse = tf.transpose(K.reshape(coarse_input,(-1,H*W,N)),(0,2,1))
    C = 64
    x = Conv2D(C,(1,1),padding='same',use_bias=False, activation=None,name='feature_map_conv1')(feature_map)
    x = BatchNormalization(name='feature_map_conv1_BN')(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.1)(x)
    x = K.reshape(x,(-1,H*W,C))
 
    x = K.batch_dot(coarse,x)
    x = tf.subtract(K.max(x,axis=-1,keepdims=True),x)
    x = tf.nn.softmax(x,axis=-1)

    x = tf.transpose(x,(0,2,1))
    x = K.batch_dot(x,coarse)

    x = tf.transpose(x,(0,2,1))
    x = K.reshape(x,(-1,H,W,C))

    x = Conv2D(C,(1,1),padding='same',use_bias=False, activation=None,name='feature_map_conv2')(x)

    return x


def SepConv_BN(x, filters, stride=1, kernel_size=3,atrous_rate=1,epsilon=1e-3):

    x = DepthwiseConv2D(kernel_size,stride, dilation_rate=atrous_rate,padding='same', use_bias=False)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',use_bias=False)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    x = Activation(tf.nn.relu)(x)

    return x


def ASPP_Model(input,atrous_rates):
    input_shape = input.get_shape().as_list()
    _,h,w,filters  = input_shape
    out_filters = filters // 4
    rate1, rate2, rate3 = tuple(atrous_rates)

    b0 = Conv2D(out_filters,1,padding='same',use_bias=False)(input)
    b0 = BatchNormalization(epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)
    b1 = SepConv_BN(input,out_filters,atrous_rate=rate1)
    b2 = SepConv_BN(input,out_filters,atrous_rate=rate2)
    b3 = SepConv_BN(input,out_filters,atrous_rate=rate3)
    b4 = GlobalAveragePooling2D()(input)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(out_filters, (1, 1), padding='same',use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, (h,w),method='bilinear', align_corners=True))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])

    return x

def FCNHead(x,out_filters,height,width):
    input_shape = x.get_shape().as_list()
    _,h,w,filters  = input_shape

    x = Conv2D(filters,3,padding='same',use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Lambda(lambda x: tf.compat.v1.image.resize(x, (height,width),method='bilinear', align_corners=True))(x)
    x = Conv2D(out_filters,1,padding='same',use_bias=False)(x)

    return x



def acf_resnet50(height, width, channel, classes):
    input = Input(shape=(height, width, channel))

    conv1_1 = Conv2D(64, 7, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    conv1_1 = BatchNormalization(axis=3)(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)
    conv1_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_1)
    
    # conv2_x  1/4
    conv2_1 = bottleneck_Block(conv1_2, 256, strides=(1, 1), with_conv_shortcut=True)
    conv2_2 = bottleneck_Block(conv2_1, 256)
    conv2_3 = bottleneck_Block(conv2_2, 256)

    # conv3_x  1/8
    conv3_1 = bottleneck_Block(conv2_3, 512, strides=(2, 2), with_conv_shortcut=True)
    conv3_2 = bottleneck_Block(conv3_1, 512)
    conv3_3 = bottleneck_Block(conv3_2, 512)
    conv3_4 = bottleneck_Block(conv3_3, 512)

    # conv4_x  1/8
    conv4_1 = bottleneck_Block(conv3_4, 1024, strides=(1, 1), dilation=(2, 2), with_conv_shortcut=True)
    conv4_2 = bottleneck_Block(conv4_1, 1024, dilation=(2, 2))
    conv4_3 = bottleneck_Block(conv4_2, 1024, dilation=(2, 2))
    conv4_4 = bottleneck_Block(conv4_3, 1024, dilation=(2, 2))
    conv4_5 = bottleneck_Block(conv4_4, 1024, dilation=(2, 2))
    conv4_6 = bottleneck_Block(conv4_5, 1024, dilation=(2, 2))
    
    # conv5_x  1/8
    conv5_1 = bottleneck_Block(conv4_6, 2048, strides=(1, 1), dilation=(4, 4), with_conv_shortcut=True)
    conv5_2 = bottleneck_Block(conv5_1, 2048, dilation=(4, 4))
    conv5_3 = bottleneck_Block(conv5_2, 2048, dilation=(4, 4))
    
    feat_aspp = ASPP_Model(conv5_3,atrous_rates=(6, 12, 18))
    # print(feat_aspp.shape)
    # feat_aspp = Lambda(lambda x: tf.compat.v1.image.resize(x, (height,width),method='bilinear', align_corners=True))(feat_aspp)
    
    auxout = FCNHead(conv4_6,classes,height,width)
    auxout = Lambda(lambda x: tf.compat.v1.image.resize(x, (height,width),method='bilinear', align_corners=True))(auxout)
    auxout = Reshape((height*width,classes))(auxout)
    auxout = Activation('softmax',name='auxiliary')(auxout)

    coarse_x = Conv2D(64,1,padding='same',use_bias=False)(feat_aspp)
    coarse_x = BatchNormalization(epsilon=1e-5)(coarse_x)
    coarse_x = Activation('relu')(coarse_x)
    coarse_x = Dropout(0.5)(coarse_x)
    coarse_x = Conv2D(64,3,padding='same',use_bias=False)(coarse_x)
    coarse_x = BatchNormalization(epsilon=1e-5)(coarse_x)
    coarse_x = Activation('relu')(coarse_x)
    coarse_x = Dropout(0.5)(coarse_x)
    coarse_x = Conv2D(classes,1,padding='same',use_bias=False)(coarse_x)
    
    acf_out = acf_module(coarse_x,feat_aspp)

    coarse_x = Lambda(lambda x: tf.compat.v1.image.resize(x, (height,width),method='bilinear', align_corners=True))(coarse_x)
    coarse_x = Reshape((height*width,classes))(coarse_x)
    coarse_x = Activation('softmax',name='coarse_segmentation')(coarse_x)
    
    feat_concat = Concatenate()([acf_out,feat_aspp])
    pre_out = Conv2D(32,3,padding='same',use_bias=False)(feat_concat)
    pre_out = BatchNormalization(epsilon=1e-5)(pre_out)
    pre_out = Activation('relu')(pre_out)
    pre_out = Dropout(0.5)(pre_out)
    
    merge7 = concatenate([conv3_4, pre_out], axis=3)
    conv7 = Conv2d_BN(merge7, 256, 3)
    conv7 = Conv2d_BN(conv7, 256, 3)

    up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 128, 2)
    merge8 = concatenate([conv2_3, up8], axis=3)
    conv8 = Conv2d_BN(merge8, 128, 3)
    conv8 = Conv2d_BN(conv8, 128, 3)

    up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 64, 2)
    merge9 = concatenate([conv1_1, up9], axis=3)
    conv9 = Conv2d_BN(merge9, 64, 3)
    conv9 = Conv2d_BN(conv9, 64, 3)

    up10 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv9), 32, 2)
    conv10 = Conv2d_BN(up10, 32, 3)
    conv10 = Conv2d_BN(conv10, 32, 3)

    conv11 = Conv2d_BN(conv10, classes, 1, use_activation=None)
    conv11_reshape = Reshape((height*width,classes))(conv11)
    output = Activation('softmax', name='Classification')(conv11_reshape)

    # out = [output,coarse_x,auxout]

    model = Model(inputs=input, outputs=output)
    return model


# from keras.utils import plot_model
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#
# model = danet_resnet101(height=512, width=512, channel=3, classes=1)
# model.summary()
# plot_model(model, to_file='danet_resnet101.png', show_shapes=True)
if __name__ == '__main__':
    model = acf_resnet50(height=128, width=128, channel=3, classes=6)
    model.summary()