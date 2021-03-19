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

def cfam_module(input,classes=6,channel=128,channel1=64):
    input_shape = input.get_shape().as_list()
    _,H,W,_ = input_shape
    N = classes
    C = channel
    C1 = channel1
    x = Conv2D(C,3,padding='same',use_bias=False)(input)
    x1 = Conv2D(C1,1,padding='same',use_bias=False)(x)
    x1 = tf.transpose(K.reshape(x1,(-1,H*W,C1)),(0,2,1))
    p = Conv2D(N,1,padding='same',use_bias=False)(x)
    p1 = Activation('softmax')(p)
    p1 = K.reshape(p1,(-1,H*W,N))
    A = K.batch_dot(x1,p1)
    A = Activation('softmax')(A)
    p1 = tf.transpose(p1,(0,2,1))
    x2 = K.batch_dot(A,p1)
    x2 = K.reshape(tf.transpose(x2,(0,2,1)),(-1,H,W,C1))
    x2 = Conv2D(C,(1,1),padding='same',use_bias=False)(x2)
    x2 = BatchNormalization(epsilon=1e-3)(x2)
    x2 = Activation(tf.nn.relu)(x2)
    x3 = Concatenate()([x2,x])
    y = Conv2D(C,(1,1),padding='same',use_bias=False)(x3)
    y = BatchNormalization(epsilon=1e-3)(y)
    y = Activation(tf.nn.relu)(y)

    return y


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



def cfamnatt(height, width, channel, classes):
    input = Input(shape=(height, width, channel))

    conv1_1 = Conv2D(64, 7, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    conv1_1 = BatchNormalization(axis=3)(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)
    conv1_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_1)
    
    # conv2_x  1/4
    conv2_1 = bottleneck_Block(conv1_2, 256, strides=(1, 1), with_conv_shortcut=True)
    conv2_2 = bottleneck_Block(conv2_1, 256)
    conv2_3 = bottleneck_Block(conv2_2, 256)
    skip1 = conv2_3

    # conv3_x  1/8
    conv3_1 = bottleneck_Block(conv2_3, 512, strides=(2, 2), with_conv_shortcut=True)
    conv3_2 = bottleneck_Block(conv3_1, 512)
    conv3_3 = bottleneck_Block(conv3_2, 512)
    conv3_4 = bottleneck_Block(conv3_3, 512)

    # conv4_x  1/16
    conv4_1 = bottleneck_Block(conv3_4, 1024, strides=(2, 2), with_conv_shortcut=True)
    conv4_2 = bottleneck_Block(conv4_1, 1024)
    conv4_3 = bottleneck_Block(conv4_2, 1024)
    conv4_4 = bottleneck_Block(conv4_3, 1024)
    conv4_5 = bottleneck_Block(conv4_4, 1024)
    conv4_6 = bottleneck_Block(conv4_5, 1024)
    
    # conv5_x  1/16
    conv5_1 = bottleneck_Block(conv4_6, 2048, strides=(1, 1), dilation=(2, 2), with_conv_shortcut=True)
    conv5_2 = bottleneck_Block(conv5_1, 2048, dilation=(2, 2))
    conv5_3 = bottleneck_Block(conv5_2, 2048, dilation=(2, 2))
    
    feat_aspp = ASPP_Model(conv5_3,atrous_rates=(6, 12, 18))
    
    aspp = Conv2d_BN(feat_aspp,256,1)

    x = Conv2d_BN(aspp,128,1)
    skip_size = tf.keras.backend.int_shape(skip1)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, skip_size[1:3],method='bilinear', align_corners=True))(x)

    up1 = Conv2d_BN(conv3_4,128,1)
    up1 = Lambda(lambda xx: tf.compat.v1.image.resize(xx, skip_size[1:3],method='bilinear', align_corners=True))(up1)
    up2 = Conv2d_BN(conv2_3,128,1)

    up = Concatenate()([up1,up2])
    up = Conv2d_BN(up,128,1)

    x = Concatenate()([x,up])
    x = Conv2d_BN(x,classes,3)
    size_before = tf.keras.backend.int_shape(input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, size_before[1:3],method='bilinear', align_corners=True))(x)

    x = Reshape((x.shape[1]*x.shape[2],classes))(x)
    x = Activation('softmax', name='Classification')(x)

    model = Model(inputs=input, outputs=x, name='CFAMNet')

    return model

if __name__ == '__main__':
    model = cfamnatt(height=128, width=128, channel=3, classes=6)
    model.summary()