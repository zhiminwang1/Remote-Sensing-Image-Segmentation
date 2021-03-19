from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D, Dropout,Lambda,Concatenate
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D,DepthwiseConv2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf


# 必须要下面这行代码
tf.compat.v1.disable_eager_execution()
print(tf.__version__)

def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def SepConv_BN(inputs, filters, stride=1, kernel_size=3,atrous_rate=1,use_activation=True):

    x = DepthwiseConv2D(kernel_size,stride, dilation_rate=atrous_rate,padding='same', use_bias=False)(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',use_bias=False,kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x

def Conv2d_BN(inputs, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x


def bottleneck_Block(input, out_filters, strides=(1, 1), dilation=(1, 1), with_conv_shortcut=False,name='block'):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal',name=name+'_conv1')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same',
               dilation_rate=dilation, use_bias=False, kernel_initializer='he_normal',name=name+'_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal',name=name+'_conv3')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal',name=name+'_residual')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x

# def bottleneck_Block(input, out_filters, strides=(1, 1), dilation=(1, 1), with_conv_shortcut=False):
#     expansion = 4
#     de_filters = int(out_filters / expansion)

#     x = SepConv_BN(input, de_filters, stride=1, kernel_size=1,atrous_rate=1,use_activation=True)
#     x = SepConv_BN(x, de_filters, stride=strides, kernel_size=3,atrous_rate=dilation,use_activation=True)
#     x = SepConv_BN(x, out_filters, stride=1, kernel_size=1,atrous_rate=1,use_activation=False)

#     if with_conv_shortcut:
#         residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
#         residual = BatchNormalization(axis=3)(residual)
#         x = add([x, residual])
#     else:
#         x = add([x, input])

#     x = Activation('relu')(x)
#     return x

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
    x2 = Activation('relu')(x2)
    x3 = Concatenate()([x2,x])
    y = Conv2D(C,(1,1),padding='same',use_bias=False)(x3)
    y = BatchNormalization(epsilon=1e-3)(y)
    y = Activation('relu')(y)

    return y


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
    b4 = Activation('relu')(b4)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, (h,w),method='bilinear', align_corners=True))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])

    return x


def cfam_resnet(height, width, channel, classes,blocks=[3,4,6,3],atrous_rate=(6, 12, 18)):
    input = Input(shape=(height, width, channel))

    conv1 = Conv2D(64, 7, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal',name='conv1')(input)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',name='maxpool')(conv1)
    
    # conv2_x  1/4
    for i in range(blocks[0]):
        if (i == 0) :
            conv2 = bottleneck_Block(conv1, 256, strides=(1, 1), with_conv_shortcut=True,name='block1_{}'.format(i+1))
        else:
            conv2 = bottleneck_Block(conv2, 256,name='block1_{}'.format(i+1))    

    # conv3_x  1/8
    for i in range(blocks[1]):
        if (i == 0) :
            conv3 = bottleneck_Block(conv2, 512, strides=(2, 2), with_conv_shortcut=True,name='block2_{}'.format(i+1))
        else:
            conv3 = bottleneck_Block(conv3, 512,name='block2_{}'.format(i+1))

    # conv4_x  1/16
    for i in range(blocks[2]):
        if(i == 0):
            conv4 = bottleneck_Block(conv3, 1024, strides=(2, 2), with_conv_shortcut=True,name='block3_{}'.format(i+1))
        else:
            conv4 = bottleneck_Block(conv4, 1024,name='block3_{}'.format(i+1))
    
    # conv5_x  1/16
    for i in range(blocks[3]):
        if(i == 0):
            conv5 = bottleneck_Block(conv4, 2048, strides=(1, 1), dilation=(2, 2), with_conv_shortcut=True,name='block4_{}'.format(i+1))
        else:
            conv5 = bottleneck_Block(conv5, 2048, dilation=(2, 2),name='block4_{}'.format(i+1))
    
    feat_aspp = ASPP_Model(conv5,atrous_rates=atrous_rate)
    aspp = Conv2d_BN(feat_aspp,256,1)

    attention1 = Conv2d_BN(conv5,256,1)
    attention1 = cfam_module(attention1,6,256,128)

    attention2 = Conv2d_BN(conv4,256,1)
    attention2 = cfam_module(attention2,6,256,128)

    x = Concatenate()([aspp,attention1,attention2])
    x = Conv2d_BN(x,128,1)
    skip_size = tf.keras.backend.int_shape(conv2)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, skip_size[1:3],method='bilinear', align_corners=True))(x)

    up1 = Conv2d_BN(conv3,128,1)
    up1 = Lambda(lambda xx: tf.compat.v1.image.resize(xx, skip_size[1:3],method='bilinear', align_corners=True))(up1)
    up2 = Conv2d_BN(conv2,128,1)

    up = Concatenate()([up1,up2])
    up = Conv2d_BN(up,128,1)
    up = cfam_module(up,6,128,64)

    x = Concatenate()([x,up])
    x = Conv2d_BN(x,classes,3)
    size_before = tf.keras.backend.int_shape(input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, size_before[1:3],method='bilinear', align_corners=True))(x)

    x = Reshape((x.shape[1]*x.shape[2],classes))(x)
    x = Activation('softmax', name='Classification')(x)

    model = Model(inputs=input, outputs=x, name='CFAMNet')

    return model

if __name__ == '__main__':
    resnetdic = {'ResNet50':[3,4,6,3],'ResNet101':[3,4,23,3],'ResNet152':[3,8,36,3]}
    model = cfam_resnet(height=128, width=128, channel=3, classes=6,blocks=resnetdic['ResNet152'])
    model.summary()
