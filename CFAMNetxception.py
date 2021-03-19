from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D, Dropout,Lambda,Concatenate,Add,add
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization,GlobalAveragePooling2D,ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D,DepthwiseConv2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf



def Conv2d_BN(inputs, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x

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


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              stride=stride if i == 2 else 1,
                              atrous_rate=rate,
                              use_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs

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


def cfam_xception(height, width, channel, classes):
    input = Input(shape=(height, width, channel))

    entry_block3_stride = 2
    middle_block_rate = 1
    exit_block_rates = (1, 2)
    atrous_rates = (6, 12, 18)

    x = Conv2D(32, (3, 3), strides=(2, 2),
                name='entry_flow_conv1_1', use_bias=False, padding='same')(input)
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation(tf.nn.relu)(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = Activation(tf.nn.relu)(x)

    conv1 = x

    x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)

    conv2 = x

    x = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                skip_connection_type='conv', stride=2,
                                depth_activation=False)
    conv3 = x

    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
    conv4 = x

    for i in range(16):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)

    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)
    conv5 = x

    feat_aspp = ASPP_Model(conv5,atrous_rates=(6, 12, 18))
    
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
    # model.load_weights(weights_path, by_name=True)

    return model

if __name__ == '__main__':
    model = cfam_xception(height=128, width=128, channel=3, classes=6)
    model.summary()