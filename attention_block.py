from keras import backend as K
from keras.activations import sigmoid
from keras.regularizers import l2
from keras.models import *
from keras.layers import *

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,  # 商取整
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    channel_feature = Add()([avg_pool, max_pool])
    channel_feature = Activation('sigmoid')(channel_feature)
    if K.image_data_format() == "channels_first":
        channel_feature = Permute((3, 1, 2))(channel_feature)
    return multiply([input_feature, channel_feature])


def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        spatial_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        spatial_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(spatial_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(spatial_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    spatial_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert spatial_feature._keras_shape[-1] == 1


    if K.image_data_format() == "channels_first":
        spatial_feature = Permute((3, 1, 2))(spatial_feature)

    return multiply([input_feature, spatial_feature])

def Attention_block(input,filter):
    input = Conv2D(filter,kernel_size=(3,3),strides=1,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(input)
    channel = channel_attention(input)
    channel = Conv2D(filter, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(channel)
    channel = BatchNormalization(axis=3)(channel)
    channel = Activation('relu')(channel)

    spatial = spatial_attention(input)
    spatial = Conv2D(filter, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(spatial)
    spatial = BatchNormalization(axis=3)(spatial)
    spatial = Activation('relu')(spatial)
    return add([channel,spatial])


