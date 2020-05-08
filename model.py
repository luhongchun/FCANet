from keras.layers import *
from attention_block import *
from keras.backend.common import normalize_data_format
from keras.regularizers import l2
from keras.losses import binary_crossentropy,categorical_crossentropy
from keras.metrics import binary_accuracy,categorical_accuracy
from our_loss import dice_loss,ce_dice_loss,jaccard_loss,ce_jaccard_loss,tversky_loss
from keras.optimizers import Adam
from metrics import dice,iou,pw_prec,pw_sens,pw_spec,pw_recall,pw_precesion
from keras.utils import multi_gpu_model,conv_utils
from keras import Input, Model, initializers, regularizers, constraints
import keras.backend as K
from keras.engine import Layer,InputSpec
from keras.applications import VGG16
from Danet_attention import PAM, CAM

def compile_model(model, num_classes, metrics, loss, lr):
    if isinstance(loss, str):
        if loss in {'ce', 'crossentropy'}:
            loss = binary_crossentropy if num_classes == 1 else categorical_crossentropy
        elif loss in {'iou','jaccard_loss'}:
            loss = jaccard_loss
        elif loss in {'dice_loss','Dice'}:
            loss = dice_loss
        elif loss in {'ce_dice_loss'}:
            loss = ce_dice_loss
        elif loss in {'ce_jaccard_loss'}:
            loss = ce_jaccard_loss
        elif loss in {'tversky_loss'}:
            loss = tversky_loss
        else:
            raise ValueError('unknown loss %s' % loss)

    if isinstance(metrics, str):
        metrics = [metrics, ]

    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'acc':
            metrics[i] = binary_accuracy if num_classes == 1 else categorical_accuracy
        elif metric == 'iou':
            metrics[i] = iou(num_classes)
        elif metric == 'dice':
            metrics[i] = dice(num_classes)
        elif metric == 'pw_precesion':
            metrics[i] = pw_precesion(num_classes)
        elif metric == 'pw_prec':  # pixelwise
            metrics[i] = pw_prec(num_classes)
        elif metric == 'pw_sens':
            metrics[i] = pw_sens(num_classes)
        elif metric == 'pw_spec':
            metrics[i] = pw_spec(num_classes)
        elif metric == 'pw_recall':
            metrics[i] = pw_recall(num_classes)
        else:
            raise ValueError('metric %s not recognized' % metric)

    model.compile(optimizer=Adam(lr=lr),
                  loss=loss,
                  metrics=metrics)
    # model.summary()



def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


#res2net--------------------------------------------------------------------------------------------------------------
def res2net_bottleneck_block(x, f, s=4, expansion=4,dilation=(1, 1)):

    num_channels = int(x._keras_shape[-1])
    input_tensor = x
    # Conv 1x1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(f, 1, kernel_initializer='he_normal', use_bias=False)(x)
    # Conv 3x3
    subset_x = []
    n = f
    w = n // s
    for i in range(s):
        slice_x = Lambda(lambda x: x[..., i * w:(i + 1) * w])(x)
        if i > 1:
            slice_x = Add()([slice_x, subset_x[-1]])
        if i > 0:
            slice_x = BatchNormalization()(slice_x)
            slice_x = Activation('relu')(slice_x)
            slice_x = Conv2D(w, 3,  kernel_initializer='he_normal', padding='same', use_bias=False, dilation_rate = dilation)(slice_x)
        subset_x.append(slice_x)
    x = Concatenate()(subset_x)
    # Conv 1x1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(f * expansion, 1, kernel_initializer='he_normal', use_bias=False)(x)

    # Add
    if num_channels == f * expansion:
        skip = input_tensor
    else:
        skip = input_tensor
        skip = Conv2D(f * expansion, 1, kernel_initializer='he_normal')(skip)
    out = Add()([x, skip])
    return out


def FCANet(pretrained_weights=None,img_input = (256,256,1)):
    input = Input(shape=img_input)
    conv1_1 = Conv2D(64, 7, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    conv1_1 = BatchNormalization(axis=3)(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)
    Attention1_1 = Attention_block(conv1_1, 64 )
    conv1_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1_1)

    conv2_1 = res2net_bottleneck_block(conv1_2, 64, s=4, expansion=4)
    conv2_2 = res2net_bottleneck_block(conv2_1, 64, s=4, expansion=4)
    conv2_3 = res2net_bottleneck_block(conv2_2, 64, s=4, expansion=4)
    Attention2_3 = Attention_block(conv2_3, 256)

    conv3_1 = res2net_bottleneck_block(conv2_3, 128, s=4, expansion=4,)
    conv3_2 = res2net_bottleneck_block(conv3_1, 128, s=4, expansion=4)
    conv3_3 = res2net_bottleneck_block(conv3_2, 128, s=4, expansion=4)
    conv3_4 = res2net_bottleneck_block(conv3_3, 128, s=4, expansion=4)
    conv3_4 = MaxPooling2D()(conv3_4)
    Attention3_4 = Attention_block(conv3_4, 512)
    conv4 =  res2net_bottleneck_block(conv3_4, 256, s=4, expansion=4,dilation=(2, 2))
    for _ in range(22):
        conv4 = res2net_bottleneck_block(conv4, 256, s=4, expansion=4,dilation=(2, 2))

    conv5_1 = res2net_bottleneck_block(conv4, 512, s=4, expansion=4,dilation=(4, 4))
    conv5_2 = res2net_bottleneck_block(conv5_1, 512, s=4, expansion=4,dilation=(4, 4))
    conv5_3 = res2net_bottleneck_block(conv5_2, 512, s=4, expansion=4,dilation=(4, 4))

    reduce_conv5_3 = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(conv5_3)
    reduce_conv5_3 = BatchNormalization(axis=3)(reduce_conv5_3)
    reduce_conv5_3 = Activation('relu')(reduce_conv5_3)

    feature_sum = Attention_block(reduce_conv5_3, 512)
    feature_sum = Dropout(0.5)(feature_sum)

    feature_sum = Conv2d_BN(feature_sum, 512, 3)
    merge7 = concatenate([Attention3_4, feature_sum], axis=3)
    conv7 = Conv2d_BN(merge7, 512, 3)
    conv7 = Conv2d_BN(conv7, 512, 3)

    up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 256, 2)
    merge8 = concatenate([Attention2_3, up8], axis=3)
    conv8 = Conv2d_BN(merge8, 256, 3)
    conv8 = Conv2d_BN(conv8, 256, 3)

    up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 64, 2)
    merge9 = concatenate([Attention1_1, up9], axis=3)
    conv9 = Conv2d_BN(merge9, 64, 3)
    conv9 = Conv2d_BN(conv9, 64, 3)

    up10 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv9), 64, 2)
    conv10 = Conv2d_BN(up10, 64, 3)
    conv10 = Conv2d_BN(conv10, 64, 3)

    conv10 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv10)

    model = Model(inputs=input, outputs=conv10)
    metrics = ['acc', 'pw_precesion', 'iou', 'dice' ]
    # model = multi_gpu_model(model, gpus=2)
    compile_model(model, 1, metrics=metrics, loss='ce_dice_loss', lr=1e-4)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

#Compare the model---------------------------------------------------------------------------------------------------------------
def FCN(img_input = (256,256,1),weight_decay=0.):
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    # classifying layer
    x = Conv2D(1 , (1, 1), kernel_initializer='he_normal',activation='relu', padding='valid', strides=(1, 1),
               kernel_regularizer=l2(weight_decay))(x)

    x = UpSampling2D(size=(32, 32))(x)
    x = Activation('sigmoid')(x)

    model = Model(img_input, x)
    return model



def FCN_8s(input_shape=(256,256,3),
                          num_classes=2,
                          num_conv_filters=4096,
                          use_bias=True,
                          weight_decay=0.,
                          last_activation='sigmoid'  # or e.g. 'sorftmax'
                          ):

    wd = weight_decay
    kr = regularizers.l2
    in1 = Input(shape=input_shape)
    ki = 'glorot_uniform'
    y_pad = input_shape[0] % 32
    x_pad = input_shape[1] % 32
    assert y_pad == 0 and x_pad == 0

    base_model = VGG16(include_top=False, input_tensor=in1, pooling=None)

    pool3 = base_model.layers[-9].output
    pool4 = base_model.layers[-5].output
    pool5 = base_model.layers[-1].output

    relu6 = Conv2D(num_conv_filters, 7,
               activation='relu',
               kernel_regularizer=kr(wd),
               kernel_initializer=ki,
               use_bias=use_bias,
               padding='same', name='fc6_relu6')(pool5)

    drop6 = Dropout(0.5)(relu6)

    relu7 = Conv2D(num_conv_filters, 1,
               activation='relu',
               kernel_regularizer=kr(wd),
               kernel_initializer=ki,
               use_bias=use_bias,
               name='fc7_relu7')(drop6)

    drop7 = Dropout(0.5)(relu7)

    score_fr = Conv2D(num_classes, 1,
                       kernel_regularizer=kr(wd),
                       use_bias=use_bias,
                       name='conv_fc3')(drop7)


    upscore2 = Conv2DTranspose(num_classes, 4,
                               strides=(2, 2),
                               padding='same',
                               kernel_regularizer=kr(wd),
                               kernel_initializer=ki,
                               use_bias=False,
                               name='upscore2')(score_fr)

    score_pool4 = Conv2D(num_classes, 1,
                         kernel_regularizer=kr(wd),
                         use_bias=use_bias)(pool4)

    fuse_pool4 = add([upscore2, score_pool4])

    upscore_pool4 = Conv2DTranspose(num_classes, 4,
                                    strides=(2, 2),
                                    padding='same',
                                    kernel_regularizer=kr(wd),
                                    kernel_initializer=ki,
                                    use_bias=False,
                                    name='upscore_pool4')(fuse_pool4)

    score_pool3 = Conv2D(num_classes, 1, kernel_regularizer=kr(wd), use_bias=use_bias)(pool3)
    fuse_pool3 = add([upscore_pool4, score_pool3])
    upscore8 = Conv2DTranspose(num_classes, 16,
                               strides=(8, 8),
                               padding='same',
                               kernel_regularizer=kr(wd),
                               kernel_initializer=ki,
                               use_bias=False,
                               name='upscore8')(fuse_pool3)


    score = Activation(last_activation)(upscore8)

    model = Model(in1, score)
    model.summary()
    return model



class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, up_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.up_size = up_size

    def call(self, inputs, output_shape=None):

        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.up_size[0],
                    input_shape[2] * self.up_size[1],
                    input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                [[input_shape[0]], [1], [1], [1]],
                axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype='int32'),
                shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.up_size[0],
            mask_shape[2] * self.up_size[1],
            mask_shape[3]
        )



def SegNet(input_shape = (256,256,1), kernel=3, pool_size=(2, 2), output_mode="sigmoid"):
    # encoder
    inputs = Input(shape=input_shape)
    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)
    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")
    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)
    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)
    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)
    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)
    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)
    conv_26 = Convolution2D(1, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape( (input_shape[0] * input_shape[1], 1),
        input_shape=(input_shape[0], input_shape[1], 1), )(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")
    model = Model(inputs=inputs, outputs=outputs)
    return model


class BilinearUpsampling(Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                   int(inputs.shape[2] * self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def xception_downsample_block(x, channels, top_relu=False):
    ##separable conv1
    if top_relu:
        x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ##separable conv2
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ##separable conv3
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_downsample_block(x, channels):
    res = Conv2D(channels, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = xception_downsample_block(x, channels)
    x = add([x, res])
    return x


def xception_block(x, channels):
    ##separable conv1
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv2
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv3
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_block(x, channels):
    res = x
    x = xception_block(x, channels)
    x = add([x, res])
    return x


def aspp(x, input_shape, out_stride):
    b0 = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    b1 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    out_shape = int(input_shape[0] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])
    return x


def deeplabv3plus(pretrained_weights=None,input_shape=(256,256,1), out_stride=16):
    img_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = res_xception_downsample_block(x, 128)

    res = Conv2D(256, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    skip = BatchNormalization()(x)
    x = Activation("relu")(skip)
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = xception_downsample_block(x, 728, top_relu=True)

    for i in range(16):
        x = res_xception_block(x, 728)

    res = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(728, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(2048, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # aspp
    x = aspp(x, input_shape, out_stride)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.9)(x)

    ##decoder
    x = BilinearUpsampling((4, 4))(x)
    dec_skip = Conv2D(48, (1, 1), padding="same", use_bias=False)(skip)
    dec_skip = BatchNormalization()(dec_skip)
    dec_skip = Activation("relu")(dec_skip)
    x = Concatenate()([x, dec_skip])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = BilinearUpsampling((4, 4))(x)
    x= Activation('sigmoid')(x)

    model = Model(inputs=img_input, outputs=x)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


def Unet(img_input = (256,256,1)):
        inputs = Input(img_input)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)
        return model




def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
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


def danet_resnet101(height = 256, width = 256, channel = 1):
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

    # conv4_x  1/16
    conv4_1 = bottleneck_Block(conv3_4, 1024, strides=(1, 1), dilation=(2, 2), with_conv_shortcut=True)
    conv4_2 = bottleneck_Block(conv4_1, 1024, dilation=(2, 2))
    conv4_3 = bottleneck_Block(conv4_2, 1024, dilation=(2, 2))
    conv4_4 = bottleneck_Block(conv4_3, 1024, dilation=(2, 2))
    conv4_5 = bottleneck_Block(conv4_4, 1024, dilation=(2, 2))
    conv4_6 = bottleneck_Block(conv4_5, 1024, dilation=(2, 2))
    conv4_7 = bottleneck_Block(conv4_6, 1024, dilation=(2, 2))
    conv4_8 = bottleneck_Block(conv4_7, 1024, dilation=(2, 2))
    conv4_9 = bottleneck_Block(conv4_8, 1024, dilation=(2, 2))
    conv4_10 = bottleneck_Block(conv4_9, 1024, dilation=(2, 2))
    conv4_11 = bottleneck_Block(conv4_10, 1024, dilation=(2, 2))
    conv4_12 = bottleneck_Block(conv4_11, 1024, dilation=(2, 2))
    conv4_13 = bottleneck_Block(conv4_12, 1024, dilation=(2, 2))
    conv4_14 = bottleneck_Block(conv4_13, 1024, dilation=(2, 2))
    conv4_15 = bottleneck_Block(conv4_14, 1024, dilation=(2, 2))
    conv4_16 = bottleneck_Block(conv4_15, 1024, dilation=(2, 2))
    conv4_17 = bottleneck_Block(conv4_16, 1024, dilation=(2, 2))
    conv4_18 = bottleneck_Block(conv4_17, 1024, dilation=(2, 2))
    conv4_19 = bottleneck_Block(conv4_18, 1024, dilation=(2, 2))
    conv4_20 = bottleneck_Block(conv4_19, 1024, dilation=(2, 2))
    conv4_21 = bottleneck_Block(conv4_20, 1024, dilation=(2, 2))
    conv4_22 = bottleneck_Block(conv4_21, 1024, dilation=(2, 2))
    conv4_23 = bottleneck_Block(conv4_22, 1024, dilation=(2, 2))

    # conv5_x  1/32
    conv5_1 = bottleneck_Block(conv4_23, 2048, strides=(1, 1), dilation=(4, 4), with_conv_shortcut=True)
    conv5_2 = bottleneck_Block(conv5_1, 2048, dilation=(4, 4))
    conv5_3 = bottleneck_Block(conv5_2, 2048, dilation=(4, 4))

    # ATTENTION
    reduce_conv5_3 = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(conv5_3)
    reduce_conv5_3 = BatchNormalization(axis=3)(reduce_conv5_3)
    reduce_conv5_3 = Activation('relu')(reduce_conv5_3)

    pam = PAM()(reduce_conv5_3)
    pam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
    pam = BatchNormalization(axis=3)(pam)
    pam = Activation('relu')(pam)
    pam = Dropout(0.5)(pam)
    pam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)

    cam = CAM()(reduce_conv5_3)
    cam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
    cam = BatchNormalization(axis=3)(cam)
    cam = Activation('relu')(cam)
    cam = Dropout(0.5)(cam)
    cam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)

    feature_sum = add([pam, cam])
    feature_sum = Dropout(0.5)(feature_sum)
    feature_sum = Conv2d_BN(feature_sum, 512, 1)
    merge7 = concatenate([conv3_4, feature_sum], axis=3)
    conv7 = Conv2d_BN(merge7, 512, 3)
    conv7 = Conv2d_BN(conv7, 512, 3)

    up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 256, 2)
    merge8 = concatenate([conv2_3, up8], axis=3)
    conv8 = Conv2d_BN(merge8, 256, 3)
    conv8 = Conv2d_BN(conv8, 256, 3)

    up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 64, 2)
    merge9 = concatenate([conv1_1, up9], axis=3)
    conv9 = Conv2d_BN(merge9, 64, 3)
    conv9 = Conv2d_BN(conv9, 64, 3)

    up10 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv9), 64, 2)
    conv10 = Conv2d_BN(up10, 64, 3)
    conv10 = Conv2d_BN(conv10, 64, 3)

    conv11 = Conv2d_BN(conv10, 1, 1)
    activation = Activation('sigmoid', name='Classification')(conv11)

    model = Model(inputs=input, outputs=activation)
    return model
