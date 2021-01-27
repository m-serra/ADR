import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.merge import add


def BasicBlock(inputs, filters, strides=1, downsample=None):

    residual = inputs

    out = TimeDistributed(Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same',
                                 kernel_initializer='he_uniform'))(inputs)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = TimeDistributed(Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer='he_uniform'))(out)
    out = BatchNormalization()(out)

    if downsample is not None:
        residual = downsample(inputs)

    out = add([out, residual])
    out = Activation('relu')(out)

    return out


def _base_layer(inputs, filters, blocks, in_filters, strides=1):
    expansion = 1
    downsample = None
    if strides != 1 or in_filters != filters * expansion:
        downsample = Sequential()
        downsample.add(TimeDistributed(Conv2D(filters=filters*expansion, kernel_size=1, strides=strides)))
        downsample.add(BatchNormalization())

    out = BasicBlock(inputs, filters, strides, downsample)

    # in_planes = filters * block.expansion

    # for i in range(1, blocks):
    out = BasicBlock(inputs=out, filters=filters)
    out = BasicBlock(inputs=out, filters=filters)

    return out


def resnet18(batch_shape, h_dim, name):

    in_filters = 64
    layer = [2, 2, 2, 2, 2]
    conv1 = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')
    bn1 = BatchNormalization()
    relu = Activation('relu')
    tanh = Activation('tanh')
    maxpool = MaxPooling2D(pool_size=3, strides=2, padding='same')

    conv_out = Conv2D(filters=h_dim, kernel_size=3, padding='same', strides=2)
    bn_out = BatchNormalization()

    x_in = Input(batch_shape=batch_shape)
    x = TimeDistributed(conv1)(x_in)
    x = bn1(x)
    x = relu(x)
    x = TimeDistributed(maxpool)(x)

    x = _base_layer(x, filters=64,  blocks=layer[0], strides=1, in_filters=in_filters)
    x = _base_layer(x, filters=128, blocks=layer[1], strides=2, in_filters=in_filters)
    x = _base_layer(x, filters=256, blocks=layer[2], strides=2, in_filters=in_filters)
    x = _base_layer(x, filters=512, blocks=layer[3], strides=2, in_filters=in_filters)

    x = TimeDistributed(conv_out)(x)
    x = bn_out(x)
    x_out = Lambda(lambda _x: tf.squeeze(tf.squeeze(tanh(_x), axis=2), axis=2))(x)
    print('X_OUT:', x_out)
    model = Model(inputs=x_in, outputs=x_out, name=name)
    return model
