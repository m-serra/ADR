import os
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import ConvLSTM2D
from tensorflow.python.keras.layers import LayerNormalization
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.regularizers import l2
from models.lstm import Sample


def base_conv_layer(x, filters, time_distr, kernel_size=4, strides=2, activation=None, use_bias=False,
                    padding='same', kernel_initializer='he_uniform', reg_lambda=0.0):

    conv_2d = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                     kernel_regularizer=l2(reg_lambda), kernel_initializer=kernel_initializer)

    x = TimeDistributed(conv_2d)(x) if time_distr is True else conv_2d(x)

    bn = BatchNormalization()(x)

    layer_output = Activation(activation)(bn) if activation else LeakyReLU(alpha=0.2)(bn)

    return layer_output


def base_conv_transpose_layer(x, filters, time_distr, kernel_size=4, strides=2, activation=None, use_bias=False,
                              padding='same', kernel_initializer='he_uniform', reg_lambda=0.00):

    conv_2d_transpose = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                        use_bias=use_bias, kernel_regularizer=l2(reg_lambda),
                                        kernel_initializer=kernel_initializer)

    x = TimeDistributed(conv_2d_transpose)(x) if time_distr is True else conv_2d_transpose(x)
    bn = BatchNormalization()(x)
    layer_output = Activation(activation)(bn) if activation else LeakyReLU(alpha=0.2)(bn)

    return layer_output


def base_convlstm_layer(x, filters, kernel_size=4, strides=2, padding='same', kernel_initializer='glorot_uniform',
                        reg_lambda=0.0, use_bias=False):

    def layer_norm_tanh(_x):
        _out = LayerNormalization()(_x)
        return Activation('tanh')(_out)

    conv = ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                      activation=layer_norm_tanh, recurrent_activation='hard_sigmoid',
                      kernel_initializer=kernel_initializer, kernel_regularizer=l2(reg_lambda),
                      recurrent_regularizer=l2(reg_lambda),
                      unit_forget_bias=use_bias, recurrent_initializer='orthogonal', return_sequences=True)

    layer_output = conv(x)
    return layer_output


def image_encoder(batch_shape, h_dim, time_distr=True, name=None, kernel_size=4, size=64, reg_lambda=0.0):
    """Add input options: kernel_size, filters, ...
    If the input is 64x64xchannels the output will be 1x1xlatent_dim
    image_shape: [batch_size, seq_len, w, h, c]. Seq len can be passed as None but in that case, if
                 gaussian is True, there will be an error in TimeDistributed(S). Maybe reprogram this to
                 avoid mistakes
    """

    names = [name + '_input', name + '_L_0'] if name else [None] * 2
    _in = Input(batch_shape=batch_shape, name=names[0])

    h1 = base_conv_layer(_in, size,  strides=2, kernel_size=kernel_size, time_distr=time_distr, reg_lambda=reg_lambda)
    h2 = base_conv_layer(h1, size*2, strides=2, kernel_size=kernel_size, time_distr=time_distr, reg_lambda=reg_lambda)
    h3 = base_conv_layer(h2, size*4, strides=2, kernel_size=kernel_size, time_distr=time_distr, reg_lambda=reg_lambda)
    h4 = base_conv_layer(h3, size*8, strides=2, kernel_size=kernel_size, time_distr=time_distr, reg_lambda=reg_lambda)

    h5 = base_conv_layer(h4, filters=h_dim, strides=1, padding='valid', activation='tanh',
                         kernel_size=4, time_distr=time_distr, reg_lambda=reg_lambda)
    h5 = Lambda(lambda x: tf.squeeze(tf.squeeze(x, axis=2), axis=2), name=names[1])(h5)

    encoder = Model(inputs=_in, outputs=[h5, [h1, h2, h3, h4]], name=name)

    return encoder


def recurrent_image_encoder(batch_shape, h_dim, name, kernel_size=4, size=64, conv_initializer='he_uniform',
                            rec_initializer='glorot_uniform', conv_lambda=0.0, recurrent_lambda=0.0, **kwargs):

    _in = Input(batch_shape=batch_shape)

    h1 = base_conv_layer(_in, size, strides=2, kernel_size=kernel_size, reg_lambda=conv_lambda, time_distr=True,
                         kernel_initializer=conv_initializer)

    h2 = base_conv_layer(h1, size*2, strides=2, kernel_size=kernel_size, reg_lambda=conv_lambda, time_distr=True,
                         kernel_initializer=conv_initializer)

    # 16x16 -> 8x8
    h3 = base_convlstm_layer(h2, size*4, strides=2, kernel_size=kernel_size, reg_lambda=recurrent_lambda, use_bias=True,
                             kernel_initializer=rec_initializer)

    # 8x8 -> 4x4
    h4 = base_conv_layer(h3, size*8, strides=2, kernel_size=kernel_size, reg_lambda=conv_lambda, time_distr=True,
                         kernel_initializer=conv_initializer)

    # 4x4 -> 1x1
    h5 = base_convlstm_layer(h4, h_dim, strides=1, kernel_size=4, padding='valid', use_bias=True,
                             kernel_initializer=rec_initializer, reg_lambda=recurrent_lambda)

    h5 = Lambda(lambda x: tf.squeeze(tf.squeeze(x, axis=2), axis=2))(h5)

    encoder = Model(inputs=_in, outputs=[h5, [h1, h2, h3, h4]], name=name)
    return encoder


def image_decoder(batch_shape, name=None, time_distr=True, output_activation='sigmoid', output_channels=3,
                  reg_lambda=0.0, kernel_size=4, size=64, initializer='he_uniform', output_initializer='glorot_uniform',
                  output_regularizer=None, skips_size=64, **kwargs):
    """Add input options: kernel_size, filters, ...
    """

    bs, seq_len = int(batch_shape[0]), int(batch_shape[1])
    z = Input(batch_shape=batch_shape)
    skip_0 = Input(batch_shape=[bs, seq_len, 32, 32, skips_size])
    skip_1 = Input(batch_shape=[bs, seq_len, 16, 16, skips_size*2])
    skip_2 = Input(batch_shape=[bs, seq_len, 8, 8, skips_size*4])
    skip_3 = Input(batch_shape=[bs, seq_len, 4, 4, skips_size*8])
    concat = Lambda(lambda _x: tf.concat(_x, axis=-1))

    _in = Lambda(lambda x_: tf.expand_dims(tf.expand_dims(x_, axis=2), axis=2))(z)
    h1 = base_conv_transpose_layer(_in, filters=size*8, strides=1, padding='valid', time_distr=time_distr,
                                   kernel_size=4, kernel_initializer=initializer, reg_lambda=reg_lambda)

    _in = concat([h1, skip_3])
    h2 = base_conv_transpose_layer(_in, filters=size*4, strides=2, kernel_size=kernel_size, time_distr=time_distr,
                                   reg_lambda=reg_lambda, kernel_initializer=initializer)

    _in = concat([h2, skip_2])
    h3 = base_conv_transpose_layer(_in, filters=size*2, strides=2, kernel_size=kernel_size, time_distr=time_distr,
                                   reg_lambda=reg_lambda, kernel_initializer=initializer,)

    _in = concat([h3, skip_1])
    h4 = base_conv_transpose_layer(_in, filters=size, strides=2, kernel_size=kernel_size, time_distr=time_distr,
                                   reg_lambda=reg_lambda, kernel_initializer=initializer)

    out_conv = Conv2DTranspose(filters=output_channels, kernel_size=kernel_size, strides=2, padding='same',
                               activation=output_activation, kernel_initializer=output_initializer,
                               activity_regularizer=output_regularizer)

    _in = concat([h4, skip_0])
    x = TimeDistributed(out_conv)(_in) if time_distr is True else out_conv(_in)

    decoder = Model(inputs=[z, [skip_0, skip_1, skip_2, skip_3]], outputs=x, name=name)
    return decoder


def image_decoder_no_skips(h_dim, name=None, output_activation='sigmoid', output_channels=3, reg_lambda=0.0,
                           output_initializer='glorot_uniform', output_regularizer=None):
    """Add input options: kernel_size, filters, ...
    """
    time_distr = True
    size = 64

    z = Input(shape=[None, h_dim])

    concat = Lambda(lambda _x: tf.concat(_x, axis=-1))

    _in = Lambda(lambda x_: tf.expand_dims(tf.expand_dims(x_, axis=2), axis=2))(z)

    h1 = base_conv_transpose_layer(_in, filters=size*8, strides=1, padding='valid', time_distr=time_distr,
                                   kernel_size=4, reg_lambda=reg_lambda)

    _in = concat(h1)
    h2 = base_conv_transpose_layer(_in, filters=size*4, strides=2, time_distr=time_distr, reg_lambda=reg_lambda,)

    _in = concat(h2)
    h3 = base_conv_transpose_layer(_in, filters=size*2, strides=2, time_distr=time_distr, reg_lambda=reg_lambda)

    _in = concat(h3)
    h4 = base_conv_transpose_layer(_in, filters=size, strides=2, time_distr=time_distr, reg_lambda=reg_lambda)

    out_conv = Conv2DTranspose(filters=output_channels, kernel_size=5, strides=2, padding='same',
                               kernel_regularizer=l2(0.0), activation=output_activation,
                               activity_regularizer=output_regularizer, kernel_initializer=output_initializer)
    _in = concat(h4)
    x = TimeDistributed(out_conv)(_in) if time_distr is True else out_conv(_in)

    decoder = Model(inputs=z, outputs=x, name=name)
    return decoder


def load_decoder(batch_shape, model_name, ckpt_dir, filename, output_activation='sigmoid', output_channels=3,
                 output_initializer='glorot_uniform', kernel_size=4, size=64, trainable=False, load_model_state=True):
    weight_path = os.path.join(ckpt_dir, filename)

    if load_model_state:
        D = load_model(weight_path)
    else:
        # --> output channels argument should be removed in future
        D = image_decoder(batch_shape=batch_shape, name=model_name, output_activation=output_activation, size=size,
                          kernel_size=kernel_size, output_initializer=output_initializer,
                          output_channels=output_channels)
        D.load_weights(weight_path)

    if trainable is False:
        D.trainable = False
    return D


def load_decoder_no_skips(h_dim, model_name, ckpt_dir, filename, output_activation='sigmoid', trainable=False,
                          load_model_state=True, output_channels=3):
    weight_path = os.path.join(ckpt_dir, filename)

    if load_model_state:
        D = load_model(weight_path)
    else:
        D = image_decoder_no_skips(h_dim=h_dim, name=model_name, output_activation=output_activation,
                                   output_channels=output_channels)
        D.load_weights(weight_path)

    if trainable is False:
        D.trainable = False
    return D


def load_encoder(batch_shape, h_dim, model_name, ckpt_dir, filename,
                 kernel_size=4, trainable=False, reg_lambda=0.0, load_model_state=True):
    weight_path = os.path.join(ckpt_dir, filename)

    if load_model_state:
        E = load_model(weight_path)
    else:
        E = image_encoder(batch_shape=batch_shape, h_dim=h_dim, kernel_size=kernel_size,
                          reg_lambda=reg_lambda, name=model_name)
        E.load_weights(weight_path)

    if trainable is False:
        E.trainable = False

    return E


def load_recurrent_encoder(batch_shape, h_dim, ckpt_dir, filename, size=64, conv_lambda=0.0, recurrent_lambda=0.0,
                           trainable=False, kernel_size=4, name='Ec', load_model_state=True):

    def layer_norm_tanh(_x):
        _out = LayerNormalization()(_x)
        return Activation('tanh')(_out)

    weight_path = os.path.join(ckpt_dir, filename)

    if load_model_state:
        E = load_model(weight_path, custom_objects={'layer_norm_tanh': layer_norm_tanh})
    else:
        E = recurrent_image_encoder(batch_shape=batch_shape, h_dim=h_dim, size=size,
                                    conv_lambda=conv_lambda, recurrent_lambda=recurrent_lambda,
                                    kernel_size=kernel_size, name=name)
        E.load_weights(weight_path)

    if trainable is False:
        E.trainable = False

    return E


def repeat_skips(skips, ntimes):
    _skips = []
    for s in skips:
        _s = Lambda(lambda x: tf.squeeze(x, axis=1))(s)
        _s = [_s] * ntimes
        _s = Lambda(lambda x: tf.stack(x, axis=1))(_s)
        _skips.append(_s)
    return _skips


def slice_skips(skips, start=0, length=1):
    _skips = []
    for s in skips:
        _s = Lambda(lambda _x: tf.slice(_x, (0, start, 0, 0, 0), (-1, length, -1, -1, -1)))(s)
        # _s = tf.slice(s, (0, start, 0, 0, 0), (-1, length, -1, -1, -1))
        _skips.append(_s)
    return _skips


def concat_skips(skips_a, skips_b, axis=0):
    _skips = []
    for a, b in zip(skips_a, skips_b):
        _s = Lambda(lambda _x: tf.concat(_x, axis=axis))([a, b])
        _skips.append(_s)
    return _skips
