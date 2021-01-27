import os
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.regularizers import l2
from models.lstm import Sample


def action_net(batch_shape, units, h_dim, name='A',  **kwargs):

    """
    seq_len can be passed as None if the length of the sequence is not known but then Sample won't work as
    time distributed
    """

    _in = Input(batch_shape=[batch_shape[0], None, batch_shape[-1]])
    # _in = Input(batch_shape=[batch_shape[0], None, batch_shape[-1]])

    # == Layer 1
    ha = TimeDistributed(Dense(units=units, activation='linear', kernel_initializer='he_uniform', use_bias=False))(_in)
    ha = BatchNormalization()(ha)
    ha = LeakyReLU(alpha=0.2)(ha)

    # == Layer 2
    ha = TimeDistributed(Dense(units=units, activation='linear', kernel_initializer='glorot_uniform'))(ha)

    ha = BatchNormalization()(ha)
    ha = LeakyReLU(alpha=0.2)(ha)
    # ha = LSTM(units=256, kernel_initializer='he_normal')(ha)
    ha = Dense(units=h_dim, activation='tanh', kernel_initializer='glorot_uniform')(ha)
    model = Model(inputs=_in, outputs=ha, name=name)

    return model


def recurrent_action_net(batch_shape, units, h_dim, name='A', initializer='he_uniform',
                         output_initializer='glorot_uniform', dense_lambda=0.0,
                         recurrent_lambda=0.0, **kwargs):

    """
    seq_len can be passed as None if the length of the sequence is not known but then Sample won't work as
    time distributed
    """

    _in = Input(batch_shape=[batch_shape[0], None, batch_shape[-1]])

    # == Layer 1
    ha = TimeDistributed(Dense(units=units, activation='linear', kernel_initializer=initializer, use_bias=False,
                               kernel_regularizer=l2(dense_lambda)))(_in)
    ha = BatchNormalization()(ha)
    ha = LeakyReLU(alpha=0.2)(ha)

    # == Layer 2
    ha = TimeDistributed(Dense(units=units, activation='linear', kernel_initializer=initializer, use_bias=False,
                               kernel_regularizer=l2(dense_lambda)))(ha)
    ha = BatchNormalization()(ha)
    ha = LeakyReLU(alpha=0.2)(ha)

    ha = LSTM(units=units, return_sequences=True, kernel_initializer='glorot_uniform',
              kernel_regularizer=l2(recurrent_lambda), recurrent_regularizer=l2(recurrent_lambda))(ha)
    ha = TimeDistributed(Dense(units=h_dim, activation='tanh', kernel_initializer=output_initializer,
                               kernel_regularizer=l2(dense_lambda)))(ha)

    model = Model(inputs=_in, outputs=ha, name=name)

    return model


def load_action_net(batch_shape, units, h_dim, ckpt_dir, filename, trainable=False, name='A', load_model_state=True):
    weight_path = os.path.join(ckpt_dir, filename)

    if load_model_state:
        A = load_model(weight_path)
        A._name = name
    else:
        A = action_net(batch_shape=batch_shape, units=units, h_dim=h_dim, name=name)
        A.load_weights(weight_path)

    if trainable is False:
        A.trainable = False
    return A


def load_recurrent_action_net(batch_shape, units, h_dim, ckpt_dir, filename, trainable=False, name='A',
                              load_model_state=True):
    weight_path = os.path.join(ckpt_dir, filename)

    if load_model_state:
        A = load_model(weight_path)
        A._name = name
    else:
        A = recurrent_action_net(batch_shape=batch_shape, units=units, h_dim=h_dim, name=name)
        A.load_weights(weight_path)

    if trainable is False:
        A.trainable = False
    return A
