import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import LSTMCell
from tensorflow.python.keras.layers import RNN
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import TimeDistributed
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.regularizers import l2


def initial_state_placeholder(units, n_layers, batch_size=None):
    initial_state = []
    for i in range(n_layers):
        if batch_size is None:
            h = Input(shape=[units])
            s = Input(shape=[units])
        else:
            h = Input(batch_shape=[batch_size, units])
            s = Input(batch_shape=[batch_size, units])
        initial_state.append([h, s])
    return initial_state


def lstm_initial_state_zeros(units, n_layers, batch_size=None):
    initial_state = []
    for i in range(n_layers):
        h = Input(tensor=K.constant(0, shape=[batch_size, units]))
        s = Input(tensor=K.constant(0, shape=[batch_size, units]))
        initial_state.append([h, s])
    return initial_state


def lstm_initial_state_zeros_np(units, n_layers, batch_size):
    initial_state = []
    for i in range(n_layers):
        h = np.zeros([batch_size, units], dtype='float32')
        s = np.zeros([batch_size, units], dtype='float32')
        initial_state.append([h, s])
    return initial_state


def simple_lstm(batch_shape, output_dim=10, n_layers=2, units=256, name=None, reg_lambda=0.0):

    def make_cell(lstm_size):
        return LSTMCell(lstm_size, activation='tanh', kernel_initializer='glorot_uniform', unit_forget_bias=False,
                        recurrent_regularizer=l2(reg_lambda))

    # ===== Define the lstm model
    lstm_cells = [make_cell(units) for _ in range(n_layers)]
    lstm = RNN(lstm_cells, return_sequences=True, return_state=True)
    embed_net = Dense(units=units, activation='linear', use_bias=False)  # --> !!!
    output_net = Dense(units=output_dim, activation='tanh')

    _in = Input(batch_shape=[batch_shape[0], None, batch_shape[-1]])
    initial_state = initial_state_placeholder(units, n_layers, batch_size=batch_shape[0])

    embed = TimeDistributed(embed_net)(_in)
    embed = BatchNormalization()(embed)  # --> !!!!!!
    out = lstm(embed, initial_state=initial_state)
    predictions, state = out[0], out[1:]
    predictions = TimeDistributed(output_net)(predictions)

    model = Model(inputs=[_in, initial_state], outputs=[predictions, state], name=name)

    return model


def lstm_gaussian(batch_shape, output_dim=10, n_layers=2, units=256, reparameterize=False, name=None, reg_lambda=0.0):
    def make_cell(lstm_size):
        return LSTMCell(lstm_size, activation='tanh', kernel_initializer='he_normal',
                        recurrent_regularizer=l2(reg_lambda))

    # ===== Define the lstm model
    lstm_cells = [make_cell(units) for _ in range(n_layers)]
    lstm = RNN(lstm_cells, return_sequences=True, return_state=True, name='lstm_model')
    embed_net = Dense(units=units, activation='linear')
    sample = Sample(output_dim=output_dim, reparameterization_flag=reparameterize)
    _in = Input(batch_shape=[batch_shape[0], None, batch_shape[-1]])
    initial_state = initial_state_placeholder(units, n_layers, batch_size=batch_shape[0])

    embed = TimeDistributed(embed_net)(_in)
    out = lstm(embed, initial_state=initial_state)
    h, state = out[0], out[1:]
    # z, mu, logvar = sample(h)
    z, mu, logvar = TimeDistributed(sample)(h)

    model = Model(inputs=[_in, initial_state], outputs=[z, mu, logvar, state], name=name)

    return model


def load_lstm(batch_shape, output_dim, lstm_units, n_layers, ckpt_dir, filename,
              lstm_type='simple', trainable=False, load_model_state=True, name='L'):
    weight_path = os.path.join(ckpt_dir, filename)

    if load_model_state:
        # this only allows output dim to be the predefined value
        custom_objects = {'Sample': Sample} if lstm_type == 'gaussian' else None
        lstm = load_model(weight_path, custom_objects=custom_objects)
    else:
        assert lstm_type in ['simple', 'gaussian'], 'Argument lstm_type must be \'simple\' or \'gaussian\''

        if lstm_type == 'simple':
            lstm = simple_lstm(batch_shape=batch_shape, output_dim=output_dim, n_layers=n_layers, units=lstm_units,
                               name=name)
        else:
            lstm = lstm_gaussian(batch_shape=batch_shape, output_dim=output_dim, n_layers=n_layers, units=lstm_units,
                                 reparameterize=trainable, name=name)
        lstm.load_weights(weight_path)

    if trainable is False:
        lstm.trainable = False

    return lstm


class Sample(layers.Layer):

    def __init__(self, output_dim=10, reparameterization_flag=True, **kwargs):
        self.output_dim = output_dim
        super(Sample, self).__init__(**kwargs)
        self.reparameterization_flag = reparameterization_flag

        self.mu_net = Dense(units=self.output_dim, activation='linear')
        self.logvar_net = Dense(units=self.output_dim, activation='linear')

    def call(self, inputs, **kwargs):

        h = inputs

        mu = self.mu_net(h)
        logvar = self.logvar_net(h)

        if self.reparameterization_flag:
            epsilon = K.random_normal(shape=[self.output_dim], mean=0.0, stddev=1.0)
            z = mu + K.exp(0.5 * logvar) * epsilon
        else:
            z = tf.sigmoid(K.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0))

        return z, mu, logvar
