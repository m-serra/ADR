from __future__ import absolute_import
import os
import tensorflow as tf
from models.encoder_decoder import image_decoder, load_decoder
from models.encoder_decoder import load_recurrent_encoder, recurrent_image_encoder
from models.action_net import action_net, load_action_net, load_recurrent_action_net, recurrent_action_net
from models.lstm import lstm_gaussian, load_lstm
from adr import adr_ao
from utils.clr import CyclicLR
from utils.utils import get_data, ModelCheckpoint

from tensorflow.python.keras.regularizers import l2
import tensorflow.python.keras.backend as K

import neptune
tf.logging.set_verbosity(tf.logging.ERROR)

best_loss = 9999
best_train_loss = 9999
best_val_loss = 9999
best_train_epoch = 0
best_val_epoch = 0


def main():

    bs = 32
    seq_len = 30
    dataset_dir = '/media/Data/datasets/bair/softmotion30_44k/'

    frames, actions, states, steps, _ = get_data(dataset='bair', mode='train', batch_size=bs, shuffle=True,
                                                 dataset_dir=dataset_dir, sequence_length_train=seq_len)

    _, _, _, val_steps, val_iterator = get_data(dataset='bair', mode='val', batch_size=bs, shuffle=False,
                                                dataset_dir=dataset_dir, sequence_length_test=seq_len)

    gpu_options = tf.GPUOptions(visible_device_list='0')
    config = tf.ConfigProto(gpu_options=gpu_options)

    hist = train_adr_ao(frames,
                        actions=actions,
                        states=states,
                        context_frames=2,
                        continue_training=False,
                        config=config,
                        clr_flag=True,
                        base_lr=1e-5,  # 1e-5,  # According to plot: 4e-6  # If gaussian True: 1e-5
                        max_lr=8e-5,  # 8e-5,   # According to plot: 1e-4  # If gaussian True: 8e-5
                        half_cycle=4,  # 4
                        hc_dim=128,
                        ha_dim=16,
                        reg_lambda=1e-4,  # 3e-5,  # 3e-5,  # 1e-4,
                        output_regularizer=l2(0.0),
                        recurrent_lambda=1e-5,
                        gaussian=True,
                        z_dim=10,
                        kl_weight=1e-6,  # 1e-6,  # 1e-5,
                        lstm_units=256,
                        lstm_layers=1,
                        epochs=500,
                        steps=steps,
                        learning_rate=4e-5,
                        ckpt_dir=os.path.join('/home/mandre/adr/trained_models/bair/random_window'),
                        val_steps=val_steps,
                        ckpt_criteria='val_rec',
                        ec_filename='Ec_a_random_window.h5',
                        d_filename='D_a_random_window.h5',
                        a_filename='A_a_random_window.h5',
                        l_filename='L_a_random_window.h5',
                        ec_load_name='Ec_a.h5',
                        d_load_name='D_a.h5',
                        a_load_name='A_a.h5',
                        l_load_name='L_a.h5',
                        neptune_log=False,
                        neptune_ckpt=False,
                        val_iterator=val_iterator,
                        random_window=True,
                        reconstruct_random_frame=False,
                        keep_all=True)


def train_adr_ao(frames, actions, states=None, context_frames=3, hc_dim=128, ha_dim=16, epochs=1, clr_flag=False,
                 base_lr=None, max_lr=None, continue_training=False, reg_lambda=0.0, recurrent_lambda=0.0,
                 output_regularizer=None, steps=1000, learning_rate=0.001, a_units=256, gaussian=False, z_dim=10,
                 kl_weight=0.1, lstm_units=256, lstm_layers=2, config=None, half_cycle=4, val_steps=None, ckpt_dir='.',
                 ckpt_criteria='val_rec', ec_filename='Ec_a.h5', d_filename='D_a.h5', a_filename='A_a.h5',
                 l_filename='L_a.h5', ec_load_name='Ec_a.h5', d_load_name='D_a.h5', a_load_name='A_a.h5',
                 l_load_name='L_a.h5', neptune_ckpt=False, neptune_log=False, val_iterator=None,
                 reconstruct_random_frame=False, random_window=True, keep_all=False):

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    bs, seq_len, w, h, c = [int(s) for s in frames.shape]

    assert context_frames > 1, 'context frames must be greater or equal than 1'
    z_dim = 0 if gaussian is False else z_dim
    a_dim = actions.shape[-1] if actions is not None else 0
    s_dim = states.shape[-1] if states is not None else 0
    L = None

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    K.set_session(sess)

    if neptune_ckpt or neptune_log:
        neptune.init('m-serra/video-prediction')
        neptune.create_experiment(name='train_a stochastic sample single epsilon')

    # ===== Define the sub modules
    if continue_training:
        Ec = load_recurrent_encoder(batch_shape=[bs, context_frames, w, h, c], h_dim=hc_dim, ckpt_dir=ckpt_dir,
                                    filename=ec_load_name, trainable=True, load_model_state=True)
        D = load_decoder(h_dim=hc_dim + ha_dim + z_dim, model_name='D', ckpt_dir=ckpt_dir, filename=d_load_name,
                         output_activation='sigmoid', trainable=True, load_model_state=True)
        if gaussian:
            A = load_action_net(batch_shape=[bs, seq_len, a_dim + s_dim], units=a_units, ha_dim=ha_dim,
                                ckpt_dir=ckpt_dir, filename=a_load_name, trainable=True, load_model_state=True)
            L = load_lstm(batch_shape=[bs, seq_len, hc_dim + ha_dim], output_dim=z_dim, lstm_units=lstm_units,
                          n_layers=lstm_layers, ckpt_dir=ckpt_dir, filename=l_load_name,  lstm_type='gaussian',
                          trainable=True, load_model_state=False)
        else:
            A = load_recurrent_action_net(batch_shape=[bs, seq_len, a_dim+s_dim], ha_dim=ha_dim,  ckpt_dir=ckpt_dir,
                                          units=a_units, filename=a_load_name, trainable=True, load_model_state=True)
    else:
        Ec = recurrent_image_encoder(image_shape=[bs, context_frames, w, h, c], output_dim=hc_dim, name='Ec',
                                     conv_lambda=reg_lambda, recurrent_lambda=recurrent_lambda)
        D = image_decoder(h_dim=hc_dim + ha_dim + z_dim, name='D_a', output_activation='sigmoid', reg_lambda=reg_lambda,
                          output_initializer='glorot_uniform', output_regularizer=output_regularizer)
        if gaussian:
            A = action_net(batch_shape=[bs, seq_len, a_dim + s_dim], units=a_units, ha_dim=ha_dim)
            L = lstm_gaussian(batch_shape=[bs, seq_len, hc_dim + ha_dim], output_dim=z_dim, n_layers=lstm_layers,
                              units=lstm_units, reparameterize=True, reg_lambda=recurrent_lambda)
        else:
            A = recurrent_action_net(batch_shape=[bs, seq_len, a_dim + s_dim], units=a_units, ha_dim=ha_dim,
                                     dense_lambda=reg_lambda, recurrent_lambda=recurrent_lambda, name='A')

    ckpt_models = [Ec, D, A]
    filenames = [ec_filename, d_filename, a_filename]
    if gaussian:
        ckpt_models.append(L)
        filenames.append(l_filename)

    ED = adr_ao(frames,
                actions,
                states,
                context_frames,
                Ec=Ec,
                A=A,
                D=D,
                learning_rate=learning_rate,
                gaussian=gaussian,
                kl_weight=kl_weight,
                L=L, lstm_units=lstm_units,
                lstm_layers=lstm_layers,
                training=True,
                random_window=random_window,
                reconstruct_random_frame=reconstruct_random_frame)

    clbks = [ModelCheckpoint(models=ckpt_models, criteria=ckpt_criteria, ckpt_dir=ckpt_dir, filenames=filenames,
                             neptune_ckpt=neptune_ckpt, keep_all=keep_all)]
    if clr_flag:
        clbks.append(CyclicLR(ED, base_lr, max_lr, step_size=half_cycle * steps))

    ED.fit(x=None,
           batch_size=bs,
           epochs=epochs,
           steps_per_epoch=steps,
           callbacks=clbks,
           validation_data=val_iterator,
           validation_steps=val_steps,
           verbose=2)

    return ED.history


if __name__ == '__main__':
    main()
