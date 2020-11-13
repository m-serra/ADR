from __future__ import absolute_import
import os
import tensorflow as tf
from models.encoder_decoder import image_encoder
from models.encoder_decoder import image_decoder
from models.encoder_decoder import load_encoder
from models.encoder_decoder import load_decoder
from models.encoder_decoder import load_recurrent_encoder
from models.action_net import load_action_net, load_recurrent_action_net
from models.lstm import load_lstm
from utils.clr import CyclicLR
from utils.utils import get_data, ModelCheckpoint
from adr import adr
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
    shuffle = True
    dataset_dir = '/media/Data/datasets/bair/softmotion30_44k/'

    frames, actions, states, steps, _ = get_data(dataset='bair', mode='train', batch_size=bs, shuffle=shuffle,
                                                 dataset_dir=dataset_dir, sequence_length_train=seq_len)

    _, _, _, val_steps, val_iterator = get_data(dataset='bair', mode='val', batch_size=bs, shuffle=False,
                                                dataset_dir=dataset_dir, sequence_length_test=seq_len)

    gpu_options = tf.GPUOptions(visible_device_list='1')
    config = tf.ConfigProto(gpu_options=gpu_options)

    hist = train_adr(frames,
                     actions,
                     states,
                     hc_dim=128,
                     ha_dim=16,
                     ho_dim=32,
                     za_dim=10,
                     gaussian_a=True,
                     context_frames=2,
                     use_seq_len=12,
                     epochs=10000,
                     steps=steps,
                     continue_training=False,
                     clr_flag=True,
                     base_lr=6e-5,
                     max_lr=4e-4,
                     half_cycle=4,
                     learning_rate=4e-5,
                     action_net_units=256,
                     val_iterator=val_iterator,
                     val_steps=val_steps,
                     reg_lambda=0.0,
                     output_regularizer=None,
                     lstm_units=256,
                     lstm_layers=1,
                     neptune_log=False,
                     neptune_ckpt=False,
                     save_dir='/home/mandre/adr/trained_models/bair',
                     ckpt_dir='/home/mandre/adr/trained_models/bair/random_window2',
                     ckpt_criteria='val_rec',
                     config=config,
                     ec_filename='Ec_o.h5',
                     a_filename='A_o.h5',
                     eo_filename='Eo.h5',
                     do_filename='D_o.h5',
                     la_filename='La_o.h5',
                     da_filename='Da_o.h5',
                     ec_load_name='Ec_a_t003693856_v0032159444.h5',
                     a_load_name='A_a_t003693856_v0032159444.h5',
                     da_load_name='D_a_t003693856_v0032159444.h5',
                     la_load_name='L_a_t003693856_v0032159444.h5',
                     do_load_name='',  # only needed if continue_training = True
                     eo_load_name='',
                     keep_all=False,
                     random_window=True,
                     reconstruct_random_frame=False)


def train_adr(frames, actions, states, hc_dim, ha_dim, ho_dim, za_dim=10, gaussian_a=False, context_frames=2, epochs=1,
              steps=1000, use_seq_len=12, clr_flag=False, base_lr=None, max_lr=None, half_cycle=4, learning_rate=0.001,
              action_net_units=256, val_iterator=None, val_steps=None, output_regularizer=None, lstm_units=256,
              lstm_layers=1, neptune_log=False, neptune_ckpt=False, save_dir='.', reg_lambda=0.0, ckpt_dir='.',
              ckpt_criteria='val_rec', config=None, ec_filename='Ec_o.h5', a_filename='A_o.h5', eo_filename='Eo.h5',
              do_filename='D_o.h5', la_filename='La_o.h5', da_filename='Da_o.h5', ec_load_name='Ec_a.h5',
              a_load_name='A_a.h5', da_load_name='D_a.h5', la_load_name='La.h5', continue_training=False,
              do_load_name='D_o.h5', eo_load_name='Eo.h5', random_window=True, keep_all=False,
              reconstruct_random_frame=False):

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    bs, seq_len, w, h, c = [int(s) for s in frames.shape]
    a_dim = 0 if actions is None else actions.shape[-1]
    s_dim = 0 if states is None else states.shape[-1]
    za_dim = za_dim if gaussian_a else 0
    La = None

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    K.set_session(sess)

    if neptune_log or neptune_ckpt:
        neptune.init('m-serra/stochastic-objects')
        neptune.create_experiment(name='train_o stochastic')

    # == Instance and load the models
    Ec = load_recurrent_encoder([bs, context_frames, w, h, c], h_dim=hc_dim, ckpt_dir=ckpt_dir,
                                filename=ec_load_name, trainable=False, load_model_state=True)
    Da = load_decoder(h_dim=hc_dim + ha_dim + za_dim, model_name='Da', ckpt_dir=ckpt_dir, output_channels=3,
                      filename=da_load_name, output_activation='sigmoid', trainable=False, load_model_state=True)

    if continue_training:
        Eo = load_encoder(batch_shape=[bs, seq_len, w, h, 2 * c], h_dim=ho_dim, model_name='Eo', ckpt_dir=ckpt_dir,
                          filename=eo_load_name, trainable=True, load_model_state=True)
        Do = load_decoder(h_dim=hc_dim + ha_dim + ho_dim, model_name='Do', ckpt_dir=ckpt_dir,
                          output_channels=6, filename=do_load_name, output_activation='sigmoid', trainable=True,
                          load_model_state=True)
    else:
        Do = image_decoder(h_dim=hc_dim+ho_dim+ha_dim, output_activation='sigmoid', output_channels=6,
                           name='D_o', reg_lambda=reg_lambda, output_initializer='glorot_uniform',
                           output_regularizer=output_regularizer)
        Eo = image_encoder(image_shape=[bs, 1, w, h, c*2], output_dim=ho_dim, name='Eo', reg_lambda=reg_lambda)

    if gaussian_a:
        A = load_action_net(batch_shape=[bs, seq_len, a_dim + s_dim], units=action_net_units, ha_dim=ha_dim,
                            ckpt_dir=ckpt_dir, filename=a_load_name, trainable=False, load_model_state=True)
        La = load_lstm(batch_shape=[bs, seq_len, hc_dim + ha_dim], output_dim=za_dim,
                       lstm_units=lstm_units, n_layers=lstm_layers, ckpt_dir=ckpt_dir, filename=la_load_name,
                       lstm_type='gaussian', trainable=False, load_model_state=False)  # --> !!!
    else:
        A = load_recurrent_action_net([bs, seq_len, a_dim + s_dim], action_net_units, ha_dim, ckpt_dir=ckpt_dir,
                                      filename=a_load_name, trainable=False, load_model_state=True)

    ckpt_models = [Ec, Eo, Da, Do,  A]
    filenames = [ec_filename, eo_filename, da_filename, do_filename, a_filename]
    if gaussian_a:
        ckpt_models.append(La)
        filenames.append(la_filename)

    adr_model = adr(frames, actions, states, context_frames, Ec=Ec, Eo=Eo, A=A, Da=Da, Do=Do, La=La,
                    use_seq_len=use_seq_len, gaussian_a=gaussian_a, lstm_units=lstm_units, learning_rate=learning_rate,
                    random_window=random_window, reconstruct_random_frame=reconstruct_random_frame)

    clbks = [ModelCheckpoint(models=ckpt_models, criteria=ckpt_criteria, ckpt_dir=save_dir, filenames=filenames,
                             neptune_ckpt=neptune_ckpt, keep_all=keep_all)]
    if clr_flag:
        clbks.append(CyclicLR(adr_model, base_lr, max_lr, step_size=half_cycle * steps))

    adr_model.fit(x=None,
                  batch_size=bs,
                  epochs=epochs,
                  steps_per_epoch=steps,
                  callbacks=clbks,
                  validation_data=val_iterator,
                  validation_steps=val_steps,
                  verbose=2)

    return adr_model.history


if __name__ == '__main__':
    main()
