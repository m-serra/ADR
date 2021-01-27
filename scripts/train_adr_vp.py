from __future__ import absolute_import
import os
import tensorflow as tf
from models.encoder_decoder import image_encoder
from models.encoder_decoder import image_decoder, image_decoder_no_skips
from models.encoder_decoder import load_encoder
from models.encoder_decoder import load_decoder
from models.encoder_decoder import load_recurrent_encoder
from models.action_net import load_action_net, load_recurrent_action_net
from models.lstm import simple_lstm, load_lstm
from utils.clr import CyclicLR
from utils.utils import get_data, ModelCheckpoint, NeptuneCallback
from adr import adr_vp_teacher_forcing
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

    frames, actions, states, steps, train_iterator = get_data(dataset='bair', mode='train', batch_size=bs,
                                                              shuffle=shuffle, dataset_dir=dataset_dir,
                                                              sequence_length_train=seq_len)

    _, _, _, val_steps, val_iterator = get_data(dataset='bair', mode='val', batch_size=bs, shuffle=False,
                                                dataset_dir=dataset_dir, sequence_length_test=seq_len)

    gpu_options = tf.GPUOptions(visible_device_list='1')
    config = tf.ConfigProto(gpu_options=gpu_options)

    train_adr_vp(frames,
                 actions,
                 states,
                 hc_dim=128,
                 ha_dim=16,
                 ho_dim=32,
                 za_dim=10,
                 gaussian_a=True,
                 context_frames=2,
                 use_seq_len=12,
                 continue_training=False,
                 epochs=100000,
                 steps=steps,
                 clr_flag=True,
                 base_lr=5e-4,
                 max_lr=2e-3,
                 half_cycle=4,
                 learning_rate=0.001,
                 action_net_units=256,
                 train_iterator=train_iterator,
                 val_iterator=val_iterator,
                 val_steps=val_steps,
                 output_regularizer=None,
                 reg_lambda=0.0,
                 lstm_layers=2,
                 lstm_units=256,
                 lstm_a_layers=1,
                 lstm_a_units=256,
                 save_dir='/home/mandre/adr/trained_models/bair/',
                 ckpt_dir='/home/mandre/adr/trained_models/bair/random_window',
                 ckpt_criteria='train_rec',
                 config=config,
                 ec_load_name='Ec_a_t00243_v0023.h5',
                 a_load_name='A_a_t00243_v0023.h5',
                 da_load_name='D_a_t00243_v0023.h5',
                 la_load_name='L_a_t00243_v0023.h5',
                 do_load_name='',
                 eo_load_name='',
                 do_filename='D_o.h5',
                 eo_filename='Eo.h5',
                 l_filename='L.h5',
                 random_window=True,
                 keep_all=False,
                 neptune_log=True,
                 neptune_ckpt=True,
                 save_model=True,  # --> !!!!!!!!!!!!!!
                 train_eo_do=True)  # --> !!!!!!!!!!!!!!


def train_adr_vp(frames, actions, states, hc_dim, ha_dim, ho_dim, za_dim=10, gaussian_a=False, context_frames=2,
                 use_seq_len=12, epochs=1, steps=1000, clr_flag=False, base_lr=None, max_lr=None, half_cycle=4,
                 learning_rate=0.001, action_net_units=256, train_interator=None, val_iterator=None, val_steps=None,
                 output_regularizer=None, lstm_units=256, lstm_layers=2, lstm_a_layers=1, lstm_a_units=256,
                 save_dir='.', reg_lambda=0.0, ckpt_dir='.', ckpt_criteria='val_rec', config=None,
                 ec_filename='Ec_o.h5', a_filename='A_o.h5', eo_filename='Eo.h5', do_filename='D_o.h5',
                 l_filename='L.h5', la_filename='La_o.h5', da_filename='Da_o.h5', ec_load_name='Ec_a.h5',
                 a_load_name='A_a.h5', da_load_name='D_a.h5', la_load_name='La.h5', continue_training=False,
                 do_load_name='D_o.h5', eo_load_name='Eo.h5', random_window=True, keep_all=False, neptune_log=False,
                 neptune_ckpt=False, save_model=True, train_eo_do=False):

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    bs, seq_len, w, h, c = [int(s) for s in frames.shape]
    a_dim = 0 if actions is None else actions.shape[-1]
    s_dim = 0 if states is None else states.shape[-1]
    za_dim = za_dim if gaussian_a else 0
    La = None
    clbks = []

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    K.set_session(sess)

    # == Instance and load the models
    Ec = load_recurrent_encoder([bs, context_frames, w, h, c], h_dim=hc_dim, ckpt_dir=ckpt_dir,
                                filename=ec_load_name, trainable=False, load_model_state=False)
    Da = load_decoder(batch_shape=[bs, seq_len, hc_dim+ha_dim+za_dim], model_name='Da', ckpt_dir=ckpt_dir,
                      output_channels=3, filename=da_load_name, output_activation='sigmoid', trainable=False,
                      load_model_state=False)

    Do = image_decoder(batch_shape=[bs, seq_len, hc_dim+ho_dim+ha_dim], output_activation='sigmoid',
                       output_channels=6, name='D_o', reg_lambda=reg_lambda, output_initializer='glorot_uniform',
                       output_regularizer=output_regularizer)
    # Do = load_decoder(batch_shape=[bs, seq_len, hc_dim + ha_dim + ho_dim], model_name='Do', ckpt_dir=ckpt_dir,
    #                   output_channels=6, filename=do_load_name, output_activation='sigmoid', trainable=train_eo_do,
    #                   load_model_state=False)

    Eo = image_encoder(batch_shape=[bs, 1, w, h, c*2], h_dim=ho_dim, name='Eo', reg_lambda=reg_lambda)
    # Eo = load_encoder(batch_shape=[bs, seq_len, w, h, c * 2], h_dim=ho_dim, model_name='Eo', ckpt_dir=ckpt_dir,
    #                   filename=eo_load_name, trainable=train_eo_do, load_model_state=False)

    L = simple_lstm(batch_shape=[bs, seq_len, hc_dim + ha_dim*2 + ho_dim], h_dim=ho_dim, n_layers=lstm_layers,
                    units=lstm_units)

    if gaussian_a:
        A = load_action_net(batch_shape=[bs, seq_len, a_dim + s_dim], units=action_net_units, h_dim=ha_dim,
                            ckpt_dir=ckpt_dir, filename=a_load_name, trainable=False, load_model_state=True)
        La = load_lstm(batch_shape=[bs, seq_len, hc_dim + ha_dim], h_dim=za_dim, lstm_units=lstm_a_units,
                       n_layers=lstm_a_layers, ckpt_dir=ckpt_dir, filename=la_load_name, lstm_type='gaussian',
                       trainable=False, load_model_state=True)
    else:
        A = load_recurrent_action_net([bs, seq_len, a_dim + s_dim], action_net_units, ha_dim, ckpt_dir=ckpt_dir,
                                      filename=a_load_name, trainable=False, load_model_state=True)

    ckpt_models = [L]
    # ckpt_models = [Ec, Eo, Da, Do, A, L]
    filenames = [l_filename]
    # filenames = [ec_filename, eo_filename, da_filename, do_filename, a_filename, l_filename]
    # if gaussian_a:
    #     ckpt_models.append(La)
    #     filenames.append(la_filename)
    if train_eo_do:
        ckpt_models.append(Eo)
        ckpt_models.append(Do)
        filenames.append(eo_filename)
        filenames.append(do_filename)

    model = adr_vp_teacher_forcing(frames, actions, states, context_frames, Ec=Ec, Eo=Eo, A=A, Do=Do, Da=Da, L=L, La=La,
                                   gaussian_a=gaussian_a, use_seq_len=use_seq_len, lstm_a_units=lstm_a_units,
                                   lstm_a_layers=lstm_a_layers, lstm_units=lstm_units, lstm_layers=lstm_layers,
                                   learning_rate=learning_rate, random_window=random_window)

    if save_model:
        clbks.append(ModelCheckpoint(models=ckpt_models, criteria=ckpt_criteria, ckpt_dir=save_dir, filenames=filenames,
                                     neptune_ckpt=neptune_ckpt, keep_all=keep_all))  # --> remove neptune flag
    if neptune_log or neptune_ckpt:
        clbks.append(NeptuneCallback(user='m-serra', project_name='adrvp', log=neptune_log, ckpt=neptune_ckpt))
    if clr_flag:
        clbks.append(CyclicLR(model, base_lr, max_lr, step_size=half_cycle*steps))

    model.fit(x=None,
              batch_size=bs,
              epochs=epochs,
              steps_per_epoch=steps,
              callbacks=clbks,
              validation_data=val_iterator,
              validation_steps=val_steps,
              verbose=2)

    return model.history


if __name__ == '__main__':
    main()
