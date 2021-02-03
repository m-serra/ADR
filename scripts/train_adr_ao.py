from __future__ import absolute_import
import os
import neptune
import tensorflow as tf
from adr import adr_ao
from adr import get_sub_model
from utils.clr import CyclicLR
from utils.utils import get_data
from utils.utils import ModelCheckpoint
from utils.utils import SaveGifsCallback
from utils.utils import NeptuneCallback
from utils.utils import EvaluateCallback
from tensorflow.python.keras.regularizers import l2
import tensorflow.python.keras.backend as K

tf.logging.set_verbosity(tf.logging.ERROR)

best_loss = 9999
best_train_loss = 9999
best_val_loss = 9999
best_train_epoch = 0
best_val_epoch = 0


def main():

    bs = 32
    use_seq_len = 12
    seq_len = 30
    shuffle = True
    dataset_dir = '/media/Data/datasets/bair/softmotion30_44k/'
    # dataset_dir = '/media/data/mserra/bair/softmotion30_44k/'

    frames, actions, states, steps, train_iterator = get_data(dataset='bair', mode='train', batch_size=bs,
                                                              shuffle=shuffle, dataset_dir=dataset_dir,
                                                              sequence_length_train=seq_len,
                                                              sequence_length_test=use_seq_len)

    _, _, _, val_steps, val_iterator = get_data(dataset='bair', mode='val', batch_size=bs, shuffle=False,
                                                dataset_dir=dataset_dir, sequence_length_train=seq_len,
                                                sequence_length_test=use_seq_len)

    gpu_options = tf.GPUOptions(visible_device_list='1')
    config = tf.ConfigProto(gpu_options=gpu_options)

    hist = train_adr_ao(frames,
                        actions=actions,
                        states=states,
                        context_frames=2,
                        use_seq_len=use_seq_len,
                        continue_training=False,
                        config=config,
                        clr_flag=True,  # --> !!!
                        base_lr=4e-6,  # 1e-5,  # According to plot: 4e-6  # If gaussian True: 1e-5
                        max_lr=2e-4,  # 8e-5,   # According to plot: 1e-4  # If gaussian True: 8e-5
                        half_cycle=4,
                        hc_dim=128,
                        ha_dim=16,
                        reg_lambda=1e-4,  # 1e-4,  # 3e-5,  # 3e-5,  # 1e-4,
                        output_regularizer=l2(0.0),
                        recurrent_lambda=1e-5,
                        gaussian=True,
                        z_dim=10,
                        # kl_weight=1e-6,  # 1e-6,  # 1e-5,
                        kl_weight=5e-7,  # 1e-6,  # 1e-5,
                        lstm_units=256,
                        lstm_layers=1,
                        epochs=500000,
                        steps=steps,
                        val_steps=val_steps,
                        learning_rate=4e-5,
                        ckpt_dir=os.path.join(os.path.expanduser('~/'), 'adr/trained_models/bair/18t3v'),
                        ckpt_criteria='val_rec',
                        ec_filename='Ec_a_test.h5',
                        d_filename='D_a_test.h5',
                        a_filename='A_a_test.h5',
                        l_filename='L_a_test.h5',
                        ec_load_name='Ec_a.h5',
                        d_load_name='D_a_test.h5',
                        a_load_name='A_a_test.h5',
                        l_load_name='L_a_test.h5',
                        neptune_log=True,
                        neptune_ckpt=False,
                        train_iterator=train_iterator,
                        val_iterator=val_iterator,
                        random_window=True,
                        save_model=True,
                        reconstruct_random_frame=False,
                        keep_all=True)  # --> !!!!!

    return hist


def train_adr_ao(frames, actions, states=None, context_frames=2, hc_dim=128, ha_dim=16, epochs=1, clr_flag=False,
                 base_lr=None, max_lr=None, continue_training=False, reg_lambda=0.0, recurrent_lambda=0.0,
                 output_regularizer=None, steps=1000, learning_rate=0.001, a_units=256, gaussian=False, z_dim=10,
                 kl_weight=0.1, lstm_units=256, lstm_layers=2, config=None, half_cycle=4, val_steps=None, ckpt_dir='.',
                 ckpt_criteria='val_rec', ec_filename='Ec_a.h5', d_filename='D_a.h5', a_filename='A_a.h5',
                 l_filename='L_a.h5', ec_load_name='Ec_a.h5', d_load_name='D_a.h5', a_load_name='A_a.h5',
                 l_load_name='L_a.h5', neptune_ckpt=False, neptune_log=False, train_iterator=None, val_iterator=None,
                 reconstruct_random_frame=False, random_window=True, keep_all=False, use_seq_len=12, save_model=True):

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    bs, seq_len, w, h, c = [int(s) for s in frames.shape]

    assert context_frames > 1, 'context frames must be greater or equal than 1'
    z_dim = 0 if gaussian is False else z_dim
    a_dim = actions.shape[-1] if actions is not None else 0
    s_dim = states.shape[-1] if states is not None else 0
    clbks = []

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    K.set_session(sess)

    # ===== Define the sub models
    a_name = 'A' if gaussian else 'rA'

    # Remove the regularization parameters that are not used anymore
    Ec = get_sub_model(name='Ec', batch_shape=[bs, context_frames, w, h, c], h_dim=hc_dim, ckpt_dir=ckpt_dir,
                       filename=ec_load_name, trainable=True, load_model_state=continue_training,
                       load_flag=continue_training, conv_lambda=reg_lambda, recurrent_lambda=recurrent_lambda)

    D = get_sub_model(name='Da', batch_shape=[bs, use_seq_len, hc_dim + ha_dim + z_dim], h_dim=None, ckpt_dir=ckpt_dir,
                      filename=d_load_name, trainable=True, load_model_state=continue_training,
                      load_flag=continue_training, reg_lambda=reg_lambda, output_regularizer=output_regularizer)

    A = get_sub_model(name=a_name, batch_shape=[bs, use_seq_len, a_dim + s_dim], h_dim=ha_dim, ckpt_dir=ckpt_dir,
                      filename=a_load_name, trainable=True, load_model_state=continue_training,
                      load_flag=continue_training, units=a_units, dense_lambda=reg_lambda,
                      recurrent_lambda=recurrent_lambda)

    L = get_sub_model(name='La', batch_shape=[bs, use_seq_len, hc_dim + ha_dim], h_dim=z_dim, ckpt_dir=ckpt_dir,
                      filename=l_load_name, trainable=True, load_model_state=continue_training,
                      load_flag=continue_training, units=lstm_units, n_layers=lstm_layers, lstm_type='gaussian',
                      reparameterize=True, reg_lambda=recurrent_lambda)

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
                L=L,
                use_seq_len=use_seq_len,
                learning_rate=learning_rate,
                gaussian=gaussian,
                kl_weight=kl_weight,
                lstm_units=lstm_units,
                lstm_layers=lstm_layers,
                training=True,
                random_window=random_window,
                reconstruct_random_frame=reconstruct_random_frame)

    # print(len(ED._collected_trainable_weights))
    # print(len(E._collected_trainable_weights))
    # print(len(C._collected_trainable_weights))

    save_gifs_flag = True
    if save_model:
        clbks.append(ModelCheckpoint(models=ckpt_models, criteria=ckpt_criteria, ckpt_dir=ckpt_dir, filenames=filenames,
                                     neptune_ckpt=neptune_ckpt, keep_all=keep_all))
    if neptune_log or neptune_ckpt:
        clbks.append(NeptuneCallback(user='m-serra', project_name='video-prediction', log=neptune_log, ckpt=neptune_ckpt))
    if save_gifs_flag:
        clbks.append(SaveGifsCallback(period=25, iterator=val_iterator,
                                      ckpt_dir=os.path.join(os.path.expanduser('~/'), 'adr/gifs'), name='pred2', bs=bs))
    if clr_flag:
        clbks.append(CyclicLR(ED, base_lr, max_lr, step_size=half_cycle * steps))

    eval_flag = True
    if eval_flag:
        clbks.append(EvaluateCallback(model=ED, iterator=val_iterator, steps=val_steps, period=25))

    # def KLD(_mu, _logvar):
    #     return -0.5 * np.mean(1 + _logvar - np.power(_mu, 2) - np.exp(_logvar), axis=0, keepdims=True)
    #
    # if save_kl_flag:
    #     _KLD = KLD(mu, logvar)
    #     _KLD /= 2.0
    #     # plt.figure()
    #     plt.bar(np.arange(z_dim), np.mean(_KLD.squeeze(), axis=0))
    #     plt.savefig(os.path.join(os.path.expanduser('~/'), 'adr/gifs/kld_aggressive.png'))  # --> !!!

    ED.fit(x=train_iterator,
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
