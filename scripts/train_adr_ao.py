from __future__ import absolute_import
import os
import numpy as np
import tensorflow as tf
from adr import adr_ao, get_sub_model
from utils.clr import CyclicLR
from utils.utils import get_data, ModelCheckpoint, print_loss, save_gifs
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils.utils import generate_grid, plot_multiple
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
    shuffle = True
    dataset_dir = '/media/Data/datasets/bair/softmotion30_44k/'

    # when doing aggressive training initialize dataset at each epoch to reshuffle data
    frames, actions, states, steps, train_iterator = get_data(dataset='bair', mode='train', batch_size=bs,
                                                              shuffle=shuffle, dataset_dir=dataset_dir,
                                                              sequence_length_train=seq_len, initializable=True)

    _, _, _, val_steps, val_iterator = get_data(dataset='bair', mode='val', batch_size=bs, shuffle=False,
                                                dataset_dir=dataset_dir, sequence_length_test=seq_len)

    gpu_options = tf.GPUOptions(visible_device_list='0')
    config = tf.ConfigProto(gpu_options=gpu_options)

    hist = train_adr_ao(frames,
                        actions=actions,
                        states=states,
                        context_frames=2,
                        use_seq_len=12,
                        continue_training=False,
                        config=config,
                        clr_flag=True,
                        base_lr=1e-5,  # 1e-5,  # According to plot: 4e-6  # If gaussian True: 1e-5
                        max_lr=8e-5,  # 8e-5,   # According to plot: 1e-4  # If gaussian True: 8e-5
                        half_cycle=4,
                        hc_dim=128,
                        ha_dim=16,
                        reg_lambda=1e-4,  # 3e-5,  # 3e-5,  # 1e-4,
                        output_regularizer=l2(0.0),
                        recurrent_lambda=1e-5,
                        gaussian=True,
                        z_dim=10,
                        # kl_weight=1e-6,  # 1e-6,  # 1e-5,
                        kl_weight=5e-7,  # 1e-6,  # 1e-5,    # --> !!!!!!!!!!!
                        lstm_units=256,
                        lstm_layers=1,
                        epochs=500000,
                        steps=steps,
                        learning_rate=4e-5,
                        ckpt_dir=os.path.join('/home/mandre/adr/trained_models/bair/'),
                        val_steps=1,  # val_steps,  --> !!!
                        ckpt_criteria='train_rec',  # 'val_rec',
                        ec_filename='Ec_a_test.h5',
                        d_filename='D_a_test.h5',
                        a_filename='A_a_test.h5',
                        l_filename='L_a_test.h5',
                        ec_load_name='Ec_a.h5',
                        d_load_name='D_a.h5',
                        a_load_name='A_a.h5',
                        l_load_name='L_a.h5',
                        neptune_log=False,
                        neptune_ckpt=False,
                        train_iterator=train_iterator,
                        val_iterator=val_iterator,
                        random_window=True,
                        reconstruct_random_frame=False,
                        keep_all=False)

    return hist


def train_adr_ao(frames, actions, states=None, context_frames=3, hc_dim=128, ha_dim=16, epochs=1, clr_flag=False,
                 base_lr=None, max_lr=None, continue_training=False, reg_lambda=0.0, recurrent_lambda=0.0,
                 output_regularizer=None, steps=1000, learning_rate=0.001, a_units=256, gaussian=False, z_dim=10,
                 kl_weight=0.1, lstm_units=256, lstm_layers=2, config=None, half_cycle=4, val_steps=None, ckpt_dir='.',
                 ckpt_criteria='val_rec', ec_filename='Ec_a.h5', d_filename='D_a.h5', a_filename='A_a.h5',
                 l_filename='L_a.h5', ec_load_name='Ec_a.h5', d_load_name='D_a.h5', a_load_name='A_a.h5',
                 l_load_name='L_a.h5', neptune_ckpt=False, neptune_log=False, train_iterator=None, val_iterator=None,
                 reconstruct_random_frame=False, random_window=True, keep_all=False, use_seq_len=12):

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

    # ===== Define the sub models
    a_name = 'A' if gaussian else 'rA'

    # Remove the regularization parameters that are not used anymmore
    Ec = get_sub_model(name='Ec', batch_shape=[bs, context_frames, w, h, c], h_dim=hc_dim, ckpt_dir=ckpt_dir,
                       filename=ec_load_name, trainable=True, load_model_state=continue_training,
                       load_flag=continue_training, conv_lambda=reg_lambda, recurrent_lambda=recurrent_lambda)

    D = get_sub_model(name='Da', batch_shape=[bs, use_seq_len, hc_dim + ha_dim + z_dim], h_dim=None, ckpt_dir=ckpt_dir,
                      filename=d_load_name, trainable=True, load_model_state=continue_training,
                      load_flag=continue_training, reg_lambda=reg_lambda, output_regularizer=output_regularizer)

    A = get_sub_model(name=a_name, batch_shape=[bs, seq_len, a_dim + s_dim], h_dim=ha_dim, ckpt_dir=ckpt_dir,
                      filename=a_load_name, trainable=True, load_model_state=continue_training,
                      load_flag=continue_training, units=a_units, dense_lambda=reg_lambda,
                      recurrent_lambda=recurrent_lambda)

    L = get_sub_model(name='La', batch_shape=[bs, seq_len, hc_dim + ha_dim], h_dim=z_dim, ckpt_dir=ckpt_dir,
                      filename=l_load_name, trainable=True, load_model_state=continue_training,
                      load_flag=continue_training, units=lstm_units, n_layers=lstm_layers, lstm_type='gaussian',
                      reparameterize=True, reg_lambda=recurrent_lambda)

    ckpt_models = [Ec, D, A]
    filenames = [ec_filename, d_filename, a_filename]

    if gaussian:
        ckpt_models.append(L)
        filenames.append(l_filename)

    ED, E = adr_ao(frames,
                   actions,
                   states,
                   context_frames,
                   Ec=Ec,
                   A=A,
                   D=D,
                   use_seq_len=use_seq_len,
                   learning_rate=learning_rate,
                   gaussian=gaussian,
                   kl_weight=kl_weight,
                   L=L, lstm_units=lstm_units,
                   lstm_layers=lstm_layers,
                   training=True,
                   random_window=random_window,
                   reconstruct_random_frame=reconstruct_random_frame)

    print(len(ED._collected_trainable_weights))
    print(len(E._collected_trainable_weights))

    model_ckpt = ModelCheckpoint(models=ckpt_models, criteria=ckpt_criteria, ckpt_dir=ckpt_dir, filenames=filenames,
                                 neptune_ckpt=neptune_ckpt, keep_all=keep_all)

    # --> Add cyclic LR
    # if clr_flag:
    #     clbks.append(CyclicLR(ED, base_lr, max_lr, step_size=half_cycle * steps))

    aggressive_flag = False  # --> !!!
    aggressive_cycles = 10  # the average cycles according to the paper. Implement stopping condition later
    iter_ = 0
    zmin, zmax, dz = -20, 20, 0.1  #  check why they were using these numbers
    grid_z = generate_grid(zmin, zmax, dz, ndim=1)

    sess.run(train_iterator.initializer)
    plot_data = sess.run(frames)

    for e in range(epochs):
        print('\n Epoch', str(e + 1) + '/' + str(epochs))
        train_loss = np.zeros(len(ED.metrics_names))
        val_ed_loss = np.zeros(len(ED.metrics_names))

        num_examples = 0
        curr_loss = 0
        prev_loss = 999
        sess.run(train_iterator.initializer)
        ed_iter = 0
        encoder_only_flag = True

        # ===== Train loop
        for s in range(steps):

            if aggressive_flag and encoder_only_flag:  # sub_iter < 100:

                print('aggressive encoder')
                aggressive_loss = E.train_on_batch(x=None)

                rec_loss = aggressive_loss[1]
                kl_loss = aggressive_loss[3]
                num_examples += bs
                curr_loss += (rec_loss + kl_weight * kl_loss)

                if s % 10 == 0:
                    curr_loss = curr_loss / num_examples
                    if prev_loss - curr_loss < 0:
                        encoder_only_flag = False  # break
                    prev_loss = curr_loss
                    curr_loss = num_examples = 0
            else:
                # print('normal')
                loss = ED.train_on_batch(x=None)
                train_loss = np.add(train_loss, loss)
                ed_iter += 1

            iter_ += 1

        train_loss = np.divide(train_loss, ed_iter)
        print_loss(train_loss, loss_names=ED.metrics_names, title='Train')

        save_gifs_flag = False  # --> !!!
        save_kl_flag = False

        if e % 25 == 0 and (save_gifs_flag or save_kl_flag):

            x, imgs, mu, logvar = ED.predict(x=None, steps=1)

            if save_gifs_flag is True:
                save_gifs(sequence=np.clip(x[:bs], a_min=0.0, a_max=1.0), name='pred_a_not_agg',
                          save_dir=os.path.join('/home/mandre/adr/gifs'))

            def KLD(_mu, _logvar):
                return -0.5 * np.mean(1 + _logvar - np.power(_mu, 2) - np.exp(_logvar), axis=0, keepdims=True)

            if save_kl_flag:
                _KLD = KLD(mu, logvar)
                _KLD /= 2.0
                plt.figure()
                plt.bar(np.arange(z_dim), np.mean(_KLD.squeeze(), axis=0))
                plt.savefig('/home/mandre/adr/gifs/kld_not_aggressive2.png')  # --> !!!

        # ===== Validation loop
        # for _ in range(val_steps):
        #     val_in = sess.run(val_frames)
            # == Eval encoder-decoder
        #     val_loss = ED.test_on_batch(x=val_in)
        #     val_ed_loss = np.add(val_ed_loss, np.divide(val_loss, val_steps))

        # self.print_loss(val_ed_loss, loss_names=ED.metrics_names, title='Validation')
        # if save_gifs_flag:  # --> DELETE THIS
        model_ckpt.on_epoch_end(epoch=e, logs={'rec_loss': train_loss[1], 'val_rec_loss': 0.0})

        plot_multiple(model=ED, plot_data=plot_data, grid_z=grid_z, iter_=iter_, ckpt_dir=ckpt_dir, nz=z_dim,
                      aggressive=aggressive_flag)

        if e >= aggressive_cycles - 1:
            aggressive_flag = False

    return 2

    # ED.fit(x=None,
    #        batch_size=bs,
    #        epochs=epochs,
    #        steps_per_epoch=steps,
    #        callbacks=clbks,
    #        validation_data=val_iterator,
    #        validation_steps=val_steps,
    #        verbose=2)

    # return ED.history


if __name__ == '__main__':
    main()
