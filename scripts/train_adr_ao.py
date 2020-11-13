from __future__ import absolute_import
import os
import numpy as np
import tensorflow as tf
from models.encoder_decoder import image_decoder, load_decoder
from models.encoder_decoder import load_recurrent_encoder, recurrent_image_encoder
from models.action_net import action_net, load_action_net, load_recurrent_action_net, recurrent_action_net
from models.lstm import lstm_gaussian, load_lstm
from adr import adr_ao
from utils.clr import CyclicLR
from utils.utils import get_data, ModelCheckpoint, print_loss, save_gifs
from tensorflow.python.keras.optimizers import Adam

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

    gpu_options = tf.GPUOptions(visible_device_list='1')
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
                        kl_weight=1e-6,  # 1e-6,  # 1e-5,
                        lstm_units=256,
                        lstm_layers=1,
                        epochs=500000,
                        steps=steps,
                        learning_rate=4e-5,
                        ckpt_dir=os.path.join('/home/mandre/adr/trained_models/bair/'),
                        val_steps=1,  # val_steps,  --> !!!
                        ckpt_criteria='train_rec',  # 'val_rec',
                        ec_filename='Ec_a_test_agg.h5',
                        d_filename='D_a_test_agg.h5',
                        a_filename='A_a_test_agg.h5',
                        l_filename='L_a_test_agg.h5',
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

    # ===== Define the sub modules
    if continue_training:
        Ec = load_recurrent_encoder(batch_shape=[bs, context_frames, w, h, c], h_dim=hc_dim, ckpt_dir=ckpt_dir,
                                    filename=ec_load_name, trainable=True, load_model_state=True, name='Ec')
        D = load_decoder(h_dim=hc_dim + ha_dim + z_dim, model_name='D_a', ckpt_dir=ckpt_dir, filename=d_load_name,
                         output_activation='sigmoid', trainable=True, load_model_state=True)
        if gaussian:
            A = load_action_net(batch_shape=[bs, seq_len, a_dim + s_dim], units=a_units, ha_dim=ha_dim,
                                ckpt_dir=ckpt_dir, filename=a_load_name, trainable=True, load_model_state=True)
            L = load_lstm(batch_shape=[bs, seq_len, hc_dim + ha_dim], output_dim=z_dim, lstm_units=lstm_units,
                          n_layers=lstm_layers, ckpt_dir=ckpt_dir, filename=l_load_name,  lstm_type='gaussian',
                          trainable=True, load_model_state=False, name='L_a')
        else:
            A = load_recurrent_action_net(batch_shape=[bs, seq_len, a_dim+s_dim], ha_dim=ha_dim,  ckpt_dir=ckpt_dir,
                                          units=a_units, filename=a_load_name, trainable=True, load_model_state=True)
    else:
        Ec = recurrent_image_encoder(image_shape=[bs, context_frames, w, h, c], output_dim=hc_dim, name='Ec',
                                     conv_lambda=reg_lambda, recurrent_lambda=recurrent_lambda)
        D = image_decoder(h_dim=hc_dim + ha_dim + z_dim, name='D_a', output_activation='sigmoid', reg_lambda=reg_lambda,
                          output_initializer='glorot_uniform', output_regularizer=output_regularizer)
        if gaussian:
            A = action_net(batch_shape=[bs, seq_len, a_dim + s_dim], units=a_units, ha_dim=ha_dim, name='A')
            L = lstm_gaussian(batch_shape=[bs, seq_len, hc_dim + ha_dim], output_dim=z_dim, n_layers=lstm_layers,
                              units=lstm_units, reparameterize=True, reg_lambda=recurrent_lambda, name='L_a')
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
                use_seq_len=use_seq_len,
                learning_rate=learning_rate,
                gaussian=gaussian,
                kl_weight=kl_weight,
                L=L, lstm_units=lstm_units,
                lstm_layers=lstm_layers,
                training=True,
                random_window=random_window,
                reconstruct_random_frame=reconstruct_random_frame)

    for l in ED.layers:
        print(l.name, ' ', l.trainable)

    model_ckpt = ModelCheckpoint(models=ckpt_models, criteria=ckpt_criteria, ckpt_dir=ckpt_dir, filenames=filenames,
                                 neptune_ckpt=neptune_ckpt, keep_all=keep_all)

    # weights = ED.get_layer('A').layers[-1].trainable_weights  # weight tensors
    # gradients = ED.optimizer.get_gradients(ED.total_loss, weights)  # gradient tensors
    # input_tensors = ED.inputs + ED.sample_weights + ED.targets + [K.learning_phase()]
    # input_tensors = ED.inputs
    # get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    # inputs = [None, np.ones(len(x), y, 0]
    inputs = None

    # --> Add cyclic LR
    # if clr_flag:
    #     clbks.append(CyclicLR(ED, base_lr, max_lr, step_size=half_cycle * steps))

    aggressive_flag = True  # --> !!!!!!!!!!
    aggressive_cycles = 1

    for e in range(epochs):
        print('\n Epoch', str(e + 1) + '/' + str(epochs))
        train_loss = np.zeros(len(ED.metrics_names))
        val_ed_loss = np.zeros(len(ED.metrics_names))

        burn_pre_loss = 1e4
        sub_iter = burn_num_examples = burn_cur_loss = 0

        sess.run(train_iterator.initializer)
        sub_iter = 0
        encoder_only_flag = True
        # ===== Train loop
        # for s in range(steps//2):
        for s in range(steps):

            # while aggressive_flag and sub_iter < steps - 1:  # sub_iter < 100:
            # if aggressive_flag and encoder_only_flag and s < steps - 1:  # sub_iter < 100:

            if aggressive_flag and encoder_only_flag:  # sub_iter < 100:

                print('aggressive encoder')
                # ED.get_layer('D_a').trainable = False
                # print('A', ED.get_layer('A').trainable)
                # print('Ec', ED.get_layer('Ec').trainable)
                # print('L_a', ED.get_layer('L_a').trainable)
                # print('D_a', ED.get_layer('D_a').trainable)
                aggressive_loss = ED.train_on_batch(x=None)

                # grads = get_gradients(inputs)

                rec_loss = aggressive_loss[1]
                kl_loss = aggressive_loss[3]
                burn_num_examples += bs
                burn_cur_loss += (rec_loss + kl_weight * kl_loss)

                # if sub_iter % 10 == 0:
                if sub_iter % 10 == -1:
                    burn_cur_loss = burn_cur_loss / burn_num_examples
                    if burn_pre_loss - burn_cur_loss < 0:
                        encoder_only_flag = False  # break
                    burn_pre_loss = burn_cur_loss
                    burn_cur_loss = burn_num_examples = 0
                continue

            if aggressive_flag:
                print('aggressive decoder')
                # print('A', ED.get_layer('A').trainable)
                # print('Ec', ED.get_layer('Ec').trainable)
                # print('L_a', ED.get_layer('L_a').trainable)
                # print('D_a', ED.get_layer('D_a').trainable)
                # ED.get_layer('A').trainable = False
                # ED.get_layer('Ec').trainable = False
                # ED.get_layer('L_a').trainable = False
                # ED.get_layer('D_a').trainable = True
            else:
                print('normal')

            loss = ED.train_on_batch(x=None)

            train_loss = np.add(train_loss, loss)
            sub_iter += 1

        train_loss = np.divide(train_loss, sub_iter)
        print_loss(train_loss, loss_names=ED.metrics_names, title='Train')
        # ED.trainable = True

        if e % 25 == 0:
            x, imgs, mu, logvar = ED.predict(x=None, steps=1)
            save_gifs(sequence=np.clip(x[:bs], a_min=0.0, a_max=1.0), name='pred_a', save_dir=os.path.join('/home/mandre/adr/gifs'))

        # ===== Validation loop
        # for _ in range(val_steps):
        #     val_in = sess.run(val_frames)
            # == Eval encoder-decoder
        #     val_loss = ED.test_on_batch(x=val_in)
        #     val_ed_loss = np.add(val_ed_loss, np.divide(val_loss, val_steps))

        # self.print_loss(val_ed_loss, loss_names=ED.metrics_names, title='Validation')
        model_ckpt.on_epoch_end(epoch=e, logs={'rec_loss': train_loss[1], 'val_rec_loss': 0.0})

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
