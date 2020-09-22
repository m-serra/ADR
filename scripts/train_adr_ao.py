from __future__ import absolute_import
import os
import tensorflow as tf
from models.encoder_decoder import image_decoder,load_decoder
from models.encoder_decoder import load_recurrent_encoder, recurrent_image_encoder
from models.action_net import action_net, load_action_net, load_recurrent_action_net, recurrent_action_net
from models.lstm import lstm_gaussian, load_lstm
from adr import adr_ao
from utils.clr import CyclicLR
from termcolor import colored
from tensorflow.python.keras.regularizers import l2
import tensorflow.python.keras.backend as K
from data_readers.bair_data_reader import BairDataReader
from data_readers.google_push_data_reader import GooglePushDataReader
import neptune
tf.logging.set_verbosity(tf.logging.ERROR)
from robonet.datasets import load_metadata
from robonet.datasets.robonet_dataset import RoboNetDataset


best_loss = 9999
best_train_loss = 9999
best_val_loss = 9999
best_train_epoch = 0
best_val_epoch = 0


def main():

    bs = 32
    seq_len = 12

    frames, actions, states, steps, \
    _, _, _, val_steps, val_iterator = get_data(dataset='bair', batch_size=bs, shuffle=True,
                                                dataset_dir='/media/Data/datasets/bair/softmotion30_44k/',
                                                sequence_length_train=seq_len, sequence_length_test=seq_len)

    gpu_options = tf.GPUOptions(visible_device_list='1')
    config = tf.ConfigProto(gpu_options=gpu_options)

    hist = train_autoencoder_A(frames,
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
                               steps_per_epoch=steps,
                               learning_rate=4e-5,
                               ckpt_dir=os.path.join('../trained_models/bair'),
                               val_steps_per_epoch=val_steps,
                               ckpt_criteria='val_rec',
                               ec_filename='Ec_a_test.h5',
                               d_filename='D_a_test.h5',
                               a_filename='A_a_test.h5',
                               l_filename='L_a_test.h5',
                               ec_load_name='Ec_a_hc0_t003198_v003210.h5',
                               d_load_name='D_a_hc0_t003198_v003210.h5',
                               a_load_name='A_a_hc0_t003198_v003210.h5',
                               l_load_name='L_a_hc0_t003198_v003210.h5',
                               neptune_log=False,
                               neptune_ckpt=False,
                               val_iterator=val_iterator)


def train_autoencoder_A(frames, actions, states=None, context_frames=3, hc_dim=128, ha_dim=16,
                        epochs=1, continue_training=False, clr_flag=False, base_lr=None, max_lr=None,
                        size=64, reg_lambda=0.0, recurrent_lambda=0.0, output_regularizer=None,
                        steps_per_epoch=1000, learning_rate=0.001, a_units=256,
                        gaussian=False, z_dim=10, kl_weight=0.1, lstm_units=256, lstm_layers=2,
                        config=None, half_cycle=8, val_steps_per_epoch=None, ckpt_dir='.',
                        ckpt_criteria='val_rec', ec_filename='Ec_a.h5', d_filename='D_a.h5', a_filename='A_a.h5',
                        l_filename='L_a.h5', ec_load_name='Ec_a.h5', d_load_name='D_a.h5', a_load_name='A_a.h5',
                        l_load_name='L_a.h5', neptune_ckpt=False, neptune_log=False, val_iterator=None):

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
                training=True)

    clbks = [ModelCheckpoint(models=ckpt_models, criteria=ckpt_criteria, ckpt_dir=ckpt_dir, filenames=filenames,
                             neptune_ckpt=neptune_ckpt)]
    if clr_flag:
        clbks.append(CyclicLR(ED, base_lr, max_lr, step_size=half_cycle * steps_per_epoch))

    ED.fit(x=None,
           batch_size=bs,
           epochs=epochs,
           steps_per_epoch=steps_per_epoch,
           callbacks=clbks,
           validation_data=val_iterator,
           validation_steps=val_steps_per_epoch)

    return ED.history


class ModelCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, models, criteria, ckpt_dir, filenames, neptune_ckpt=False):
        super(tf.keras.callbacks.Callback, self).__init__()
        super().__init__()
        self.models = models
        self.criteria = criteria
        self.ckpt_dir = ckpt_dir
        self.filenames = filenames
        self.best_loss = 9999
        self.best_train_loss = 9999
        self.best_train_epoch = 0
        self.best_val_loss = 9999
        self.best_val_epoch = 0
        self.saver = None
        self.neptune_ckpt = neptune_ckpt

        if type(self.models) is not list:
            self.models = [self.models]
        if type(self.filenames) is not list:
            self.filenames = [self.filenames]

        assert len(models) == len(filenames), 'models and filenames must have the same length'
        assert criteria in ['train_rec', 'val_rec'], 'criteria must be either train_rec or val_rec'

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('rec_loss')
        val_loss = logs.get('val_rec_loss')
        self.model_checkpoint(train_loss, val_loss, epoch)

    def model_checkpoint(self, train_loss, val_loss, epoch):

        new_best_train, new_best_val = False, False

        # criteria_map = {'train_rec': train_loss[1], 'val_rec': val_loss[1]}
        criteria_map = {'train_rec': train_loss, 'val_rec': val_loss}

        loss = criteria_map.get(self.criteria)

        if loss < self.best_loss:
            for m, f in zip(self.models, self.filenames):
                tf.keras.models.save_model(m, os.path.join(self.ckpt_dir, f))
                # m.save_weights(os.path.join(self.ckpt_dir, f))
                if self.neptune_ckpt:
                    neptune.log_artifact(os.path.join(self.ckpt_dir, f))
            self.best_loss = loss

        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            self.best_train_epoch = epoch + 1
            new_best_train = True

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = epoch + 1
            new_best_val = True

        if new_best_train:
            print(colored('Best train loss: %.7f, epoch  %d' %
                          (self.best_train_loss, self.best_train_epoch), 'magenta'))
        # else:
        #     print('Best train loss: %.7f, epoch  %d' % (self.best_train_loss, self.best_train_epoch))
        if new_best_val:
            print(colored('Best val loss: %.7f, epoch %d' % (self.best_val_loss, self.best_val_epoch), 'green'))
        # else:
        #     print('Best val loss: %.7f, epoch %d' % (self.best_val_loss, self.best_val_epoch))
        return


def get_data(dataset, dataset_dir, batch_size=32, sequence_length_train=12, sequence_length_test=12, shuffle=True):

    assert dataset in ['bair', 'google', 'robonet']

    if dataset == 'bair':
        d = BairDataReader(dataset_dir=dataset_dir,  # '/media/Data/datasets/bair/softmotion30_44k/',
                           batch_size=batch_size,
                           use_state=1,
                           sequence_length_train=sequence_length_train,
                           sequence_length_test=sequence_length_test,
                           shuffle=shuffle,
                           batch_repeat=1)
    elif dataset == 'google':
        d = GooglePushDataReader(dataset_dir=dataset_dir,  # '/media/Data/datasets/google_push/push/',
                                 batch_size=batch_size,
                                 sequence_length_train=sequence_length_train,
                                 sequence_length_test=sequence_length_test,
                                 shuffle=shuffle,
                                 train_dir_name='push_train',
                                 test_dir_name='push_train',
                                 batch_repeat=1)
    elif dataset == 'robonet':
        train_database = load_metadata('/home/mandre/RoboNet/hdf5/train')
        val_database = load_metadata('/home/mandre/RoboNet/hdf5/val2')
        train_database = train_database[train_database['robot'] == 'fetch']
        d_train = RoboNetDataset(batch_size=batch_size, dataset_files_or_metadata=train_database,
                                 hparams={'img_size': [64, 64], 'target_adim': 2, 'target_sdim': 3})
        d_val = RoboNetDataset(batch_size=batch_size, dataset_files_or_metadata=val_database,
                               hparams={'img_size': [64, 64], 'target_adim': 2, 'target_sdim': 3})

    """
    d.train_filenames = ['/media/Data/datasets/bair/softmotion30_44k/train/traj_10174_to_10429.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_1024_to_1279.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_10430_to_10685.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_10686_to_10941.tfrecords',
                         # '/media/Data/datasets/bair/softmotion30_44k/train/traj_10942_to_11197.tfrecords',
                         # '/media/Data/datasets/bair/softmotion30_44k/train/traj_11198_to_11453.tfrecords',
                         # '/media/Data/datasets/bair/softmotion30_44k/train/traj_11454_to_11709.tfrecords',
                         # '/media/Data/datasets/bair/softmotion30_44k/train/traj_11710_to_11965.tfrecords',
                         # '/media/Data/datasets/bair/softmotion30_44k/train/traj_11966_to_12221.tfrecords',
                         # '/media/Data/datasets/bair/softmotion30_44k/train/traj_12222_to_12477.tfrecords',
                         # '/media/Data/datasets/bair/softmotion30_44k/train/traj_12478_to_12733.tfrecords',
                         # '/media/Data/datasets/bair/softmotion30_44k/train/traj_12734_to_12989.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_1280_to_1535.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_12990_to_13245.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_13341_to_13596.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_13597_to_13852.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_13853_to_14108.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_14109_to_14364.tfrecords']

    d.val_filenames = ['/media/Data/datasets/bair/softmotion30_44k/train/traj_5983_to_6238.tfrecords',
                       '/media/Data/datasets/bair/softmotion30_44k/train/traj_6239_to_6494.tfrecords',
                       '/media/Data/datasets/bair/softmotion30_44k/train/traj_6495_to_6750.tfrecords']
    """

    if dataset == 'robonet':
        frames = tf.squeeze(d_train['images'])  # images, states, and actions are from paired
        actions = d_train['actions']
        states = d_train['states']
        val_frames = tf.squeeze(d_val['images'])
        val_actions = d_val['actions']
        val_states = d_val['states']
        steps = 545
        val_steps = 545
    else:
        steps = d.num_examples_per_epoch('train') // d.batch_size
        val_steps = d.num_examples_per_epoch('val') // d.batch_size

        train_iterator = d.build_tf_iterator(mode='train')
        val_iterator = d.build_tf_iterator(mode='val')

        input_get_next_op = train_iterator.get_next()
        val_input_get_next_op = val_iterator.get_next()

        frames = input_get_next_op['images']
        actions = input_get_next_op['actions'][:, :, :4]
        states = input_get_next_op['states'][:, :, :3]
        val_frames = val_input_get_next_op['images']
        val_actions = val_input_get_next_op['actions'][:, :, :4]
        val_states = val_input_get_next_op['states'][:, :, :3]

    return frames, actions, states, steps, val_frames, val_actions, val_states, val_steps, val_iterator


if __name__ == '__main__':
    main()
