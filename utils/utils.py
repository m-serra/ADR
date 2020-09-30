import os
import tensorflow as tf
from termcolor import colored
from data_readers.bair_data_reader import BairDataReader
from data_readers.google_push_data_reader import GooglePushDataReader
from robonet.datasets import load_metadata
from robonet.datasets.robonet_dataset import RoboNetDataset


def get_data(dataset, mode, dataset_dir, batch_size=32, sequence_length_train=12, sequence_length_test=12,
             shuffle=True):

    assert dataset in ['bair', 'google', 'robonet']
    assert mode in ['train', 'val', 'test']

    if dataset == 'bair':
        d = BairDataReader(dataset_dir=dataset_dir,
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
        steps = d.num_examples_per_epoch(mode) // d.batch_size
        iterator = d.build_tf_iterator(mode=mode)
        input_get_next_op = iterator.get_next()
        frames = input_get_next_op['images']
        actions = input_get_next_op['actions'][:, :, :4]
        states = input_get_next_op['states'][:, :, :3]

    return frames, actions, states, steps, iterator


class ModelCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, models, criteria, ckpt_dir, filenames, neptune_ckpt=False, keep_all=False):
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
        self.keep_all = keep_all

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

        # if loss < self.best_loss:
        if loss < self.best_loss or train_loss < self.best_train_loss:  # --> !!!!
            for m, f in zip(self.models, self.filenames):
                if self.keep_all:
                    f = f.replace('.h5', '') + '_t' + str(train_loss).replace('0.', '') + \
                        '_v' + str(val_loss).replace('0.', '') + '.h5'
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
        if new_best_val:
            print(colored('Best val loss: %.7f, epoch %d' % (self.best_val_loss, self.best_val_epoch), 'green'))
        return

